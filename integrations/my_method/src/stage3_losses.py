from __future__ import annotations

import json
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
import torch.nn.functional as F
from PIL import Image

from .model_utils import infer_input_device, uses_qwen_vision_utils
from .utils import ensure_dir, logger, resolve_path, save_json


try:
    from qwen_vl_utils import process_vision_info
except ImportError:  # pragma: no cover
    process_vision_info = None


def _move_inputs(inputs: Any, device: Optional[torch.device]) -> Any:
    if device is None:
        return inputs
    try:
        return inputs.to(device)
    except Exception:
        for key, value in list(inputs.items()):
            if torch.is_tensor(value):
                inputs[key] = value.to(device)
        return inputs


def _messages(
    image_path: str,
    instruction: str,
    response: Optional[str] = None,
    *,
    max_pixels: Optional[int] = 200704,
) -> List[Dict[str, Any]]:
    image_item = {"type": "image", "image": str(image_path)}
    if max_pixels is not None:
        image_item["max_pixels"] = int(max_pixels)
    msgs = [
        {
            "role": "user",
            "content": [
                image_item,
                {"type": "text", "text": instruction or ""},
            ],
        }
    ]
    if response is not None:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": response or ""}]})
    return msgs


def _processor_inputs(processor, messages: List[Dict[str, Any]], text: str, device: Optional[torch.device]):
    if uses_qwen_vision_utils(processor):
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    else:
        image_paths = [
            item["image"]
            for msg in messages
            for item in msg.get("content", [])
            if isinstance(item, dict) and item.get("type") == "image"
        ]
        images = [Image.open(path).convert("RGB") for path in image_paths]
        inputs = processor(text=[text], images=images, padding=True, return_tensors="pt")
    return _move_inputs(inputs, device)


def prepare_prompt_inputs(
    processor,
    sample: Dict[str, Any],
    device: Optional[torch.device],
    *,
    max_pixels: Optional[int] = 200704,
):
    messages = _messages(sample["image_path"], sample["instruction"], None, max_pixels=max_pixels)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return _processor_inputs(processor, messages, text, device)


def prepare_supervised_inputs(
    processor,
    sample: Dict[str, Any],
    response: str,
    device: Optional[torch.device],
    *,
    debug_path: Optional[Path] = None,
    max_pixels: Optional[int] = 200704,
):
    prompt_messages = _messages(sample["image_path"], sample["instruction"], None, max_pixels=max_pixels)
    full_messages = _messages(sample["image_path"], sample["instruction"], response, max_pixels=max_pixels)
    prompt_text = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    full_text = processor.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
    prompt_inputs = _processor_inputs(processor, prompt_messages, prompt_text, device)
    full_inputs = _processor_inputs(processor, full_messages, full_text, device)
    labels = full_inputs["input_ids"].clone()
    prompt_len = int(prompt_inputs["input_ids"].shape[1])
    labels[:, : min(prompt_len, labels.shape[1])] = -100
    if "attention_mask" in full_inputs:
        labels = labels.masked_fill(full_inputs["attention_mask"] == 0, -100)
    full_inputs["labels"] = labels
    if debug_path is not None and not debug_path.exists():
        save_json(
            {
                "input_length": int(full_inputs["input_ids"].shape[1]),
                "prompt_length": prompt_len,
                "label_count": int((labels != -100).sum().item()),
                "sample_id": sample.get("sample_id"),
            },
            debug_path,
        )
    return full_inputs


def last_token_hidden(outputs, inputs: Dict[str, torch.Tensor], layer: int) -> torch.Tensor:
    hidden = outputs.hidden_states[layer]
    mask = inputs.get("attention_mask")
    if mask is None:
        pos = hidden.shape[1] - 1
    else:
        pos = int(mask[0].nonzero()[-1].item())
    return hidden[0, pos, :].float()


@contextmanager
def adapter_disabled(model):
    if hasattr(model, "disable_adapter"):
        with model.disable_adapter():
            yield
    else:
        raise RuntimeError("PEFT model does not support disable_adapter(). Please upgrade peft.")


def load_risk_tensors(config: Dict[str, Any], device: torch.device) -> Dict[str, Dict[int, torch.Tensor]]:
    path = resolve_path(config, config["stage3"]["risk_space"]["risk_basis_path"])
    if not path.exists():
        raise FileNotFoundError(f"Missing risk basis: {path}")
    data = torch.load(path, map_location="cpu", weights_only=False)
    layers = [int(x) for x in config["stage3"]["risk_space"].get("risk_layers", [20])]
    out = {"risk_basis": {}, "safe_center": {}}
    for layer in layers:
        if layer not in data["risk_basis"]:
            logger.warning("Risk basis missing layer %s; skipping.", layer)
            continue
        out["risk_basis"][layer] = data["risk_basis"][layer].float().to(device)
        if "safe_center" not in data or layer not in data["safe_center"]:
            raise ValueError(f"risk_basis.pt missing safe_center for layer {layer}; rebuild Stage 1.5/Stage 1 basis.")
        out["safe_center"][layer] = data["safe_center"][layer].float().to(device)
    if not out["risk_basis"]:
        raise ValueError("No usable risk layers found for Stage 3.")
    return out


def masked_kl(old_logits: torch.Tensor, new_logits: torch.Tensor, labels: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    mask = labels != -100
    if not mask.any():
        return new_logits.sum() * 0.0
    old = old_logits[mask].float() / temperature
    new = new_logits[mask].float() / temperature
    old_probs = F.softmax(old, dim=-1)
    new_log_probs = F.log_softmax(new, dim=-1)
    return F.kl_div(new_log_probs, old_probs, reduction="batchmean") * (temperature**2)


def _sequence_logp(model, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    labels = inputs["labels"]
    model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
    out = model(**model_inputs, return_dict=True)
    logits = out.logits[:, :-1, :].float()
    shifted_labels = labels[:, 1:].clone()
    mask = shifted_labels != -100
    if not mask.any():
        return logits.sum() * 0.0
    safe_labels = shifted_labels.masked_fill(~mask, 0)
    token_logps = F.log_softmax(logits, dim=-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    return (token_logps * mask.float()).sum(dim=-1).mean()


def _npo_forget_loss(
    model,
    processor,
    harmful: Dict[str, Any],
    device: torch.device,
    *,
    beta: float,
    max_pixels: Optional[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    response = harmful.get("response")
    if not isinstance(response, str) or not response.strip():
        raise ValueError("NPO requires the original harmful dataset response, but harmful.response is empty.")
    inputs = prepare_supervised_inputs(processor, harmful, response, device, max_pixels=max_pixels)
    was_training = model.training
    model.eval()
    with adapter_disabled(model), torch.no_grad():
        reference_logp = _sequence_logp(model, inputs)
    if was_training:
        model.train()
    policy_logp = _sequence_logp(model, inputs)
    log_ratio = policy_logp - reference_logp.detach()
    loss = -(2.0 / float(beta)) * F.logsigmoid(-float(beta) * log_ratio)
    return loss, policy_logp.detach(), reference_logp.detach(), log_ratio.detach()


@lru_cache(maxsize=8)
def _load_po_idk_responses(path: str) -> tuple[str, ...]:
    prompt_path = Path(path)
    if not prompt_path.exists():
        raise FileNotFoundError(
            f"SafeEraser PO requires its IDK prompt file: {prompt_path}. "
            "Copy dataset/prompt.json from the SafeEraser dataset, or pass --po_prompt_path."
        )
    data = json.loads(prompt_path.read_text(encoding="utf-8"))
    responses = data.get("idk") if isinstance(data, dict) else None
    responses = tuple(str(x).strip() for x in (responses or []) if str(x).strip())
    if not responses:
        raise ValueError(f"SafeEraser PO prompt file has no non-empty 'idk' list: {prompt_path}")
    return responses


def _token_nll_sum_and_count(model, inputs: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    labels = inputs["labels"]
    out = model(**inputs, return_dict=True)
    logits = out.logits[:, :-1, :].float()
    shifted_labels = labels[:, 1:].clone()
    mask = shifted_labels != -100
    if not mask.any():
        return logits.sum() * 0.0, mask.sum()
    nll_sum = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        shifted_labels.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    return nll_sum, mask.sum()


def _safeeraser_po_loss(
    model,
    processor,
    harmful: Dict[str, Any],
    retains: list[Dict[str, Any]],
    config: Dict[str, Any],
    device: torch.device,
    *,
    max_pixels: Optional[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
    if not retains:
        raise ValueError("SafeEraser PO requires one retain sample per harmful sample.")
    prompt_path = resolve_path(config, config["stage3"].get("po", {}).get("prompt_path", "dataset/prompt.json"))
    idk_responses = _load_po_idk_responses(str(prompt_path))
    idk_index = int(torch.randint(0, len(idk_responses), (1,)).item())
    idk_response = idk_responses[idk_index].strip(" ").capitalize()
    retain_index = int(torch.randint(0, len(retains), (1,)).item())
    retain = retains[retain_index]
    retain_response = retain.get("response")
    if not isinstance(retain_response, str) or not retain_response.strip():
        raise ValueError("SafeEraser PO requires a non-empty retain response.")

    harmful_inputs = prepare_supervised_inputs(
        processor, harmful, idk_response, device, max_pixels=max_pixels
    )
    retain_inputs = prepare_supervised_inputs(
        processor, retain, retain_response, device, max_pixels=max_pixels
    )
    harmful_nll, harmful_count = _token_nll_sum_and_count(model, harmful_inputs)
    retain_nll, retain_count = _token_nll_sum_and_count(model, retain_inputs)
    total_count = (harmful_count + retain_count).clamp_min(1).to(harmful_nll.dtype)
    loss = (harmful_nll + retain_nll) / total_count
    harmful_ce = harmful_nll / harmful_count.clamp_min(1).to(harmful_nll.dtype)
    retain_ce = retain_nll / retain_count.clamp_min(1).to(retain_nll.dtype)
    return loss, harmful_ce.detach(), retain_ce.detach(), idk_response


def _zero_like_model(model) -> torch.Tensor:
    for p in model.parameters():
        if p.requires_grad:
            return p.sum() * 0.0
    return torch.tensor(0.0)


def _target_mode(cfg: Dict[str, Any], key: str, default: str = "safe_neighbor") -> Dict[str, float | str]:
    raw = cfg.get(key, {})
    mode_raw = str(raw.get("mode", default)).lower()
    aliases = {
        "safe": "safe_neighbor",
        "safenb": "safe_neighbor",
        "safe_nb": "safe_neighbor",
        "safe_neighbor": "safe_neighbor",
        "retain": "retain",
        "mixed": "mixed",
        "mix": "mixed",
        "safe_retain_mix": "mixed",
    }
    mode = aliases.get(mode_raw)
    if mode is None:
        raise ValueError(f"Unsupported stage3.{key}.mode={raw.get('mode')!r}; use safe_neighbor, retain, or mixed.")
    safe_weight = float(raw.get("safe_weight", raw.get("alpha_safe", 0.5)))
    retain_weight = float(raw.get("retain_weight", raw.get("beta_retain", 0.5)))
    if mode == "safe_neighbor":
        safe_weight, retain_weight = 1.0, 0.0
    elif mode == "retain":
        safe_weight, retain_weight = 0.0, 1.0
    else:
        denom = safe_weight + retain_weight
        if denom <= 0:
            raise ValueError(f"stage3.{key}.mode=mixed requires safe_weight + retain_weight > 0.")
        safe_weight, retain_weight = safe_weight / denom, retain_weight / denom
    return {"mode": mode, "safe_weight": safe_weight, "retain_weight": retain_weight}


def _old_prompt_projection(
    model,
    processor,
    sample: Dict[str, Any],
    device: Optional[torch.device],
    risk_tensors: Dict[str, Dict[int, torch.Tensor]],
    *,
    max_pixels: Optional[int],
) -> Dict[int, torch.Tensor]:
    prompt = prepare_prompt_inputs(processor, sample, device, max_pixels=max_pixels)
    with adapter_disabled(model), torch.no_grad():
        out = model(**prompt, output_hidden_states=True, return_dict=True)
    projections: Dict[int, torch.Tensor] = {}
    for layer, basis in risk_tensors["risk_basis"].items():
        center = risk_tensors["safe_center"][layer]
        h = last_token_hidden(out, prompt, layer).to(basis.device)
        projections[int(layer)] = (h - center) @ basis.T
    return projections


def compute_triplet_losses(
    model,
    processor,
    triplet: Dict[str, Any],
    config: Dict[str, Any],
    risk_tensors: Dict[str, Dict[int, torch.Tensor]],
    *,
    debug_label_path: Optional[Path] = None,
) -> Dict[str, torch.Tensor]:
    device = infer_input_device(model)
    weights = config["stage3"]["loss_weights"]
    kl_temp = float(config["stage3"].get("kl", {}).get("temperature", 1.0))
    harmful = triplet["harmful"]
    safe = triplet["safe"]
    retains = triplet.get("retains", [])
    align_cfg = _target_mode(config["stage3"], "align_target", default="safe_neighbor")
    risk_weight = float(harmful.get("risk_weight", 1.0))
    max_pixels = config["stage3"].get("preprocessing", {}).get("max_pixels", 200704)

    zero = _zero_like_model(model)
    losses: Dict[str, torch.Tensor] = {
        "loss_safe_ce": zero,
        "loss_safe_neighbor_ce": zero,
        "loss_npo": zero,
        "npo_policy_logp": zero,
        "npo_reference_logp": zero,
        "npo_log_ratio": zero,
        "loss_po": zero,
        "loss_po_harmful_ce": zero,
        "loss_po_retain_ce": zero,
        "loss_align": zero,
        "loss_implicit": zero,
        "loss_safe_kl": zero,
        "loss_retain_kl": zero,
        "loss_retain_hidden": zero,
    }

    safe_ce_weight = float(weights.get("safe_ce", 1.0))
    safe_neighbor_ce_weight = float(weights.get("safe_neighbor_ce", 1.0))
    align_weight = float(weights.get("align", 1.0))
    implicit_weight = float(weights.get("implicit", 0.15))
    safe_kl_weight = float(weights.get("safe_kl", 1.5))
    retain_kl_weight = float(weights.get("retain_kl", 2.0))
    retain_hidden_weight = float(weights.get("retain_hidden", 0.5))

    if safe_ce_weight != 0:
        safe_target = harmful.get("target_safe_response") or config["stage3"]["safe_response"]["fallback_template"]
        sup_harm = prepare_supervised_inputs(
            processor,
            harmful,
            safe_target,
            device,
            debug_path=debug_label_path,
            max_pixels=max_pixels,
        )
        out_harm_ce = model(**sup_harm, return_dict=True)
        losses["loss_safe_ce"] = out_harm_ce.loss

    npo_weight = float(weights.get("npo", 0.0))
    if npo_weight > 0:
        beta = float(config["stage3"].get("npo", {}).get("beta", 0.1))
        if beta <= 0:
            raise ValueError("stage3.npo.beta must be positive when NPO is enabled.")
        loss_npo, policy_logp, reference_logp, log_ratio = _npo_forget_loss(
            model,
            processor,
            harmful,
            device,
            beta=beta,
            max_pixels=max_pixels,
        )
        losses["loss_npo"] = loss_npo
        losses["npo_policy_logp"] = policy_logp
        losses["npo_reference_logp"] = reference_logp
        losses["npo_log_ratio"] = log_ratio

    po_weight = float(weights.get("po", 0.0))
    if po_weight > 0:
        loss_po, po_harmful_ce, po_retain_ce, po_target = _safeeraser_po_loss(
            model,
            processor,
            harmful,
            retains,
            config,
            device,
            max_pixels=max_pixels,
        )
        losses["loss_po"] = loss_po
        losses["loss_po_harmful_ce"] = po_harmful_ce
        losses["loss_po_retain_ce"] = po_retain_ce
        harmful["po_idk_target"] = po_target

    if align_weight != 0 or implicit_weight != 0:
        harm_prompt = prepare_prompt_inputs(processor, harmful, device, max_pixels=max_pixels)
        out_new_h = model(**harm_prompt, output_hidden_states=True, return_dict=True)
        safe_proj = None
        retain_proj_terms: list[Dict[int, torch.Tensor]] = []
        if float(align_cfg["safe_weight"]) > 0:
            safe_proj = _old_prompt_projection(model, processor, safe, device, risk_tensors, max_pixels=max_pixels)
        if float(align_cfg["retain_weight"]) > 0:
            for retain in retains:
                retain_proj_terms.append(
                    _old_prompt_projection(model, processor, retain, device, risk_tensors, max_pixels=max_pixels)
                )
            if not retain_proj_terms:
                raise ValueError("stage3.align_target.mode requires retain samples, but triplet has no retain entries.")
        align_terms = []
        implicit_terms = []
        for layer, basis in risk_tensors["risk_basis"].items():
            center = risk_tensors["safe_center"][layer]
            h_new = last_token_hidden(out_new_h, harm_prompt, layer).to(basis.device)
            proj_new = (h_new - center) @ basis.T
            if align_cfg["mode"] == "safe_neighbor":
                proj_target = safe_proj[int(layer)]
            elif align_cfg["mode"] == "retain":
                proj_target = torch.stack([x[int(layer)] for x in retain_proj_terms]).mean(dim=0)
            else:
                proj_retain = torch.stack([x[int(layer)] for x in retain_proj_terms]).mean(dim=0)
                proj_target = (
                    float(align_cfg["safe_weight"]) * safe_proj[int(layer)]
                    + float(align_cfg["retain_weight"]) * proj_retain
                )
            align_terms.append(F.mse_loss(proj_new, proj_target))
            relative_proj = proj_new - proj_target
            implicit_terms.append(torch.mean(torch.relu(relative_proj).pow(2)))
        if align_terms:
            losses["loss_align"] = torch.stack(align_terms).mean() * risk_weight
            losses["loss_implicit"] = torch.stack(implicit_terms).mean() * risk_weight

    # Safe-neighbor behavior branch: answer the safety-oriented prompt instead
    # of generalizing the harmful-prompt refusal target to all risky contexts.
    if (safe_neighbor_ce_weight != 0 or safe_kl_weight != 0) and safe.get("response"):
        sup_safe = prepare_supervised_inputs(processor, safe, safe["response"], device, max_pixels=max_pixels)
        new_safe = model(**sup_safe, return_dict=True)
        if safe_neighbor_ce_weight != 0:
            losses["loss_safe_neighbor_ce"] = new_safe.loss
        if safe_kl_weight != 0:
            with adapter_disabled(model), torch.no_grad():
                old_safe = model(**{k: v for k, v in sup_safe.items() if k != "labels"}, return_dict=True)
            losses["loss_safe_kl"] = masked_kl(old_safe.logits, new_safe.logits, sup_safe["labels"], kl_temp)

    retain_kl_terms = []
    retain_hidden_terms = []
    for retain in retains if (retain_kl_weight != 0 or retain_hidden_weight != 0) else []:
        response = retain.get("response")
        if retain_kl_weight != 0 and response:
            sup_ret = prepare_supervised_inputs(processor, retain, response, device, max_pixels=max_pixels)
            with adapter_disabled(model), torch.no_grad():
                old_ret = model(**{k: v for k, v in sup_ret.items() if k != "labels"}, return_dict=True)
            new_ret = model(**{k: v for k, v in sup_ret.items() if k != "labels"}, return_dict=True)
            retain_kl_terms.append(masked_kl(old_ret.logits, new_ret.logits, sup_ret["labels"], kl_temp))
        if retain_hidden_weight != 0:
            ret_prompt = prepare_prompt_inputs(processor, retain, device, max_pixels=max_pixels)
            with adapter_disabled(model), torch.no_grad():
                old_ret_h = model(**ret_prompt, output_hidden_states=True, return_dict=True)
            new_ret_h = model(**ret_prompt, output_hidden_states=True, return_dict=True)
            for layer in config["stage3"].get("hidden_retain", {}).get("layers", [20]):
                layer = int(layer)
                retain_hidden_terms.append(
                    F.mse_loss(last_token_hidden(new_ret_h, ret_prompt, layer), last_token_hidden(old_ret_h, ret_prompt, layer))
                )
    if retain_kl_terms:
        losses["loss_retain_kl"] = torch.stack(retain_kl_terms).mean()
    if retain_hidden_terms:
        losses["loss_retain_hidden"] = torch.stack(retain_hidden_terms).mean()

    total = (
        safe_ce_weight * losses["loss_safe_ce"]
        + safe_neighbor_ce_weight * losses["loss_safe_neighbor_ce"]
        + npo_weight * losses["loss_npo"]
        + po_weight * losses["loss_po"]
        + align_weight * losses["loss_align"]
        + implicit_weight * losses["loss_implicit"]
        + safe_kl_weight * losses["loss_safe_kl"]
        + retain_kl_weight * losses["loss_retain_kl"]
        + retain_hidden_weight * losses["loss_retain_hidden"]
    )
    losses["loss_total"] = total
    return losses
