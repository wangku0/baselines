from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from ..risk_space.recommended_config import RecommendedRiskConfig, load_recommended_risk_config
from ..stage3_losses import adapter_disabled, last_token_hidden, prepare_prompt_inputs
from ..utils import resolve_path
from .model import FlowVectorField, euler_integrate_flow
from .utils import compute_risk_coefficients, compute_risk_delta_coefficients, lambda_flow_ramp


def load_flow_teacher(config: Dict[str, Any], device: torch.device) -> Optional[Dict[str, Any]]:
    flow_cfg = config.get("stage3", {}).get("flow_distillation", {})
    if not bool(flow_cfg.get("enabled", False)):
        return None
    path = resolve_path(config, flow_cfg.get("teacher_path", "integrations/my_method/outputs/stage2_5_flow/flow_teacher.pt"))
    if not path.exists():
        raise FileNotFoundError(f"Flow teacher not found: {path}. Run scripts/06_stage2_5_train_flow_matching.py first.")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    teacher_cfg = ckpt.get("teacher_cfg", {})
    model = FlowVectorField(
        hidden_dim=int(ckpt["hidden_dim"]),
        cond_dim=int(ckpt["cond_dim"]),
        hidden_width=int(teacher_cfg.get("hidden_width", 1024)),
        time_embedding_dim=int(teacher_cfg.get("time_embedding_dim", 128)),
        layer_embedding_dim=int(teacher_cfg.get("layer_embedding_dim", 16)),
        dropout=float(teacher_cfg.get("dropout", 0.05)),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return {"model": model, "ckpt": ckpt}


def _validate_teacher_layers(flow_bundle: Dict[str, Any], required_layers: list[int]) -> None:
    rec = flow_bundle.get("ckpt", {}).get("recommended", {})
    teacher_layers = sorted({int(x) for x in rec.get("recommended_hidden_layers", [])})
    required = sorted({int(x) for x in required_layers})
    if teacher_layers and teacher_layers != required:
        raise RuntimeError(
            "Flow teacher layer mismatch: "
            f"teacher was trained for hidden_layers={teacher_layers}, but Stage 3 is using hidden_layers={required}. "
            "Retrain Stage 2.5 with the same layer selection method, for example: "
            "python scripts/06_stage2_5_train_flow_matching.py --config integrations/my_method/configs/safeeraser_llava.yaml "
            "--split train --layer_selection_method risk_transport_influence"
        )


def _condition(x: torch.Tensor, coeff: torch.Tensor, r_explicit: float, group_id: int, cond_dim: int) -> torch.Tensor:
    group = torch.zeros(3, dtype=x.dtype, device=x.device)
    group[int(group_id)] = 1.0
    cond = torch.cat([x, coeff.to(x.dtype), torch.tensor([float(r_explicit)], dtype=x.dtype, device=x.device), group], dim=0)
    if cond.numel() < cond_dim:
        cond = F.pad(cond, (0, cond_dim - cond.numel()))
    elif cond.numel() > cond_dim:
        cond = cond[:cond_dim]
    return cond


def _target_mode(config: Dict[str, Any]) -> Dict[str, float | str]:
    raw = config.get("stage3", {}).get("align_target", {})
    mode_raw = str(raw.get("mode", "safe_neighbor")).lower()
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
        raise ValueError(f"Unsupported stage3.align_target.mode={raw.get('mode')!r}; use safe_neighbor, retain, or mixed.")
    safe_weight = float(raw.get("safe_weight", raw.get("alpha_safe", 0.5)))
    retain_weight = float(raw.get("retain_weight", raw.get("beta_retain", 0.5)))
    if mode == "safe_neighbor":
        safe_weight, retain_weight = 1.0, 0.0
    elif mode == "retain":
        safe_weight, retain_weight = 0.0, 1.0
    else:
        denom = safe_weight + retain_weight
        if denom <= 0:
            raise ValueError("stage3.align_target.mode=mixed requires safe_weight + retain_weight > 0.")
        safe_weight, retain_weight = safe_weight / denom, retain_weight / denom
    return {"mode": mode, "safe_weight": safe_weight, "retain_weight": retain_weight}


def compute_flow_stage3_losses(
    model,
    processor,
    triplet: Dict[str, Any],
    config: Dict[str, Any],
    risk_tensors: Dict[str, Dict[int, torch.Tensor]],
    flow_bundle: Optional[Dict[str, Any]],
    recommended: RecommendedRiskConfig,
    *,
    global_step: int,
    total_steps: int,
) -> Dict[str, torch.Tensor | float]:
    zero = next((p for p in model.parameters() if p.requires_grad), None)
    zero_t = zero.sum() * 0.0 if zero is not None else torch.tensor(0.0)
    out: Dict[str, torch.Tensor | float] = {
        "loss_flow_distill": zero_t,
        "loss_flow_delta": zero_t,
        "loss_flow_cos": zero_t,
        "loss_flow_risk": zero_t,
        "loss_flow_identity": zero_t,
        "lambda_flow": 0.0,
        "harmful_delta_lora_norm": 0.0,
        "harmful_delta_flow_norm": 0.0,
        "harmful_delta_cos_lora_flow": 0.0,
        "safe_delta_norm": 0.0,
        "retain_delta_norm": 0.0,
    }
    if flow_bundle is None:
        return out
    _validate_teacher_layers(flow_bundle, recommended.recommended_hidden_layers)

    flow_cfg = config["stage3"]["flow_distillation"]
    teacher: FlowVectorField = flow_bundle["model"]
    cond_dim = int(flow_bundle["ckpt"]["cond_dim"])
    device = next(teacher.parameters()).device
    max_pixels = config["stage3"].get("preprocessing", {}).get("max_pixels", 200704)
    lambda_flow = lambda_flow_ramp(
        global_step,
        total_steps,
        float(flow_cfg.get("lambda_flow_max", 0.5)),
        float(flow_cfg.get("ramp_start_ratio", 0.10)),
        float(flow_cfg.get("ramp_end_ratio", 0.40)),
        int(flow_cfg.get("min_flow_warmup_steps", 10)),
    )
    lambda_identity = float(flow_cfg.get("lambda_identity", 0.5))
    ode_steps = int(flow_cfg.get("ode_steps", 8))

    harmful = triplet["harmful"]
    safe = triplet["safe"]
    retains = triplet.get("retains", [])
    target_cfg = _target_mode(config)
    from ..model_utils import infer_input_device

    input_device = infer_input_device(model)
    harm_prompt = prepare_prompt_inputs(processor, harmful, input_device, max_pixels=max_pixels)
    with adapter_disabled(model), torch.no_grad():
        base_h = model(**harm_prompt, output_hidden_states=True, return_dict=True)
        safe_prompt = prepare_prompt_inputs(processor, safe, input_device, max_pixels=max_pixels)
        safe_base = model(**safe_prompt, output_hidden_states=True, return_dict=True)
        retain_bases = []
        retain_prompts = []
        if float(target_cfg["retain_weight"]) > 0:
            for retain in retains:
                retain_prompt = prepare_prompt_inputs(processor, retain, input_device, max_pixels=max_pixels)
                retain_prompts.append(retain_prompt)
                retain_bases.append(model(**retain_prompt, output_hidden_states=True, return_dict=True))
            if not retain_bases:
                raise ValueError("Flow distillation target requires retain samples, but triplet has none.")
    lora_h = model(**harm_prompt, output_hidden_states=True, return_dict=True)
    delta_terms = []
    cos_terms = []
    risk_terms = []
    lora_norms = []
    flow_norms = []
    cos_vals = []
    for layer in recommended.recommended_hidden_layers:
        basis = risk_tensors["risk_basis"][layer]
        center = risk_tensors["safe_center"][layer]
        x_base = last_token_hidden(base_h, harm_prompt, layer).to(device)
        x_lora = last_token_hidden(lora_h, harm_prompt, layer).to(device)
        x_safe_target = last_token_hidden(safe_base, safe_prompt, layer).to(device)
        if target_cfg["mode"] == "safe_neighbor":
            x_target = x_safe_target
        else:
            retain_targets = [
                last_token_hidden(retain_base, retain_prompt, layer).to(device)
                for retain_base, retain_prompt in zip(retain_bases, retain_prompts)
            ]
            x_retain_target = torch.stack(retain_targets).mean(dim=0)
            if target_cfg["mode"] == "retain":
                x_target = x_retain_target
            else:
                x_target = float(target_cfg["safe_weight"]) * x_safe_target + float(target_cfg["retain_weight"]) * x_retain_target
        coeff = compute_risk_delta_coefficients(
            (x_base - x_target)[None, :],
            layer,
            recommended,
            risk_tensors["risk_basis"],
        )[0].to(device)
        cond = _condition(x_base, coeff, harmful.get("R_explicit", 0.0), 0, cond_dim)[None, :]
        layer_id = torch.tensor([int(layer)], device=device)
        with torch.no_grad():
            x_flow = euler_integrate_flow(teacher, x_base[None, :], cond, layer_id=layer_id, steps=ode_steps)[0]
        delta_lora = x_lora - x_base.to(x_lora.device)
        delta_flow = x_flow.to(x_lora.device) - x_base.to(x_lora.device)
        delta_denom = delta_flow.detach().pow(2).mean().clamp_min(1e-6)
        layer_delta_loss = F.mse_loss(delta_lora, delta_flow) / delta_denom
        delta_terms.append(layer_delta_loss)
        cos = F.cosine_similarity(delta_lora[None, :], delta_flow[None, :], dim=-1).mean()
        cos_terms.append(1.0 - cos)
        risk_lora = compute_risk_coefficients(x_lora[None, :], layer, recommended, risk_tensors["risk_basis"], risk_tensors["safe_center"])
        risk_flow = compute_risk_coefficients(x_flow[None, :].to(x_lora.device), layer, recommended, risk_tensors["risk_basis"], risk_tensors["safe_center"])
        risk_denom = risk_flow.detach().pow(2).mean().clamp_min(1e-6)
        layer_risk_loss = F.mse_loss(risk_lora, risk_flow) / risk_denom
        risk_terms.append(layer_risk_loss)
        lora_norms.append(torch.norm(delta_lora).detach())
        flow_norms.append(torch.norm(delta_flow).detach())
        cos_vals.append(cos.detach())
        out[f"flow_delta_norm_layer_{layer}"] = float(torch.norm(delta_flow).detach().cpu())
        out[f"lora_delta_norm_layer_{layer}"] = float(torch.norm(delta_lora).detach().cpu())
        out[f"flow_risk_loss_layer_{layer}"] = float(layer_risk_loss.detach().float().cpu())

    loss_delta = torch.stack(delta_terms).mean() if delta_terms else zero_t
    loss_cos = torch.stack(cos_terms).mean() if cos_terms else zero_t
    loss_risk = torch.stack(risk_terms).mean() if risk_terms else zero_t
    loss_distill = (
        float(flow_cfg.get("delta_mse_weight", 0.5)) * loss_delta
        + float(flow_cfg.get("delta_cos_weight", 0.2)) * loss_cos
        + float(flow_cfg.get("risk_mse_weight", 1.0)) * loss_risk
    )

    identity_terms = []
    safe_delta_norms = []
    retain_delta_norms = []
    for sample, bucket in [(triplet["safe"], safe_delta_norms)] + [(r, retain_delta_norms) for r in triplet.get("retains", [])]:
        prompt = prepare_prompt_inputs(processor, sample, input_device, max_pixels=max_pixels)
        with adapter_disabled(model), torch.no_grad():
            base = model(**prompt, output_hidden_states=True, return_dict=True)
        new = model(**prompt, output_hidden_states=True, return_dict=True)
        for layer in recommended.recommended_hidden_layers:
            xb = last_token_hidden(base, prompt, layer)
            xn = last_token_hidden(new, prompt, layer)
            identity_terms.append(F.mse_loss(xn, xb))
            bucket.append(torch.norm(xn - xb).detach())
            rb = compute_risk_coefficients(xb[None, :], layer, recommended, risk_tensors["risk_basis"], risk_tensors["safe_center"])
            rn = compute_risk_coefficients(xn[None, :], layer, recommended, risk_tensors["risk_basis"], risk_tensors["safe_center"])
            identity_terms.append(F.mse_loss(rn, rb))
    loss_identity = torch.stack(identity_terms).mean() if identity_terms else zero_t

    out.update(
        {
            "loss_flow_distill": lambda_flow * loss_distill,
            "loss_flow_delta": loss_delta,
            "loss_flow_cos": loss_cos,
            "loss_flow_risk": loss_risk,
            "loss_flow_identity": lambda_identity * loss_identity,
            "lambda_flow": float(lambda_flow),
            "harmful_delta_lora_norm": float(torch.stack(lora_norms).mean().cpu()) if lora_norms else 0.0,
            "harmful_delta_flow_norm": float(torch.stack(flow_norms).mean().cpu()) if flow_norms else 0.0,
            "harmful_delta_cos_lora_flow": float(torch.stack(cos_vals).mean().cpu()) if cos_vals else 0.0,
            "safe_delta_norm": float(torch.stack(safe_delta_norms).mean().cpu()) if safe_delta_norms else 0.0,
            "retain_delta_norm": float(torch.stack(retain_delta_norms).mean().cpu()) if retain_delta_norms else 0.0,
        }
    )
    return out
