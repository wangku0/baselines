from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm

from .data_loader import load_dataset
from .model_utils import infer_input_device, load_model_and_processor, prepare_vl_batch_inputs, prepare_vl_inputs
from .utils import cuda_oom_help, ensure_dir, ensure_output_dirs, logger, resolve_path, save_json


def _requested_layers(config: Dict[str, Any]) -> List[int]:
    hidden_cfg = config.get("hidden_states", {})
    offset = int(hidden_cfg.get("layer_index_offset", 0))
    return [int(layer) + offset for layer in hidden_cfg.get("target_layers", [])]


def _last_input_token_position(attention_mask: torch.Tensor) -> int:
    positions = torch.nonzero(attention_mask[0] > 0, as_tuple=False).flatten()
    if positions.numel() == 0:
        raise ValueError("attention_mask has no valid input tokens.")
    return int(positions[-1].item())


def _last_input_token_positions(attention_mask: torch.Tensor) -> List[int]:
    return [_last_input_token_position(attention_mask[row : row + 1]) for row in range(attention_mask.shape[0])]


def _chunks(items: List[Dict[str, Any]], batch_size: int) -> List[List[Dict[str, Any]]]:
    return [items[start : start + batch_size] for start in range(0, len(items), batch_size)]


def _load_safeeraser_checkpoint(model: Any, checkpoint_path: str, r: int = 32, alpha: int = 256) -> Any:
    from peft import LoraConfig, get_peft_model

    target_modules = (
        r".*language_model.*\."
        r"(up_proj|k_proj|linear_2|down_proj|v_proj|q_proj|o_proj|gate_proj|linear_1)"
    )
    peft_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected SafeEraser checkpoint state dict, got {type(checkpoint).__name__}: {checkpoint_path}")
    model_keys = set(model.state_dict())
    matched_keys = model_keys.intersection(checkpoint)
    matched_lora_keys = [key for key in matched_keys if "lora_" in key]
    if not matched_lora_keys:
        raise RuntimeError(
            "The SafeEraser checkpoint did not match any LoRA parameters. "
            f"Sample checkpoint keys: {list(checkpoint)[:5]}"
        )
    incompatible = model.load_state_dict(checkpoint, strict=False)
    logger.info(
        "Loaded SafeEraser checkpoint for hidden extraction: %s matched=%d lora=%d unexpected=%d",
        checkpoint_path,
        len(matched_keys),
        len(matched_lora_keys),
        len(incompatible.unexpected_keys),
    )
    return model.merge_and_unload()


def _select_layers_once(config: Dict[str, Any], hidden_states: Any, requested_layers: List[int]) -> List[int]:
    hidden_cfg = config.get("hidden_states", {})
    total_layers = len(hidden_states)
    logger.info("outputs.hidden_states total length: %d", total_layers)
    if bool(hidden_cfg.get("all_layers", False)):
        selected_layers = list(range(1, total_layers))
        logger.info("All-layer extraction enabled; using hidden state layer indices: %s", selected_layers)
        return selected_layers

    logger.info("Requested hidden state layer indices: %s", requested_layers)
    selected_layers = [layer for layer in requested_layers if 0 <= layer < total_layers]
    dropped = sorted(set(requested_layers) - set(selected_layers))
    if dropped:
        logger.warning("Dropping out-of-range hidden state layer indices: %s", dropped)
    if not selected_layers:
        raise ValueError(f"No valid target layers. Requested {requested_layers}, but hidden_states length is {total_layers}.")
    logger.info("Actual target hidden state layer indices: %s", selected_layers)
    return selected_layers


def extract_hidden_states_for_split(
    config: Dict[str, Any],
    split: str,
    max_samples: Optional[int] = None,
    model_path_override: Optional[str] = None,
    safeeraser_checkpoint_path: Optional[str] = None,
    safeeraser_lora_r: int = 32,
    safeeraser_lora_alpha: int = 256,
) -> Path:
    ensure_output_dirs(config)
    if max_samples is None:
        max_samples = config.get("hidden_states", {}).get("max_samples")
    if max_samples is None:
        max_samples = config.get("debug", {}).get("max_samples")

    samples = load_dataset(config, split=split, max_samples=max_samples)
    if not samples:
        raise ValueError(f"No samples available for split={split}. Check dataset paths and image files.")

    counts = Counter(sample["sample_type"] for sample in samples)
    logger.info("Extracting hidden states for split=%s, sample counts=%s", split, dict(counts))

    model, processor = load_model_and_processor(config, model_path_override=model_path_override)
    if safeeraser_checkpoint_path:
        model = _load_safeeraser_checkpoint(
            model,
            safeeraser_checkpoint_path,
            r=int(safeeraser_lora_r),
            alpha=int(safeeraser_lora_alpha),
        )
    input_device = infer_input_device(model)
    hidden_cfg = config.get("hidden_states", {})
    max_pixels = hidden_cfg.get("max_pixels")
    clear_cache_every = int(hidden_cfg.get("clear_cuda_cache_every", 0) or 0)
    batch_size = max(1, int(hidden_cfg.get("batch_size", 1) or 1))
    logger.info("Hidden-state image max_pixels=%s", max_pixels)
    logger.info("Hidden-state extraction batch_size=%d", batch_size)
    requested_layers = _requested_layers(config)
    selected_layers: Optional[List[int]] = None
    hidden_by_layer: Dict[int, List[torch.Tensor]] = {}
    metadata: List[Dict[str, Any]] = []
    failed_samples: List[Dict[str, Any]] = []

    processed_count = 0
    for batch in tqdm(_chunks(samples, batch_size), desc=f"extract {split} hidden states"):
        try:
            inputs = (
                prepare_vl_inputs(processor, batch[0]["image_path"], batch[0]["instruction"], input_device, max_pixels=max_pixels)
                if len(batch) == 1
                else prepare_vl_batch_inputs(processor, batch, input_device, max_pixels=max_pixels)
            )
            with torch.inference_mode():
                outputs = model(**inputs, output_hidden_states=True, return_dict=True)
            hidden_states = outputs.hidden_states
            if hidden_states is None:
                raise RuntimeError("Model forward returned hidden_states=None.")

            if selected_layers is None:
                selected_layers = _select_layers_once(config, hidden_states, requested_layers)
                hidden_by_layer = {layer: [] for layer in selected_layers}

            last_positions = _last_input_token_positions(inputs["attention_mask"])
            for row, sample in enumerate(batch):
                last_pos = last_positions[row]
                item_meta = dict(sample)
                item_meta["last_token_position"] = last_pos
                item_meta["input_length"] = int(inputs["attention_mask"][row].sum().item())
                for layer in selected_layers:
                    vec = hidden_states[layer][row, last_pos, :].detach().to("cpu", dtype=torch.float32)
                    hidden_by_layer[layer].append(vec)
                metadata.append(item_meta)
                processed_count += 1
            del outputs, hidden_states, inputs
            if clear_cache_every and torch.cuda.is_available() and processed_count % clear_cache_every == 0:
                torch.cuda.empty_cache()
        except RuntimeError as exc:
            message = str(exc)
            if "out of memory" in message.lower():
                message = f"{message}\n{cuda_oom_help()}"
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if len(batch) > 1:
                logger.warning("Failed batch of %d samples: %s. Retrying one by one.", len(batch), message)
                for sample in batch:
                    try:
                        inputs = prepare_vl_inputs(
                            processor,
                            sample["image_path"],
                            sample["instruction"],
                            input_device,
                            max_pixels=max_pixels,
                        )
                        with torch.inference_mode():
                            outputs = model(**inputs, output_hidden_states=True, return_dict=True)
                        hidden_states = outputs.hidden_states
                        if hidden_states is None:
                            raise RuntimeError("Model forward returned hidden_states=None.")
                        if selected_layers is None:
                            selected_layers = _select_layers_once(config, hidden_states, requested_layers)
                            hidden_by_layer = {layer: [] for layer in selected_layers}
                        last_pos = _last_input_token_position(inputs["attention_mask"])
                        item_meta = dict(sample)
                        item_meta["last_token_position"] = last_pos
                        item_meta["input_length"] = int(inputs["attention_mask"][0].sum().item())
                        for layer in selected_layers:
                            vec = hidden_states[layer][0, last_pos, :].detach().to("cpu", dtype=torch.float32)
                            hidden_by_layer[layer].append(vec)
                        metadata.append(item_meta)
                        processed_count += 1
                        del outputs, hidden_states, inputs
                    except Exception as single_exc:
                        logger.warning("Failed sample %s: %s", sample.get("id"), single_exc)
                        failed_samples.append({"sample": sample, "error": str(single_exc)})
            else:
                sample = batch[0]
                logger.warning("Failed sample %s: %s", sample.get("id"), message)
                failed_samples.append({"sample": sample, "error": message})
        except Exception as exc:
            if len(batch) > 1:
                logger.warning("Failed batch of %d samples: %s. Retrying one by one.", len(batch), exc)
                for sample in batch:
                    try:
                        inputs = prepare_vl_inputs(
                            processor,
                            sample["image_path"],
                            sample["instruction"],
                            input_device,
                            max_pixels=max_pixels,
                        )
                        with torch.inference_mode():
                            outputs = model(**inputs, output_hidden_states=True, return_dict=True)
                        hidden_states = outputs.hidden_states
                        if hidden_states is None:
                            raise RuntimeError("Model forward returned hidden_states=None.")
                        if selected_layers is None:
                            selected_layers = _select_layers_once(config, hidden_states, requested_layers)
                            hidden_by_layer = {layer: [] for layer in selected_layers}
                        last_pos = _last_input_token_position(inputs["attention_mask"])
                        item_meta = dict(sample)
                        item_meta["last_token_position"] = last_pos
                        item_meta["input_length"] = int(inputs["attention_mask"][0].sum().item())
                        for layer in selected_layers:
                            vec = hidden_states[layer][0, last_pos, :].detach().to("cpu", dtype=torch.float32)
                            hidden_by_layer[layer].append(vec)
                        metadata.append(item_meta)
                        processed_count += 1
                        del outputs, hidden_states, inputs
                    except Exception as single_exc:
                        logger.warning("Failed sample %s: %s", sample.get("id"), single_exc)
                        failed_samples.append({"sample": sample, "error": str(single_exc)})
            else:
                sample = batch[0]
                logger.warning("Failed sample %s: %s", sample.get("id"), exc)
                failed_samples.append({"sample": sample, "error": str(exc)})

    if not metadata or selected_layers is None:
        raise RuntimeError(f"All samples failed for split={split}; no hidden states were saved.")

    stacked = {layer: torch.stack(values, dim=0) for layer, values in hidden_by_layer.items()}
    output = {
        "metadata": metadata,
        "hidden_states": stacked,
        "target_layers": selected_layers,
        "model_path": model_path_override or config.get("model", {}).get("local_path"),
        "safeeraser_checkpoint_path": safeeraser_checkpoint_path,
        "token_position": config.get("hidden_states", {}).get("token_position", "last_input_token"),
        "all_layers": bool(config.get("hidden_states", {}).get("all_layers", False)),
        "batch_size": batch_size,
    }

    hidden_dir = ensure_dir(resolve_path(config, config["outputs"]["hidden_states_dir"]))
    out_path = hidden_dir / f"{split}_hidden_states.pt"
    torch.save(output, out_path)
    logger.info("Saved hidden states to %s", out_path)

    metrics_dir = ensure_dir(resolve_path(config, config["outputs"]["metrics_dir"]))
    failed_path = metrics_dir / f"{split}_failed_hidden_state_samples.json"
    save_json(failed_samples, failed_path)
    logger.info("Saved %d failed samples to %s", len(failed_samples), failed_path)
    return out_path
