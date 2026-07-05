from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm

from .data_loader import load_dataset
from .model_utils import infer_input_device, load_model_and_processor, prepare_vl_inputs
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


def extract_hidden_states_for_split(
    config: Dict[str, Any],
    split: str,
    max_samples: Optional[int] = None,
    model_path_override: Optional[str] = None,
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
    input_device = infer_input_device(model)
    hidden_cfg = config.get("hidden_states", {})
    max_pixels = hidden_cfg.get("max_pixels")
    clear_cache_every = int(hidden_cfg.get("clear_cuda_cache_every", 0) or 0)
    logger.info("Hidden-state image max_pixels=%s", max_pixels)
    requested_layers = _requested_layers(config)
    selected_layers: Optional[List[int]] = None
    hidden_by_layer: Dict[int, List[torch.Tensor]] = {}
    metadata: List[Dict[str, Any]] = []
    failed_samples: List[Dict[str, Any]] = []

    for sample_index, sample in enumerate(tqdm(samples, desc=f"extract {split} hidden states"), start=1):
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
                total_layers = len(hidden_states)
                logger.info("outputs.hidden_states total length: %d", total_layers)
                logger.info("Requested hidden state layer indices: %s", requested_layers)
                selected_layers = [layer for layer in requested_layers if 0 <= layer < total_layers]
                dropped = sorted(set(requested_layers) - set(selected_layers))
                if dropped:
                    logger.warning("Dropping out-of-range hidden state layer indices: %s", dropped)
                if not selected_layers:
                    raise ValueError(
                        f"No valid target layers. Requested {requested_layers}, but hidden_states length is {total_layers}."
                    )
                logger.info("Actual target hidden state layer indices: %s", selected_layers)
                hidden_by_layer = {layer: [] for layer in selected_layers}

            last_pos = _last_input_token_position(inputs["attention_mask"])
            item_meta = dict(sample)
            item_meta["last_token_position"] = last_pos
            item_meta["input_length"] = int(inputs["attention_mask"][0].sum().item())
            for layer in selected_layers:
                vec = hidden_states[layer][0, last_pos, :].detach().to("cpu", dtype=torch.float32)
                hidden_by_layer[layer].append(vec)
            metadata.append(item_meta)
            del outputs, hidden_states, inputs
            if clear_cache_every and torch.cuda.is_available() and sample_index % clear_cache_every == 0:
                torch.cuda.empty_cache()
        except RuntimeError as exc:
            message = str(exc)
            if "out of memory" in message.lower():
                message = f"{message}\n{cuda_oom_help()}"
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            logger.warning("Failed sample %s: %s", sample.get("id"), message)
            failed_samples.append({"sample": sample, "error": message})
        except Exception as exc:
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
        "token_position": config.get("hidden_states", {}).get("token_position", "last_input_token"),
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
