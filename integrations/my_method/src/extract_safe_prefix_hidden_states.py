from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm

from .data_loader import load_dataset
from .extract_hidden_states import _last_input_token_positions, _requested_layers, _select_layers_once
from .model_utils import (
    infer_input_device,
    load_model_and_processor,
    prepare_vl_batch_inputs_with_assistant_prefix,
    prepare_vl_inputs_with_assistant_prefix,
)
from .utils import cuda_oom_help, ensure_dir, logger, resolve_path, save_json


def safe_prefix_hidden_dir(config: Dict[str, Any]) -> Path:
    outputs = config.get("outputs", {})
    target_cfg = config.get("flow_matching", {}).get("target", {})
    raw = target_cfg.get("safe_prefix_hidden_states_dir") or outputs.get("safe_prefix_hidden_states_dir")
    if raw:
        return resolve_path(config, raw)
    hidden_dir = Path(str(outputs.get("hidden_states_dir", "integrations/my_method/outputs/hidden_states")))
    return resolve_path(config, str(hidden_dir.parent / "safe_prefix_hidden_states"))


def _prefix_text(processor: Any, response: str, prefix_tokens: int) -> str:
    response = str(response or "").strip()
    if not response:
        return ""
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None or prefix_tokens <= 0:
        return response
    ids = tokenizer.encode(response, add_special_tokens=False)
    if not ids:
        return response
    return tokenizer.decode(ids[: int(prefix_tokens)], skip_special_tokens=True).strip() or response


def _chunks(items: List[Dict[str, Any]], batch_size: int) -> List[List[Dict[str, Any]]]:
    return [items[start : start + batch_size] for start in range(0, len(items), batch_size)]


def extract_safe_prefix_hidden_states_for_split(
    config: Dict[str, Any],
    split: str,
    max_samples: Optional[int] = None,
    model_path_override: Optional[str] = None,
    safe_prefix_tokens: Optional[int] = None,
) -> Path:
    if max_samples is None:
        max_samples = config.get("hidden_states", {}).get("max_samples")
    if max_samples is None:
        max_samples = config.get("debug", {}).get("max_samples")

    safe_samples = load_dataset(config, split=split, sample_types=["safe_neighbor"], max_samples=max_samples)
    if not safe_samples:
        raise ValueError(f"No safe_neighbor samples available for split={split}; cannot extract safe answer prefix hidden.")

    model, processor = load_model_and_processor(config, model_path_override=model_path_override)
    input_device = infer_input_device(model)
    hidden_cfg = config.get("hidden_states", {})
    max_pixels = hidden_cfg.get("max_pixels")
    clear_cache_every = int(hidden_cfg.get("clear_cuda_cache_every", 0) or 0)
    batch_size = max(1, int(hidden_cfg.get("batch_size", 1) or 1))
    prefix_tokens = int(
        safe_prefix_tokens
        if safe_prefix_tokens is not None
        else config.get("flow_matching", {}).get("target", {}).get("safe_answer_prefix_tokens", 16)
    )

    prepared_samples: List[Dict[str, Any]] = []
    skipped = []
    for sample in safe_samples:
        response = str(sample.get("response") or "").strip()
        if not response:
            skipped.append({"sample_id": sample.get("id"), "reason": "empty_safe_response"})
            continue
        item = dict(sample)
        item["assistant_prefix"] = _prefix_text(processor, response, prefix_tokens)
        item["safe_answer_prefix_tokens"] = prefix_tokens
        prepared_samples.append(item)

    if not prepared_samples:
        raise ValueError(f"All safe_neighbor responses are empty for split={split}; cannot extract safe prefix hidden.")

    counts = Counter(sample["sample_type"] for sample in prepared_samples)
    logger.info(
        "Extracting safe answer prefix hidden states for split=%s, samples=%s, prefix_tokens=%d, batch_size=%d",
        split,
        dict(counts),
        prefix_tokens,
        batch_size,
    )

    requested_layers = _requested_layers(config)
    selected_layers: Optional[List[int]] = None
    hidden_by_layer: Dict[int, List[torch.Tensor]] = {}
    metadata: List[Dict[str, Any]] = []
    failed_samples: List[Dict[str, Any]] = []
    processed_count = 0

    for batch in tqdm(_chunks(prepared_samples, batch_size), desc=f"extract {split} safe prefix hidden states"):
        try:
            inputs = (
                prepare_vl_inputs_with_assistant_prefix(
                    processor,
                    batch[0]["image_path"],
                    batch[0]["instruction"],
                    batch[0]["assistant_prefix"],
                    input_device,
                    max_pixels=max_pixels,
                )
                if len(batch) == 1
                else prepare_vl_batch_inputs_with_assistant_prefix(processor, batch, input_device, max_pixels=max_pixels)
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
                item_meta["token_position"] = "safe_answer_prefix_last_token"
                for layer in selected_layers:
                    vec = hidden_states[layer][row, last_pos, :].detach().to("cpu", dtype=torch.float32)
                    hidden_by_layer[layer].append(vec)
                metadata.append(item_meta)
                processed_count += 1
            del outputs, hidden_states, inputs
            if clear_cache_every and torch.cuda.is_available() and processed_count % clear_cache_every == 0:
                torch.cuda.empty_cache()
        except Exception as exc:
            message = str(exc)
            if "out of memory" in message.lower():
                message = f"{message}\n{cuda_oom_help()}"
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if len(batch) > 1:
                logger.warning("Failed safe-prefix batch of %d samples: %s. Retrying one by one.", len(batch), message)
                for sample in batch:
                    try:
                        inputs = prepare_vl_inputs_with_assistant_prefix(
                            processor,
                            sample["image_path"],
                            sample["instruction"],
                            sample["assistant_prefix"],
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
                        last_pos = _last_input_token_positions(inputs["attention_mask"])[0]
                        item_meta = dict(sample)
                        item_meta["last_token_position"] = last_pos
                        item_meta["input_length"] = int(inputs["attention_mask"][0].sum().item())
                        item_meta["token_position"] = "safe_answer_prefix_last_token"
                        for layer in selected_layers:
                            vec = hidden_states[layer][0, last_pos, :].detach().to("cpu", dtype=torch.float32)
                            hidden_by_layer[layer].append(vec)
                        metadata.append(item_meta)
                        processed_count += 1
                        del outputs, hidden_states, inputs
                    except Exception as single_exc:
                        logger.warning("Failed safe-prefix sample %s: %s", sample.get("id"), single_exc)
                        failed_samples.append({"sample": sample, "error": str(single_exc)})
            else:
                sample = batch[0]
                logger.warning("Failed safe-prefix sample %s: %s", sample.get("id"), message)
                failed_samples.append({"sample": sample, "error": message})

    if not metadata or selected_layers is None:
        raise RuntimeError(f"All safe-prefix samples failed for split={split}; no hidden states were saved.")

    output = {
        "metadata": metadata,
        "hidden_states": {layer: torch.stack(values, dim=0) for layer, values in hidden_by_layer.items()},
        "target_layers": selected_layers,
        "model_path": model_path_override or config.get("model", {}).get("local_path"),
        "token_position": "safe_answer_prefix_last_token",
        "safe_answer_prefix_tokens": prefix_tokens,
        "batch_size": batch_size,
        "skipped_samples": skipped,
        "failed_samples": failed_samples,
    }

    out_dir = ensure_dir(safe_prefix_hidden_dir(config))
    out_path = out_dir / f"{split}_safe_prefix_hidden_states.pt"
    torch.save(output, out_path)
    logger.info("Saved safe answer prefix hidden states to %s", out_path)

    save_json(
        {
            "split": split,
            "num_samples": len(metadata),
            "num_skipped": len(skipped),
            "num_failed": len(failed_samples),
            "target_layers": selected_layers,
            "safe_answer_prefix_tokens": prefix_tokens,
            "output_path": str(out_path),
        },
        out_dir / f"{split}_safe_prefix_hidden_states.summary.json",
    )
    return out_path
