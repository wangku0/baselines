from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch
from tqdm import tqdm

from .model_utils import infer_input_device
from .model_utils import uses_qwen_vision_utils
from .stage3_losses import _messages, prepare_prompt_inputs, process_vision_info
from .utils import cuda_oom_help, logger, write_jsonl


def _tokenizer_ids(processor, name: str) -> set[int]:
    tokenizer = getattr(processor, "tokenizer", processor)
    value = getattr(tokenizer, name, None)
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {int(v) for v in value if v is not None}
    return {int(value)}


def _decode_new_tokens(processor, inputs: Dict[str, Any], generated_ids: torch.Tensor) -> str:
    prompt_len = int(inputs["input_ids"].shape[1])
    new_ids = generated_ids[:, prompt_len:]
    decoded = processor.batch_decode(new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return decoded[0].strip() if decoded else ""


def _decode_batch_new_tokens(processor, inputs: Dict[str, Any], generated_ids: torch.Tensor) -> list[str]:
    prompt_len = int(inputs["input_ids"].shape[1])
    new_ids = generated_ids[:, prompt_len:]
    return [text.strip() for text in processor.batch_decode(new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)]


def _generation_metadata(processor, inputs: Dict[str, Any], generated_ids: torch.Tensor, max_new_tokens: int) -> list[Dict[str, Any]]:
    prompt_len = int(inputs["input_ids"].shape[1])
    new_ids = generated_ids[:, prompt_len:].detach().cpu()
    eos_ids = _tokenizer_ids(processor, "eos_token_id")
    pad_ids = _tokenizer_ids(processor, "pad_token_id")
    meta = []
    for seq in new_ids:
        ids = [int(x) for x in seq.tolist()]
        effective = [x for x in ids if x not in pad_ids]
        saw_eos = any(x in eos_ids for x in effective) if eos_ids else False
        meta.append(
            {
                "generated_token_count": len(effective),
                "hit_max_new_tokens": len(effective) >= int(max_new_tokens) and not saw_eos,
            }
        )
    return meta


def _batch_prompt_inputs(processor, samples: list[Dict[str, Any]], device, max_pixels: int):
    all_messages = [_messages(sample["image_path"], sample["instruction"], None, max_pixels=max_pixels) for sample in samples]
    texts = [processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for messages in all_messages]
    if uses_qwen_vision_utils(processor):
        flat_messages = [message for messages in all_messages for message in messages]
        image_inputs, video_inputs = process_vision_info(flat_messages)
        inputs = processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    else:
        from PIL import Image

        images = [Image.open(sample["image_path"]).convert("RGB") for sample in samples]
        inputs = processor(text=texts, images=images, padding=True, return_tensors="pt")
    try:
        return inputs.to(device)
    except Exception:
        for key, value in list(inputs.items()):
            if torch.is_tensor(value):
                inputs[key] = value.to(device)
        return inputs


def _generate_single_response_with_meta(
    model,
    processor,
    sample: Dict[str, Any],
    config: Dict[str, Any],
    device,
    max_pixels: int,
) -> tuple[str, Dict[str, Any]]:
    gen_cfg = config["stage3"]["evaluation"]
    max_new_tokens = int(gen_cfg.get("max_new_tokens", 768))
    inputs = prepare_prompt_inputs(
        processor,
        {"image_path": sample["image_path"], "instruction": sample["instruction"]},
        device,
        max_pixels=max_pixels,
    )
    kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": bool(gen_cfg.get("do_sample", False)),
        "top_p": float(gen_cfg.get("top_p", 1.0)),
    }
    if kwargs["do_sample"]:
        kwargs["temperature"] = max(float(gen_cfg.get("temperature", 0.0)), 1e-6)
    with torch.no_grad():
        ids = model.generate(**inputs, **kwargs)
    meta = _generation_metadata(processor, inputs, ids, max_new_tokens)[0]
    return _decode_new_tokens(processor, inputs, ids), meta


def generate_for_samples(
    model,
    processor,
    samples,
    config,
    split: str,
    out_path: Path,
    source: str,
    *,
    model_path: str | None = None,
) -> list[Dict[str, Any]]:
    gen_cfg = config["stage3"]["evaluation"]
    device = infer_input_device(model)
    max_pixels = config["stage3"].get("preprocessing", {}).get("max_pixels", 200704)
    batch_size = max(1, int(gen_cfg.get("generation_batch_size", gen_cfg.get("batch_size", 1))))
    tokenizer = getattr(processor, "tokenizer", None)
    if batch_size > 1 and tokenizer is not None:
        tokenizer.padding_side = "left"
    rows = []
    model.eval()
    for start in tqdm(range(0, len(samples), batch_size), desc=f"stage3 generate {split} {source}"):
        batch = samples[start : start + batch_size]
        batch_rows = [
            {
                "sample_id": sample["id"],
                "split": split,
                "sample_type": sample["sample_type"],
                "pair_id": sample.get("pair_id"),
                "category": sample.get("category"),
                "keyword": sample.get("keyword"),
                "image_path": sample.get("image_path"),
                "instruction": sample.get("instruction"),
                "reference_response": sample.get("response"),
                "generated_response": None,
                "generation_error": None,
                "generated_token_count": None,
                "hit_max_new_tokens": None,
                "model_path": model_path,
                "response_source": source,
            }
            for sample in batch
        ]
        try:
            inputs = _batch_prompt_inputs(processor, batch, device, max_pixels=max_pixels)
            max_new_tokens = int(gen_cfg.get("max_new_tokens", 768))
            kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": bool(gen_cfg.get("do_sample", False)),
                "top_p": float(gen_cfg.get("top_p", 1.0)),
            }
            if kwargs["do_sample"]:
                kwargs["temperature"] = max(float(gen_cfg.get("temperature", 0.0)), 1e-6)
            with torch.no_grad():
                ids = model.generate(**inputs, **kwargs)
            decoded = _decode_batch_new_tokens(processor, inputs, ids)
            metas = _generation_metadata(processor, inputs, ids, max_new_tokens)
            for row, text, meta in zip(batch_rows, decoded, metas):
                row["generated_response"] = text
                row.update(meta)
        except RuntimeError as exc:
            msg = str(exc)
            if "out of memory" in msg.lower():
                msg += "\n" + cuda_oom_help()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            logger.warning("Stage 3 batch generation failed for batch starting at %s: %s", start, msg)
            if batch_size > 1:
                for sample, row in zip(batch, batch_rows):
                    try:
                        text, meta = _generate_single_response_with_meta(model, processor, sample, config, device, max_pixels)
                        row["generated_response"] = text
                        row.update(meta)
                    except Exception as inner_exc:
                        row["generation_error"] = str(inner_exc)
                        logger.warning("Stage 3 fallback generation failed for %s: %s", sample["id"], inner_exc)
            else:
                for row in batch_rows:
                    row["generation_error"] = msg
        except Exception as exc:
            logger.warning("Stage 3 batch generation failed for batch starting at %s: %s", start, exc)
            if batch_size > 1:
                for sample, row in zip(batch, batch_rows):
                    try:
                        text, meta = _generate_single_response_with_meta(model, processor, sample, config, device, max_pixels)
                        row["generated_response"] = text
                        row.update(meta)
                    except Exception as inner_exc:
                        row["generation_error"] = str(inner_exc)
                        logger.warning("Stage 3 fallback generation failed for %s: %s", sample["id"], inner_exc)
            else:
                for row in batch_rows:
                    row["generation_error"] = str(exc)
        rows.extend(batch_rows)
    write_jsonl(rows, out_path)
    return rows
