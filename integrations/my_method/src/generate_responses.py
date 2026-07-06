from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch
from tqdm import tqdm

from .data_loader import RETAIN_FIELDS, _load_json_tolerant, load_dataset
from .model_utils import infer_input_device, load_model_and_processor, prepare_vl_inputs
from .utils import cuda_oom_help, logger, read_jsonl, resolve_path, write_jsonl


def generation_path(config: Dict[str, Any], split: str) -> Path:
    out_dir = resolve_path(config, config["stage2"]["outputs"]["generations_dir"])
    return out_dir / f"{split}_generations.jsonl"


def _model_config_for_generation(config: Dict[str, Any], model_path_override: Optional[str]) -> Dict[str, Any]:
    cfg = copy.deepcopy(config)
    gen_cfg = cfg.get("stage2", {}).get("generation", {})
    cfg.setdefault("model", {})["local_path"] = model_path_override or gen_cfg.get("model_path") or cfg["model"]["local_path"]
    return cfg


def _dataset_file(config: Dict[str, Any], split: str) -> Path:
    key = "train_file" if split == "train" else "val_file"
    return resolve_path(config, config["dataset"][key])


def _retain_answer_and_prediction(config: Dict[str, Any], split: str) -> dict[tuple[int, str], tuple[Any, Any]]:
    records = _load_json_tolerant(_dataset_file(config, split))
    lookup: dict[tuple[int, str], tuple[Any, Any]] = {}
    for idx, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        for field in RETAIN_FIELDS:
            item = record.get(field)
            if not isinstance(item, dict):
                continue
            answer = item.get("Answer") or item.get("answer")
            prediction = item.get("Prediction") or item.get("prediction")
            if prediction is None:
                for key, value in item.items():
                    if "predict" in str(key).lower() and value not in (None, ""):
                        prediction = value
                        break
            lookup[(idx, field)] = (answer, prediction)
    return lookup


def _existing_matches_source(path: Path, response_source: str, expected_count: int) -> bool:
    if not path.exists():
        return False
    existing = read_jsonl(path)
    if len(existing) < expected_count:
        return False
    return all(record.get("response_source") == response_source for record in existing[:expected_count])


def load_dataset_responses_for_split(
    config: Dict[str, Any],
    split: str,
    *,
    max_samples: Optional[int] = None,
    force_regenerate: bool = False,
) -> Path:
    """Create generation-format JSONL from responses already stored in the dataset."""
    out_path = generation_path(config, split)
    samples = load_dataset(config, split=split, max_samples=max_samples)
    expected_count = len(samples)
    reuse = bool(config.get("stage2", {}).get("generation", {}).get("reuse_existing_generations", True)) and not force_regenerate
    if reuse and _existing_matches_source(out_path, "dataset", expected_count):
        logger.info("Reusing dataset response file for split=%s from %s", split, out_path)
        return out_path

    retain_lookup = _retain_answer_and_prediction(config, split)
    records = []
    for sample in samples:
        reference_response = None
        generated_response = sample.get("response")
        generation_error = None
        if sample.get("sample_type") == "retain":
            answer, prediction = retain_lookup.get((int(sample.get("sample_index", -1)), sample.get("source_field")), (None, None))
            reference_response = answer
            generated_response = prediction
        if generated_response in (None, ""):
            generation_error = "missing_dataset_response"
        records.append(
            {
                "sample_id": sample["id"],
                "split": split,
                "sample_type": sample["sample_type"],
                "pair_id": sample.get("pair_id"),
                "category": sample.get("category"),
                "keyword": sample.get("keyword"),
                "image_path": sample.get("image_path"),
                "instruction": sample.get("instruction"),
                "reference_response": reference_response,
                "generated_response": generated_response,
                "generation_error": generation_error,
                "model_path": None,
                "response_source": "dataset",
            }
        )
    write_jsonl(records, out_path)
    logger.info("Saved %d dataset response records to %s", len(records), out_path)
    return out_path


def _decode_new_tokens(processor, inputs: Dict[str, Any], generated_ids: torch.Tensor) -> str:
    prompt_len = int(inputs["input_ids"].shape[1])
    new_ids = generated_ids[:, prompt_len:]
    decoded = processor.batch_decode(new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return decoded[0].strip() if decoded else ""


def generate_responses_for_split(
    config: Dict[str, Any],
    split: str,
    *,
    max_samples: Optional[int] = None,
    force_regenerate: bool = False,
    model_path_override: Optional[str] = None,
) -> Path:
    gen_cfg = config.get("stage2", {}).get("generation", {})
    out_path = generation_path(config, split)
    reuse = bool(gen_cfg.get("reuse_existing_generations", True)) and not force_regenerate
    if reuse and out_path.exists():
        existing = read_jsonl(out_path)
        expected_count = max_samples
        if expected_count is None:
            expected_count = len(load_dataset(config, split=split))
        if len(existing) >= expected_count and all(record.get("response_source") == "generated" for record in existing[:expected_count]):
            logger.info("Reusing existing generations for split=%s from %s", split, out_path)
            return out_path
        logger.warning(
            "Existing generations for split=%s contain %d rows, but %d are needed; regenerating.",
            split,
            len(existing),
            expected_count,
        )

    samples = load_dataset(config, split=split, max_samples=max_samples)
    model_cfg = _model_config_for_generation(config, model_path_override)
    model, processor = load_model_and_processor(model_cfg)
    device = infer_input_device(model)
    model_path = model_path_override or gen_cfg.get("model_path") or config["model"]["local_path"]

    records = []
    for sample in tqdm(samples, desc=f"generate {split} responses"):
        record = {
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
            "model_path": model_path,
            "response_source": "generated",
        }
        try:
            inputs = prepare_vl_inputs(processor, sample["image_path"], sample["instruction"], device)
            generate_kwargs = {
                "max_new_tokens": int(gen_cfg.get("max_new_tokens", 256)),
                "do_sample": bool(gen_cfg.get("do_sample", False)),
                "top_p": float(gen_cfg.get("top_p", 1.0)),
            }
            if generate_kwargs["do_sample"]:
                generate_kwargs["temperature"] = max(float(gen_cfg.get("temperature", 0.0)), 1e-6)
            with torch.no_grad():
                generated_ids = model.generate(**inputs, **generate_kwargs)
            record["generated_response"] = _decode_new_tokens(processor, inputs, generated_ids)
        except RuntimeError as exc:
            message = str(exc)
            if "out of memory" in message.lower():
                message = f"{message}\n{cuda_oom_help()}"
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            logger.warning("Generation failed for sample_id=%s: %s", sample["id"], message)
            record["generation_error"] = message
        except Exception as exc:
            logger.warning("Generation failed for sample_id=%s: %s", sample["id"], exc)
            record["generation_error"] = str(exc)
        records.append(record)

    write_jsonl(records, out_path)
    logger.info("Saved %d generation records to %s", len(records), out_path)
    return out_path


def ensure_generation_files(
    config: Dict[str, Any],
    splits: Iterable[str],
    *,
    max_samples: Optional[int] = None,
    skip_generation: bool = False,
    force_regenerate: bool = False,
    model_path_override: Optional[str] = None,
) -> Dict[str, Path]:
    paths = {}
    response_source = config.get("stage2", {}).get("response_source", "dataset")
    if response_source not in {"dataset", "generate"}:
        raise ValueError(f"stage2.response_source must be 'dataset' or 'generate', got {response_source}")
    for split in splits:
        path = generation_path(config, split)
        if skip_generation:
            if not path.exists():
                raise FileNotFoundError(f"Missing generations file for split={split}: {path}")
            expected_count = max_samples
            if expected_count is None:
                expected_count = len(load_dataset(config, split=split))
            if not _existing_matches_source(path, response_source, expected_count):
                raise ValueError(
                    f"Existing generations file for split={split} does not match "
                    f"response_source={response_source!r} or has fewer than "
                    f"{expected_count} records: {path}. Rerun without "
                    "--skip_generation to rebuild it from the configured source."
                )
            paths[split] = path
        elif response_source == "dataset":
            paths[split] = load_dataset_responses_for_split(
                config,
                split,
                max_samples=max_samples,
                force_regenerate=force_regenerate,
            )
        else:
            paths[split] = generate_responses_for_split(
                config,
                split,
                max_samples=max_samples,
                force_regenerate=force_regenerate,
                model_path_override=model_path_override,
            )
    return paths
