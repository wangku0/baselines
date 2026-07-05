import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .utils import logger, resolve_path


RETAIN_FIELDS = [
    "UnharmPair_text1",
    "UnharmPair_text2",
    "UnharmPair_image1",
    "UnharmPair_image2",
]


def _escape_inner_quotes_in_value_line(line: str) -> str:
    """Repair common hand-edited JSON lines like: "Question": "say "hi"",."""
    match = re.match(r'^(\s*"[^"]+"\s*:\s*")(.*)("\s*,?\s*)$', line)
    if not match:
        return line
    prefix, value, suffix = match.groups()
    repaired = []
    for i, ch in enumerate(value):
        if ch == '"' and (i == 0 or value[i - 1] != "\\"):
            repaired.append('\\"')
        else:
            repaired.append(ch)
    return prefix + "".join(repaired) + suffix


def _merge_broken_string_lines(lines: List[str]) -> List[str]:
    """Merge accidental physical line breaks inside JSON string values.

    The provided dataset has examples where a value such as
    "Prediction": "... movie "Title,
    " continuation ..."
    is split across two lines. This repairs that common pattern before JSON
    parsing without changing the source file on disk.
    """
    merged: List[str] = []
    i = 0
    start_re = re.compile(r'^\s*"[^"]+"\s*:\s*"')
    complete_re = re.compile(r'"\s*,?\s*$')
    while i < len(lines):
        line = lines[i]
        if start_re.match(line) and not complete_re.search(line):
            current = line.rstrip()
            i += 1
            while i < len(lines):
                continuation = lines[i].strip()
                if continuation.startswith('"') and not re.match(r'^"[^"]+"\s*:', continuation):
                    continuation = continuation[1:]
                current = current + "\\n" + continuation
                if complete_re.search(current):
                    break
                i += 1
            merged.append(current)
        else:
            merged.append(line)
        i += 1
    return merged


def _load_json_tolerant(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"Expected a list in {path}, got {type(data)}")
        return data
    except json.JSONDecodeError as exc:
        logger.warning("Strict JSON parsing failed for %s: %s. Trying tolerant repair.", path, exc)

    merged_lines = _merge_broken_string_lines(text.splitlines())
    repaired_lines = [_escape_inner_quotes_in_value_line(line) for line in merged_lines]
    repaired = "\n".join(repaired_lines)
    repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)
    try:
        data = json.loads(repaired)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Could not parse {path}. The file is not valid JSON and automatic repair failed: {exc}"
        ) from exc
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}, got {type(data)}")
    logger.warning("Loaded %s after tolerant JSON repair. Original file was not modified.", path)
    return data


def _first_present(mapping: Dict[str, Any], keys: Iterable[str]) -> Optional[Any]:
    for key in keys:
        if key in mapping and mapping[key] not in (None, ""):
            return mapping[key]
    return None


def _prediction_value(mapping: Dict[str, Any]) -> Optional[Any]:
    value = _first_present(mapping, ["Answer", "answer"])
    if value is not None:
        return value
    value = _first_present(mapping, ["Prediction", "prediction"])
    if value is not None:
        return value
    # Some records contain typos such as "Predicti1on".
    for key, item in mapping.items():
        if "predict" in str(key).lower() and item not in (None, ""):
            return item
    return None


def _image_path_for_record(config: Dict[str, Any], image_id: Optional[str]) -> Optional[str]:
    if not image_id:
        logger.warning("Skipping record with missing image_id.")
        return None
    image_root = config.get("dataset", {}).get("image_root", "dataset")
    base = resolve_path(config, image_root)
    candidate = (base / image_id).resolve()
    if not candidate.exists():
        logger.warning("Image not found, skipping corresponding samples: %s", candidate)
        return None
    return str(candidate)


def _make_common(
    *,
    split: str,
    sample_index: int,
    record: Dict[str, Any],
    image_path: str,
    instruction: str,
    response: Optional[str],
    sample_type: str,
    pair_id: Optional[str],
    sample_id: str,
    source_field: str,
) -> Dict[str, Any]:
    return {
        "id": sample_id,
        "split": split,
        "image_path": image_path,
        "instruction": instruction,
        "response": response,
        "category": record.get("category", ""),
        "keyword": record.get("keyword", ""),
        "sample_type": sample_type,
        "pair_id": pair_id,
        "source_field": source_field,
        "raw_type": record.get("type", ""),
        "sample_index": sample_index,
    }


def _samples_from_record(config: Dict[str, Any], record: Dict[str, Any], split: str, sample_index: int) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    image_path = _image_path_for_record(config, record.get("image_id"))
    if image_path is None:
        return samples

    unsafe_pairs = record.get("unsafe_pairs") or []
    safe_pairs = record.get("safeNb_pairs") or []
    if not isinstance(unsafe_pairs, list):
        logger.warning("%s_%s unsafe_pairs is not a list; ignoring.", split, sample_index)
        unsafe_pairs = []
    if not isinstance(safe_pairs, list):
        logger.warning("%s_%s safeNb_pairs is not a list; ignoring.", split, sample_index)
        safe_pairs = []
    if len(unsafe_pairs) != len(safe_pairs):
        logger.warning(
            "%s_%s has unmatched unsafe/safe pairs: unsafe=%d, safe=%d. Keeping paired indices only.",
            split,
            sample_index,
            len(unsafe_pairs),
            len(safe_pairs),
        )

    for i in range(min(len(unsafe_pairs), len(safe_pairs))):
        unsafe = unsafe_pairs[i] or {}
        safe = safe_pairs[i] or {}
        harmful_question = unsafe.get("question")
        safe_question = safe.get("question")
        if not harmful_question or not safe_question:
            logger.warning("%s_%s pair_%d missing question; skipping pair.", split, sample_index, i)
            continue
        pair_id = f"{split}_{sample_index}_pair_{i}"
        samples.append(
            _make_common(
                split=split,
                sample_index=sample_index,
                record=record,
                image_path=image_path,
                instruction=harmful_question,
                response=unsafe.get("model_response"),
                sample_type="harmful_trigger",
                pair_id=pair_id,
                sample_id=f"{pair_id}_harmful",
                source_field="unsafe_pairs",
            )
        )
        samples.append(
            _make_common(
                split=split,
                sample_index=sample_index,
                record=record,
                image_path=image_path,
                instruction=safe_question,
                response=safe.get("model_response"),
                sample_type="safe_neighbor",
                pair_id=pair_id,
                sample_id=f"{pair_id}_safe",
                source_field="safeNb_pairs",
            )
        )

    for field_name in RETAIN_FIELDS:
        item = record.get(field_name)
        if not isinstance(item, dict):
            logger.warning("%s_%s missing retain field %s; skipping.", split, sample_index, field_name)
            continue
        question = item.get("Question") or item.get("question")
        if not question:
            logger.warning("%s_%s retain field %s has no Question; skipping.", split, sample_index, field_name)
            continue
        samples.append(
            _make_common(
                split=split,
                sample_index=sample_index,
                record=record,
                image_path=image_path,
                instruction=question,
                response=_prediction_value(item),
                sample_type="retain",
                pair_id=None,
                sample_id=f"{split}_{sample_index}_retain_{field_name}",
                source_field=field_name,
            )
        )
    return samples


def load_dataset(
    config: Dict[str, Any],
    split: str = "train",
    sample_types: Optional[Iterable[str]] = None,
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if split == "both":
        data = load_dataset(config, split="train", sample_types=sample_types, max_samples=None)
        data.extend(load_dataset(config, split="val", sample_types=sample_types, max_samples=None))
        return data[:max_samples] if max_samples is not None else data
    if split not in {"train", "val"}:
        raise ValueError(f"split must be train, val, or both; got {split}")

    dataset_cfg = config.get("dataset", {})
    file_key = "train_file" if split == "train" else "val_file"
    path = resolve_path(config, dataset_cfg.get(file_key, f"dataset/all_{split}.json"))
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    records = _load_json_tolerant(path)
    samples: List[Dict[str, Any]] = []
    for idx, record in enumerate(records):
        if not isinstance(record, dict):
            logger.warning("%s record %d is not a dict; skipping.", split, idx)
            continue
        samples.extend(_samples_from_record(config, record, split, idx))

    if sample_types is not None:
        allowed = set(sample_types)
        samples = [sample for sample in samples if sample.get("sample_type") in allowed]
    if max_samples is not None:
        samples = samples[: int(max_samples)]

    counts: Dict[str, int] = {}
    for sample in samples:
        counts[sample["sample_type"]] = counts.get(sample["sample_type"], 0) + 1
    for sample_type in ["harmful_trigger", "safe_neighbor", "retain"]:
        if counts.get(sample_type, 0) == 0:
            logger.warning("No %s samples loaded for split=%s.", sample_type, split)
    logger.info("Loaded %d samples for split=%s: %s", len(samples), split, counts)
    return samples
