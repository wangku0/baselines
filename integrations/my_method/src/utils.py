import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import yaml


LOGGER_NAME = "risk_space_stage1"

DATASET_PRESETS = {
    "sex": ("dataset/Sex_all_train.json", "dataset/Sex_all_val.json"),
    "violence": ("dataset/all_train.json", "dataset/all_val.json"),
    "hatespeech": ("dataset/HateSpeech_all_train.json", "dataset/HateSpeech_all_val.json"),
    "illegalactivity": ("dataset/IllegalActivity_all_train.json", "dataset/IllegalActivity_all_val.json"),
    "privacy": ("dataset/Privacy_all_train.json", "dataset/Privacy_all_val.json"),
    "weapon": ("dataset/Weapon_all_train.json", "dataset/Weapon_all_val.json"),
}
DATASET_ALIASES = {
    "hate_speech": "hatespeech",
    "hate-speech": "hatespeech",
    "illegal_activity": "illegalactivity",
    "illegal-activity": "illegalactivity",
}
DATASET_CHOICES = (
    "config",
    "sex",
    "violence",
    "hatespeech",
    "hate_speech",
    "hate-speech",
    "illegalactivity",
    "illegal_activity",
    "illegal-activity",
    "privacy",
    "weapon",
    "original",
)


def setup_logger(name: str = LOGGER_NAME, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    return logger


logger = setup_logger()


def load_config(config_path: str) -> Dict[str, Any]:
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    config["_config_path"] = str(path)
    return config


def add_dataset_argument(parser: Any) -> None:
    parser.add_argument(
        "--dataset",
        choices=DATASET_CHOICES,
        default="config",
        help=(
            "Override dataset paths for this run: sex, violence, hatespeech, illegalactivity, privacy, "
            "and weapon use their matching dataset files. config keeps the YAML paths. "
            "'original' is a backward-compatible alias for violence. "
            "Do not reuse generic outputs produced by another dataset."
        ),
    )


def apply_dataset_preset(config: Dict[str, Any], preset: str) -> str:
    normalized = "violence" if preset == "original" else preset
    normalized = DATASET_ALIASES.get(normalized, normalized)
    if normalized != "config":
        train_file, val_file = DATASET_PRESETS[normalized]
        dataset_cfg = config.setdefault("dataset", {})
        dataset_cfg["train_file"] = train_file
        dataset_cfg["val_file"] = val_file
    dataset_cfg = config.setdefault("dataset", {})
    logger.info(
        "Using dataset=%s train_file=%s val_file=%s",
        dataset_label(config),
        dataset_cfg.get("train_file"),
        dataset_cfg.get("val_file"),
    )
    return normalized


def dataset_label(config: Dict[str, Any]) -> str:
    dataset_cfg = config.get("dataset", {})
    train_file = str(dataset_cfg.get("train_file", "")).lower()
    val_file = str(dataset_cfg.get("val_file", "")).lower()
    combined = f"{train_file} {val_file}"
    known_labels = {
        "sex_all": "sex",
        "hatespeech_all": "hatespeech",
        "illegalactivity_all": "illegalactivity",
        "privacy_all": "privacy",
        "weapon_all": "weapon",
    }
    for marker, label in known_labels.items():
        if marker in combined:
            return label
    if Path(train_file).name == "all_train.json" or Path(val_file).name == "all_val.json":
        return "violence"
    stem = Path(dataset_cfg.get("val_file") or dataset_cfg.get("train_file") or "dataset").stem
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in stem).strip("_") or "dataset"


def get_project_root(config: Dict[str, Any]) -> Path:
    raw_root = config.get("project_root", ".")
    root = Path(raw_root).expanduser()
    if not root.is_absolute():
        config_path = Path(config.get("_config_path", ".")).resolve()
        # Scripts are expected to run from project root. If the config was loaded
        # from configs/, resolving against cwd keeps the documented workflow sane.
        root = Path.cwd() / root
    return root.resolve()


def resolve_path(config: Dict[str, Any], path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return (get_project_root(config) / path).resolve()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_output_dirs(config: Dict[str, Any]) -> None:
    outputs = config.get("outputs", {})
    for key in ["hidden_states_dir", "risk_space_dir", "metrics_dir", "figures_dir"]:
        if key in outputs:
            ensure_dir(resolve_path(config, outputs[key]))


def save_json(data: Any, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[Dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning("Skipping invalid JSONL line %d in %s: %s", line_no, path, exc)
    return records


def write_jsonl(records: Iterable[Dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def get_by_path(config: Dict[str, Any], keys: Iterable[str], default: Optional[Any] = None) -> Any:
    cur: Any = config
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def cuda_oom_help() -> str:
    return (
        "CUDA OOM. Suggestions: use Qwen2-VL-2B instead of Qwen2.5-VL-3B; "
        "use float16 or bfloat16; reduce --max_samples; check whether other "
        "processes are occupying GPU memory."
    )
