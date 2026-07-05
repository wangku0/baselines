from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..utils import logger, resolve_path, save_json

VALID_RECOMMENDED_SCORE_MODES = {
    "raw",
    "centered",
    "raw_positive",
    "centered_positive",
    "raw_signed",
    "centered_signed",
    "paired_delta",
    "paired_delta_positive",
    "paired_delta_signed",
}


@dataclass
class RecommendedRiskConfig:
    recommended_k: int
    recommended_score_mode: str
    recommended_hidden_layers: List[int]
    lora_train_layers: List[int]
    risk_basis_path: str
    normalization_config: Dict[str, Any]
    source_path: str
    recommended_config_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def hidden_layers_to_lora_layers(hidden_layers: List[int]) -> List[int]:
    layers = sorted({int(layer) for layer in hidden_layers})
    if any(layer <= 0 for layer in layers):
        raise ValueError(
            f"Invalid hidden layer(s) {layers}. hidden_states[0] is the embedding output, "
            "so LoRA train layer is hidden_layer - 1 and hidden layer must be > 0."
        )
    return sorted({layer - 1 for layer in layers})


def _stage1_5_paths(config: Dict[str, Any]) -> tuple[Path, Path]:
    metrics_root = resolve_path(config, config.get("outputs", {}).get("metrics_dir", "integrations/my_method/outputs/metrics"))
    return (
        metrics_root / "stage1_5" / "recommended_config.json",
        metrics_root / "stage1_5" / "recommended_layers.json",
    )


def _risk_basis_path(config: Dict[str, Any], k: int) -> Path:
    return resolve_path(config, config.get("outputs", {}).get("risk_space_dir", "integrations/my_method/outputs/risk_space")) / f"k_{k}" / "risk_basis.pt"


def load_recommended_risk_config(
    config: Dict[str, Any],
    *,
    recommended_config_path: Optional[str] = None,
    allow_fallback: bool = False,
) -> RecommendedRiskConfig:
    rec_cfg_path, rec_layers_path = _stage1_5_paths(config)
    if recommended_config_path:
        rec_layers_path = resolve_path(config, recommended_config_path)

    if not rec_layers_path.exists():
        if not allow_fallback:
            raise FileNotFoundError(
                f"Stage 1.5 recommended config file not found: {rec_layers_path}. "
                "Run scripts/04_stage1_5_analysis.py first."
            )
        logger.warning("Using fallback recommended config; Stage 1.5 recommended config file not found.")
        fallback_layers = [int(x) for x in config.get("stage3", {}).get("risk_space", {}).get("risk_layers", [20])]
        fallback_k = int(config.get("stage3", {}).get("risk_space", {}).get("k", 2))
        fallback_mode = config.get("stage3", {}).get("risk_space", {}).get("score_mode", "centered")
        return RecommendedRiskConfig(
            recommended_k=fallback_k,
            recommended_score_mode=fallback_mode,
            recommended_hidden_layers=fallback_layers,
            lora_train_layers=hidden_layers_to_lora_layers(fallback_layers),
            risk_basis_path=str(_risk_basis_path(config, fallback_k)),
            normalization_config=config.get("stage2", {}).get("normalization", {}),
            source_path=str(rec_layers_path),
            recommended_config_path=str(rec_cfg_path) if rec_cfg_path.exists() else None,
        )

    with rec_layers_path.open("r", encoding="utf-8") as f:
        rec_layers = json.load(f)
    rec_cfg = {}
    if rec_cfg_path.exists():
        with rec_cfg_path.open("r", encoding="utf-8") as f:
            rec_cfg = json.load(f)

    layers = rec_layers.get("recommended_layers")
    if not layers:
        raise ValueError(f"{rec_layers_path} does not contain recommended_layers.")
    k = int(rec_layers.get("recommended_k", rec_cfg.get("recommended_k", config.get("risk_space", {}).get("k", 2))))
    score_mode = rec_layers.get(
        "recommended_score_mode",
        rec_cfg.get("recommended_score_mode", config.get("stage2", {}).get("implicit_risk", {}).get("score_mode", "centered")),
    )
    if score_mode not in VALID_RECOMMENDED_SCORE_MODES:
        raise ValueError(
            f"Unsupported recommended_score_mode={score_mode!r}; "
            f"expected one of {sorted(VALID_RECOMMENDED_SCORE_MODES)}."
        )
    hidden_layers = sorted({int(layer) for layer in layers})
    risk_basis_path = _risk_basis_path(config, k)
    return RecommendedRiskConfig(
        recommended_k=k,
        recommended_score_mode=score_mode,
        recommended_hidden_layers=hidden_layers,
        lora_train_layers=hidden_layers_to_lora_layers(hidden_layers),
        risk_basis_path=str(risk_basis_path),
        normalization_config=config.get("stage2", {}).get("normalization", {}),
        source_path=str(rec_layers_path),
        recommended_config_path=str(rec_cfg_path) if rec_cfg_path.exists() else None,
    )


def save_resolved_recommended_config(resolved: RecommendedRiskConfig, path: Path) -> None:
    save_json(resolved.to_dict(), path)
