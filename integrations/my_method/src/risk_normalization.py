from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd

from .evaluate_risk_space import evaluate_risk_space
from .risk_space.recommended_config import load_recommended_risk_config
from .risk_space.transport_layer_selection import select_layers_by_risk_transport
from .utils import ensure_dir, logger, resolve_path, save_json


def _stage2_metrics_dir(config: Dict[str, Any]) -> Path:
    return ensure_dir(resolve_path(config, config["stage2"]["outputs"]["metrics_dir"]))


def resolve_implicit_settings(config: Dict[str, Any]) -> Dict[str, Any]:
    settings = dict(config.get("stage2", {}).get("implicit_risk", {}))
    if settings.get("use_stage1_5_recommended", True):
        metrics_root = resolve_path(config, config["outputs"]["metrics_dir"])
        rec_cfg = metrics_root / "stage1_5" / "recommended_config.json"
        rec_layers = metrics_root / "stage1_5" / "recommended_layers.json"
        if rec_cfg.exists():
            data = json.load(rec_cfg.open("r", encoding="utf-8"))
            settings["k"] = int(data.get("recommended_k", settings.get("k", 2)))
            settings["score_mode"] = data.get("recommended_score_mode", settings.get("score_mode", "centered"))
        else:
            logger.warning("Missing %s; falling back to config stage2.implicit_risk.", rec_cfg)
        if rec_layers.exists():
            data = json.load(rec_layers.open("r", encoding="utf-8"))
            settings["layers"] = data.get("recommended_layers", settings.get("layers", [12, 16, 20, 24]))
            settings["score_mode"] = data.get("recommended_score_mode", settings.get("score_mode", "centered"))
            settings["k"] = int(data.get("recommended_k", settings.get("k", 2)))
        else:
            logger.warning("Missing %s; falling back to config stage2.implicit_risk layers.", rec_layers)
    layer_selection = config.get("stage3", {}).get("layer_selection", {})
    layer_method = str(layer_selection.get("method", "") or "").lower()
    if settings.get("use_stage3_selected_layers", False) or layer_method == "risk_transport_influence":
        rec = load_recommended_risk_config(
            config,
            allow_fallback=bool(config.get("flow_matching", {}).get("recommended_config", {}).get("allow_fallback", False)),
        )
        selection = select_layers_by_risk_transport(config, rec)
        settings["layers"] = selection.selected_hidden_layers
        settings["k"] = int(rec.recommended_k)
        settings["score_mode"] = rec.recommended_score_mode
        settings["risk_basis_path"] = rec.risk_basis_path
        settings["layer_selection_method"] = "risk_transport_influence"
        settings["layer_selection_output_path"] = selection.output_path
        settings["transport_target"] = selection.transport_target
        settings["use_stage1_5_recommended"] = False
        settings["use_stage3_selected_layers"] = True
    settings["layers"] = [int(layer) for layer in settings.get("layers", [12, 16, 20, 24])]
    settings["k"] = int(settings.get("k", 2))
    settings["score_mode"] = settings.get("score_mode", "centered")
    return settings


def _basis_path(config: Dict[str, Any], settings: Dict[str, Any]) -> Path:
    if settings.get("risk_basis_path"):
        return resolve_path(config, settings["risk_basis_path"])
    return resolve_path(config, config["outputs"]["risk_space_dir"]) / f"k_{settings['k']}" / "risk_basis.pt"


def _find_stage1_5_scores(config: Dict[str, Any], split: str, settings: Dict[str, Any]) -> Path | None:
    if settings.get("use_stage3_selected_layers") or settings.get("layer_selection_method"):
        return None
    metrics_root = resolve_path(config, config["outputs"]["metrics_dir"])
    candidates = [
        metrics_root / "stage1_5" / f"k_{settings['k']}" / f"{split}_risk_scores_{settings['score_mode']}.csv",
        metrics_root / "stage1_5" / f"k_{settings['k']}" / f"{split}_risk_scores_implicit_{settings['score_mode']}.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _load_or_compute_implicit(config: Dict[str, Any], split: str, settings: Dict[str, Any]) -> tuple[pd.DataFrame, str]:
    existing = _find_stage1_5_scores(config, split, settings)
    if existing is not None:
        return pd.read_csv(existing), str(existing)
    basis_path = _basis_path(config, settings)
    if not basis_path.exists():
        raise FileNotFoundError(
            f"Risk basis not found: {basis_path}. Run scripts/04_stage1_5_analysis.py first; "
            "Stage 2 will not re-extract hidden states."
        )
    df = evaluate_risk_space(
        config,
        split=split,
        risk_basis_path=str(basis_path),
        score_mode=settings["score_mode"],
        target_layers_override=settings["layers"],
        output_dir=_stage2_metrics_dir(config),
        output_suffix="implicit",
        generate_figures=False,
    )
    return df, f"computed:{basis_path}"


def normalize_implicit_risk(config: Dict[str, Any], splits: Iterable[str] = ("train", "val")) -> Dict[str, pd.DataFrame]:
    settings = resolve_implicit_settings(config)
    norm_cfg = config["stage2"]["normalization"]
    metrics_dir = _stage2_metrics_dir(config)
    raw = {}
    sources = {}
    for split in set(list(splits) + ["train"]):
        raw[split], sources[split] = _load_or_compute_implicit(config, split, settings)

    train_scores = raw["train"]["R_total"].dropna().astype(float).values
    if len(train_scores) == 0:
        raise ValueError("No train R_total values available for implicit risk normalization.")
    lower_p = float(norm_cfg.get("lower_percentile", 5))
    upper_p = float(norm_cfg.get("upper_percentile", 95))
    lower = float(np.percentile(train_scores, lower_p))
    upper = float(np.percentile(train_scores, upper_p))
    if upper <= lower:
        logger.warning("Percentile normalization upper <= lower; using train min/max fallback.")
        lower = float(np.min(train_scores))
        upper = float(np.max(train_scores))
    if upper <= lower:
        upper = lower + 1e-6

    save_json(
        {
            "method": norm_cfg.get("method", "train_minmax_percentile"),
            "lower_percentile": lower_p,
            "upper_percentile": upper_p,
            "lower_value": lower,
            "upper_value": upper,
            "train_count": int(len(train_scores)),
            "created_from": sources,
            "implicit_settings": settings,
        },
        metrics_dir / "implicit_normalization.json",
    )

    outputs = {}
    for split in splits:
        df = raw[split].copy()
        df = df.rename(columns={"R_total": "R_implicit_raw"})
        df["R_implicit_norm_before_clip"] = (df["R_implicit_raw"] - lower) / (upper - lower)
        df["R_implicit_norm"] = df["R_implicit_norm_before_clip"]
        if norm_cfg.get("clip", True):
            df["R_implicit_norm"] = df["R_implicit_norm"].clip(0.0, 1.0)
        layer_cols = [f"R_layer_{layer}" for layer in settings["layers"] if f"R_layer_{layer}" in df.columns]
        keep = [
            "sample_id",
            "split",
            "sample_type",
            "pair_id",
            "category",
            "keyword",
            "R_implicit_raw",
            "R_implicit_norm_before_clip",
            "R_implicit_norm",
        ] + layer_cols
        df = df[keep]
        df.to_csv(metrics_dir / f"{split}_implicit_risk_normalized.csv", index=False, encoding="utf-8-sig")
        outputs[split] = df
    return outputs
