from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import torch

from .build_risk_space import build_risk_space
from .evaluate_risk_space import evaluate_risk_space
from .utils import ensure_dir, logger, resolve_path, save_json
from .visualize import (
    plot_group_layer_val_auc,
    plot_k_sweep_val_auc,
    plot_k_sweep_val_means,
    plot_single_layer_val_auc,
)

VALID_SCORE_MODES = {
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


def _ensure_hidden_states_exist(config: Dict[str, Any]) -> None:
    hidden_dir = resolve_path(config, config["outputs"]["hidden_states_dir"])
    missing = []
    for split in ["train", "val"]:
        path = hidden_dir / f"{split}_hidden_states.pt"
        if not path.exists():
            missing.append(str(path))
    if missing:
        raise FileNotFoundError(
            "Stage 1.5 reuses saved hidden states and does not load the model. "
            "Missing hidden-state files: "
            + "; ".join(missing)
            + ". Run scripts/01_extract_hidden_states.py for train and val first."
        )


def _load_target_layers(config: Dict[str, Any]) -> List[int]:
    hidden_path = resolve_path(config, config["outputs"]["hidden_states_dir"]) / "train_hidden_states.pt"
    data = torch.load(hidden_path, map_location="cpu", weights_only=False)
    return [int(layer) for layer in data["target_layers"]]


def _metric(eval_result: Dict[str, Any], path: List[str], default: Optional[float] = None) -> Optional[float]:
    cur: Any = eval_result
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur if cur is not None else default


def _summary_row(
    eval_result: Dict[str, Any],
    *,
    k: int,
    score_mode: str,
    split: str,
    layer_set_name: Optional[str] = None,
    layers: Optional[List[int]] = None,
) -> Dict[str, Any]:
    stats = eval_result.get("stats_by_sample_type", {})
    harmful_mean = _metric(stats, ["harmful_trigger", "mean"])
    safe_mean = _metric(stats, ["safe_neighbor", "mean"])
    retain_mean = _metric(stats, ["retain", "mean"])
    paired = eval_result.get("paired_analysis", {})
    row = {
        "k": int(k),
        "score_mode": score_mode,
        "split": split,
        "risk_target": (eval_result.get("risk_target") or {}).get("mode", "safe_neighbor"),
        "risk_target_safe_weight": (eval_result.get("risk_target") or {}).get("safe_weight", 1.0),
        "risk_target_retain_weight": (eval_result.get("risk_target") or {}).get("retain_weight", 0.0),
        "harmful_mean": harmful_mean,
        "safe_mean": safe_mean,
        "retain_mean": retain_mean,
        "harmful_minus_safe_mean": None if harmful_mean is None or safe_mean is None else float(harmful_mean - safe_mean),
        "harmful_minus_retain_mean": None if harmful_mean is None or retain_mean is None else float(harmful_mean - retain_mean),
        "harmful_vs_safe_auc": _metric(eval_result, ["auc", "harmful_trigger_vs_safe_neighbor"]),
        "harmful_vs_retain_auc": _metric(eval_result, ["auc", "harmful_trigger_vs_retain"]),
        "paired_mean_diff": paired.get("mean_diff"),
        "paired_median_diff": paired.get("median_diff"),
        "paired_ratio_harmful_greater_than_safe": paired.get("ratio_harmful_greater_than_safe"),
        "num_pairs": paired.get("num_pairs", 0),
    }
    if layer_set_name is not None:
        row["layer_set_name"] = layer_set_name
        row["layers"] = ",".join(str(layer) for layer in (layers or []))
        row["num_layers"] = len(layers or [])
    return row


def _read_eval_json(output_dir: Path, split: str, score_mode: str) -> Dict[str, Any]:
    path = output_dir / f"{split}_risk_space_eval_{score_mode}.json"
    if not path.exists():
        raise FileNotFoundError(f"Expected evaluation JSON not found: {path}")
    import json

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _recommended_from_rows(rows: List[Dict[str, Any]], min_retain_gap_constraint: bool = False) -> Dict[str, Any]:
    df = pd.DataFrame(rows)
    val_df = df[df["split"] == "val"].copy()
    if val_df.empty:
        raise ValueError("Cannot recommend configuration because no val rows are available.")
    if min_retain_gap_constraint:
        constrained = val_df[val_df["harmful_minus_retain_mean"].fillna(-1e9) >= 0]
        if not constrained.empty:
            val_df = constrained

    max_auc = val_df["harmful_vs_safe_auc"].max()
    candidates = val_df[val_df["harmful_vs_safe_auc"] >= max_auc - 0.02].copy()
    candidates["retain_penalty"] = (candidates["retain_mean"] - candidates["harmful_mean"]).fillna(1e9)
    candidates["centered_priority"] = candidates["score_mode"].map({"centered": 0, "raw": 1}).fillna(2)
    candidates = candidates.sort_values(
        by=["paired_mean_diff", "retain_penalty", "k", "centered_priority"],
        ascending=[False, True, True, True],
    )
    best = candidates.iloc[0].to_dict()
    train_match = df[
        (df["split"] == "train")
        & (df["k"] == best["k"])
        & (df["score_mode"] == best["score_mode"])
    ]
    reason = (
        f"Selected among validation configs within 0.02 AUC of the best AUC ({max_auc:.4f}); "
        "tie-breakers prefer larger paired mean diff, less retain inflation, smaller k, then centered score."
    )
    return {
        "recommended_k": int(best["k"]),
        "recommended_score_mode": str(best["score_mode"]),
        "reason": reason,
        "val_metrics": best,
        "train_metrics": train_match.iloc[0].to_dict() if not train_match.empty else None,
    }


def _recommend_layers(rows: List[Dict[str, Any]], recommended_k: int, score_mode_hint: str) -> Dict[str, Any]:
    df = pd.DataFrame(rows)
    val_df = df[df["split"] == "val"].copy()
    if val_df.empty:
        raise ValueError("Cannot recommend layers because no val layer-ablation rows are available.")

    # Prefer the recommended score mode if it is competitive; otherwise keep all modes.
    hinted = val_df[val_df["score_mode"] == score_mode_hint]
    if not hinted.empty and hinted["harmful_vs_safe_auc"].max() >= val_df["harmful_vs_safe_auc"].max() - 0.02:
        val_df = hinted

    max_auc = val_df["harmful_vs_safe_auc"].max()
    candidates = val_df[val_df["harmful_vs_safe_auc"] >= max_auc - 0.02].copy()
    positive = candidates[candidates["paired_mean_diff"].fillna(-1e9) > 0]
    if not positive.empty:
        candidates = positive
    candidates["retain_penalty"] = (candidates["retain_mean"] - candidates["harmful_mean"]).fillna(1e9)
    candidates["all_penalty"] = (candidates["layer_set_name"] == "all").astype(int)
    candidates["mode_priority"] = candidates["score_mode"].map({"centered": 0, "raw": 1}).fillna(2)
    candidates["compact_group_priority"] = candidates["layer_set_name"].map(
        {"late": 0, "mid_late": 0, "middle": 1, "early": 1}
    ).fillna(2)
    candidates = candidates.sort_values(
        by=[
            "paired_ratio_harmful_greater_than_safe",
            "retain_penalty",
            "num_layers",
            "compact_group_priority",
            "paired_mean_diff",
            "harmful_vs_safe_auc",
            "all_penalty",
            "mode_priority",
        ],
        ascending=[False, True, True, True, False, False, True, True],
    )
    best = candidates.iloc[0].to_dict()
    layers = [int(x) for x in str(best["layers"]).split(",") if x]
    train_match = df[
        (df["split"] == "train")
        & (df["layer_set_name"] == best["layer_set_name"])
        & (df["score_mode"] == best["score_mode"])
    ]
    reason = (
        "Selected from validation layer ablations by high harmful-vs-safe AUC, positive paired diff, "
        "high harmful-greater ratio, lower retain inflation, and fewer layers. If a compact late/mid-late "
        "set is close to all layers, the compact set is preferred for sparse second-stage editing."
    )
    return {
        "recommended_layers": layers,
        "recommended_score_mode": str(best["score_mode"]),
        "recommended_k": int(recommended_k),
        "reason": reason,
        "val_metrics": best,
        "train_metrics": train_match.iloc[0].to_dict() if not train_match.empty else None,
    }


def _write_summary(rows: List[Dict[str, Any]], csv_path: Path, json_path: Path) -> pd.DataFrame:
    ensure_dir(csv_path.parent)
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    save_json(rows, json_path)
    return df


def run_k_sweep(config: Dict[str, Any], k_values: Iterable[int], score_modes: Iterable[str]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    stage_metrics_dir = ensure_dir(resolve_path(config, config["outputs"]["metrics_dir"]) / "stage1_5")
    risk_root = ensure_dir(resolve_path(config, config["outputs"]["risk_space_dir"]))

    for k in k_values:
        k = int(k)
        logger.info("Stage 1.5 k sweep: building risk basis for k=%d", k)
        basis_dir = ensure_dir(risk_root / f"k_{k}")
        k_metrics_dir = ensure_dir(stage_metrics_dir / f"k_{k}")
        basis_path = build_risk_space(config, k_override=k, output_dir=basis_dir, metrics_dir=k_metrics_dir)

        for score_mode in score_modes:
            for split in ["train", "val"]:
                logger.info("Stage 1.5 k=%d score_mode=%s split=%s", k, score_mode, split)
                evaluate_risk_space(
                    config,
                    split=split,
                    risk_basis_path=str(basis_path),
                    score_mode=score_mode,
                    output_dir=k_metrics_dir,
                    generate_figures=False,
                )
                eval_result = _read_eval_json(k_metrics_dir, split, score_mode)
                rows.append(_summary_row(eval_result, k=k, score_mode=score_mode, split=split))

    summary_df = _write_summary(
        rows,
        stage_metrics_dir / "k_sweep_summary.csv",
        stage_metrics_dir / "k_sweep_summary.json",
    )
    figures_dir = ensure_dir(resolve_path(config, config["outputs"]["figures_dir"]) / "stage1_5")
    plot_k_sweep_val_auc(summary_df, figures_dir)
    plot_k_sweep_val_means(summary_df, figures_dir)

    recommended = _recommended_from_rows(
        rows,
        min_retain_gap_constraint=bool(config.get("stage1_5", {}).get("min_retain_gap_constraint", False)),
    )
    save_json(recommended, stage_metrics_dir / "recommended_config.json")
    logger.info("Saved Stage 1.5 recommended config to %s", stage_metrics_dir / "recommended_config.json")
    return rows, recommended


def _filter_layers(existing_layers: List[int], requested: List[int], name: str) -> List[int]:
    existing = set(existing_layers)
    actual = [layer for layer in requested if layer in existing]
    missing = [layer for layer in requested if layer not in existing]
    if missing:
        logger.warning("Layer set %s dropped unavailable layers: %s", name, missing)
    return actual


def run_layer_ablation(
    config: Dict[str, Any],
    recommended_k: int,
    score_modes: Iterable[str],
    recommended_score_mode: str = "centered",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    target_layers = _load_target_layers(config)
    stage_metrics_dir = ensure_dir(resolve_path(config, config["outputs"]["metrics_dir"]) / "stage1_5")
    ablation_dir = ensure_dir(stage_metrics_dir / "layer_ablation")
    details_dir = ensure_dir(ablation_dir / "details")
    figures_dir = ensure_dir(resolve_path(config, config["outputs"]["figures_dir"]) / "stage1_5")
    basis_path = resolve_path(config, config["outputs"]["risk_space_dir"]) / f"k_{recommended_k}" / "risk_basis.pt"
    if not basis_path.exists():
        logger.warning("Recommended-k basis missing at %s; rebuilding.", basis_path)
        build_risk_space(config, k_override=recommended_k, output_dir=basis_path.parent, metrics_dir=stage_metrics_dir / f"k_{recommended_k}")

    single_rows: List[Dict[str, Any]] = []
    group_rows: List[Dict[str, Any]] = []

    single_sets = [(f"layer_{layer}", [layer]) for layer in target_layers]
    group_sets = [
        ("early", [4, 8]),
        ("middle", [12, 16]),
        ("late", [20, 24]),
        ("mid_late", [12, 16, 20, 24]),
        ("all", [4, 8, 12, 16, 20, 24]),
    ]

    for layer_set_name, requested_layers in single_sets + group_sets:
        layers = _filter_layers(target_layers, requested_layers, layer_set_name)
        if not layers:
            logger.warning("Skipping empty layer set %s.", layer_set_name)
            continue
        out_dir = ensure_dir(details_dir / layer_set_name)
        is_single = layer_set_name.startswith("layer_")
        for score_mode in score_modes:
            for split in ["train", "val"]:
                evaluate_risk_space(
                    config,
                    split=split,
                    risk_basis_path=str(basis_path),
                    score_mode=score_mode,
                    target_layers_override=layers,
                    output_dir=out_dir,
                    generate_figures=False,
                )
                eval_result = _read_eval_json(out_dir, split, score_mode)
                row = _summary_row(
                    eval_result,
                    k=recommended_k,
                    score_mode=score_mode,
                    split=split,
                    layer_set_name=layer_set_name,
                    layers=layers,
                )
                if is_single:
                    single_rows.append(row)
                else:
                    group_rows.append(row)

    single_df = _write_summary(
        single_rows,
        ablation_dir / "single_layer_summary.csv",
        ablation_dir / "single_layer_summary.json",
    )
    group_df = _write_summary(
        group_rows,
        ablation_dir / "group_layer_summary.csv",
        ablation_dir / "group_layer_summary.json",
    )
    plot_single_layer_val_auc(single_df, figures_dir)
    plot_group_layer_val_auc(group_df, figures_dir)

    recommended = _recommend_layers(single_rows + group_rows, recommended_k, score_mode_hint=recommended_score_mode)
    save_json(recommended, stage_metrics_dir / "recommended_layers.json")
    logger.info("Saved Stage 1.5 recommended layers to %s", stage_metrics_dir / "recommended_layers.json")
    return single_rows + group_rows, recommended


def run_stage1_5_analysis(
    config: Dict[str, Any],
    *,
    skip_k_sweep: bool = False,
    skip_layer_ablation: bool = False,
    k_values: Optional[List[int]] = None,
    score_modes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    _ensure_hidden_states_exist(config)
    cfg = copy.deepcopy(config)
    stage_cfg = cfg.get("stage1_5", {})
    k_values = [int(k) for k in (k_values or stage_cfg.get("k_values", [1, 2, 4, 8]))]
    score_modes = [mode.lower() for mode in (score_modes or stage_cfg.get("score_modes", ["raw", "centered"]))]
    invalid_modes = sorted(set(score_modes) - VALID_SCORE_MODES)
    if invalid_modes:
        raise ValueError(f"Unsupported score_modes: {invalid_modes}")

    stage_metrics_dir = ensure_dir(resolve_path(cfg, cfg["outputs"]["metrics_dir"]) / "stage1_5")
    recommendation: Optional[Dict[str, Any]] = None
    if not skip_k_sweep:
        _, recommendation = run_k_sweep(cfg, k_values, score_modes)
    else:
        import json

        rec_path = stage_metrics_dir / "recommended_config.json"
        if not rec_path.exists():
            raise FileNotFoundError("--skip_k_sweep requires existing outputs/metrics/stage1_5/recommended_config.json")
        recommendation = json.load(rec_path.open("r", encoding="utf-8"))

    layer_recommendation = None
    if not skip_layer_ablation and bool(stage_cfg.get("layer_ablation", {}).get("enabled", True)):
        use_k = stage_cfg.get("layer_ablation", {}).get("use_k")
        ablation_k = int(use_k if use_k is not None else recommendation["recommended_k"])
        layer_recommendation = run_layer_ablation(
            cfg,
            ablation_k,
            score_modes,
            recommended_score_mode=str(recommendation["recommended_score_mode"]),
        )[1]

    result = {
        "recommended_config": recommendation,
        "recommended_layers": layer_recommendation,
    }
    save_json(result, stage_metrics_dir / "stage1_5_final_summary.json")
    return result
