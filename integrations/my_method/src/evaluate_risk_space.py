from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from .build_risk_space import _layer_tensor
from .utils import ensure_dir, ensure_output_dirs, logger, resolve_path, save_json
from .visualize import generate_all_figures


def _ensure_torch_runtime() -> None:
    if not hasattr(torch, "load") or not hasattr(torch, "linalg"):
        raise ImportError("PyTorch is not fully installed. Install dependencies with: pip install -r requirements.txt")


def _summary(values: pd.Series) -> Dict[str, Optional[float]]:
    values = values.dropna()
    if values.empty:
        return {"count": 0, "mean": None, "std": None, "median": None, "min": None, "max": None}
    return {
        "count": int(values.shape[0]),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.shape[0] > 1 else 0.0,
        "median": float(values.median()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _auc(df: pd.DataFrame, positive_type: str, negative_type: str) -> Optional[float]:
    subset = df[df["sample_type"].isin([positive_type, negative_type])].copy()
    counts = subset["sample_type"].value_counts().to_dict()
    if counts.get(positive_type, 0) < 1 or counts.get(negative_type, 0) < 1:
        logger.warning("Skipping AUC %s vs %s due to insufficient samples: %s", positive_type, negative_type, counts)
        return None
    labels = (subset["sample_type"] == positive_type).astype(int).values
    if len(np.unique(labels)) < 2:
        return None
    return float(roc_auc_score(labels, subset["R_total"].values))


def _paired_analysis(df: pd.DataFrame, compute_tests: bool = True) -> Dict[str, Any]:
    harmful = df[df["sample_type"] == "harmful_trigger"].dropna(subset=["pair_id"])
    safe = df[df["sample_type"] == "safe_neighbor"].dropna(subset=["pair_id"])
    merged = harmful[["pair_id", "R_total"]].merge(
        safe[["pair_id", "R_total"]],
        on="pair_id",
        suffixes=("_harmful", "_safe"),
    )
    if merged.empty:
        logger.warning("No paired harmful/safe samples for paired analysis.")
        return {"num_pairs": 0}

    diffs = merged["R_total_harmful"] - merged["R_total_safe"]
    result: Dict[str, Any] = {
        "num_pairs": int(len(diffs)),
        "mean_diff": float(diffs.mean()),
        "median_diff": float(diffs.median()),
        "std_diff": float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.0,
        "ratio_harmful_greater_than_safe": float((diffs > 0).mean()),
    }
    if compute_tests and len(diffs) > 1:
        try:
            from scipy import stats

            t_stat, t_p = stats.ttest_rel(merged["R_total_harmful"], merged["R_total_safe"])
            result["paired_t_test"] = {"statistic": float(t_stat), "pvalue": float(t_p)}
            try:
                w_stat, w_p = stats.wilcoxon(diffs)
                result["wilcoxon_signed_rank"] = {"statistic": float(w_stat), "pvalue": float(w_p)}
            except ValueError as exc:
                result["wilcoxon_signed_rank"] = {"skipped": str(exc)}
        except ImportError:
            logger.warning("scipy is not available; skipping paired t-test and Wilcoxon test.")
            result["stat_tests"] = "skipped: scipy not available"
    return result


def _category_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for category, group in df.groupby("category", dropna=False):
        category_key = str(category)
        counts = group["sample_type"].value_counts().to_dict()
        entry: Dict[str, Any] = {
            "counts": {str(k): int(v) for k, v in counts.items()},
            "mean_R_total_by_type": {
                sample_type: float(values)
                for sample_type, values in group.groupby("sample_type")["R_total"].mean().dropna().items()
            },
        }
        if counts.get("harmful_trigger", 0) >= 2 and counts.get("safe_neighbor", 0) >= 2:
            entry["auc_harmful_vs_safe_neighbor"] = _auc(group, "harmful_trigger", "safe_neighbor")
        else:
            entry["auc_harmful_vs_safe_neighbor"] = None
        if counts.get("harmful_trigger", 0) >= 2 and counts.get("retain", 0) >= 2:
            entry["auc_harmful_vs_retain"] = _auc(group, "harmful_trigger", "retain")
        else:
            entry["auc_harmful_vs_retain"] = None
        result[category_key] = entry
    return result


def _output_stem(split: str, kind: str, score_mode: str, output_suffix: Optional[str]) -> str:
    parts = [split, kind]
    if output_suffix:
        parts.append(output_suffix)
    parts.append(score_mode)
    return "_".join(parts)


def _valid_layers(requested_layers: List[int], hidden_states: Dict[Any, torch.Tensor], risk_basis: Dict[Any, torch.Tensor]) -> List[int]:
    valid = []
    for layer in requested_layers:
        if (layer in hidden_states or str(layer) in hidden_states) and (layer in risk_basis or str(layer) in risk_basis):
            valid.append(layer)
        else:
            logger.warning("Layer %s missing from hidden states or risk basis; skipping.", layer)
    if not valid:
        raise ValueError(f"No valid target layers after filtering requested layers: {requested_layers}")
    return valid


def _paired_delta_baselines(metadata: List[Dict[str, Any]]) -> Dict[int, Optional[int]]:
    harmful_by_pair: Dict[str, int] = {}
    safe_by_pair: Dict[str, int] = {}
    safe_by_record: Dict[Any, List[int]] = {}
    for idx, meta in enumerate(metadata):
        pair_id = meta.get("pair_id")
        sample_type = meta.get("sample_type")
        if sample_type == "harmful_trigger" and pair_id:
            harmful_by_pair[str(pair_id)] = idx
        elif sample_type == "safe_neighbor" and pair_id:
            safe_by_pair[str(pair_id)] = idx
            safe_by_record.setdefault(meta.get("sample_index"), []).append(idx)

    baselines: Dict[int, Optional[int]] = {}
    for idx, meta in enumerate(metadata):
        sample_type = meta.get("sample_type")
        pair_id = meta.get("pair_id")
        if sample_type == "harmful_trigger" and pair_id and str(pair_id) in safe_by_pair:
            baselines[idx] = safe_by_pair[str(pair_id)]
        elif sample_type == "safe_neighbor" and pair_id and str(pair_id) in harmful_by_pair:
            baselines[idx] = harmful_by_pair[str(pair_id)]
        elif sample_type == "retain":
            same_record_safe = safe_by_record.get(meta.get("sample_index")) or []
            baselines[idx] = same_record_safe[0] if same_record_safe else None
        else:
            baselines[idx] = None
    return baselines


def _score_projection(proj: torch.Tensor, score_mode: str) -> float:
    if score_mode.endswith("_positive"):
        return float(torch.linalg.vector_norm(torch.relu(proj), ord=2).item())
    if score_mode.endswith("_signed"):
        return float(proj.mean().item())
    return float(torch.linalg.vector_norm(proj, ord=2).item())


def evaluate_risk_space(
    config: Dict[str, Any],
    split: str,
    risk_basis_path: Optional[str] = None,
    score_mode: str = "raw",
    target_layers_override: Optional[List[int]] = None,
    output_suffix: Optional[str] = None,
    output_dir: Optional[str | Path] = None,
    figures_dir: Optional[str | Path] = None,
    generate_figures: bool = True,
) -> pd.DataFrame:
    _ensure_torch_runtime()
    ensure_output_dirs(config)
    score_mode = score_mode.lower()
    valid_score_modes = {
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
    if score_mode not in valid_score_modes:
        raise ValueError(f"score_mode must be one of {sorted(valid_score_modes)}, got {score_mode}")

    hidden_path = resolve_path(config, config["outputs"]["hidden_states_dir"]) / f"{split}_hidden_states.pt"
    basis_path = Path(risk_basis_path).expanduser().resolve() if risk_basis_path else resolve_path(config, config["outputs"]["risk_space_dir"]) / "risk_basis.pt"
    if not hidden_path.exists():
        raise FileNotFoundError(
            f"Hidden state file not found: {hidden_path}. Run scripts/01_extract_hidden_states.py first."
        )
    if not basis_path.exists():
        raise FileNotFoundError(f"Risk basis file not found: {basis_path}")

    hidden_data = torch.load(hidden_path, map_location="cpu", weights_only=False)
    basis_data = torch.load(basis_path, map_location="cpu", weights_only=False)
    metadata = hidden_data["metadata"]
    hidden_states = hidden_data["hidden_states"]
    risk_basis = basis_data["risk_basis"]
    safe_center = basis_data.get("safe_center", {})
    base_layers = [int(layer) for layer in basis_data["target_layers"]]
    requested_layers = [int(layer) for layer in (target_layers_override or base_layers)]
    target_layers = _valid_layers(requested_layers, hidden_states, risk_basis)
    if target_layers_override is not None:
        logger.info("Using target layer override: requested=%s, actual=%s", target_layers_override, target_layers)
    center_required = score_mode.startswith("centered")
    paired_delta_mode = score_mode.startswith("paired_delta")
    if center_required and not safe_center:
        raise ValueError(
            f"score_mode={score_mode!r} requires safe_center in {basis_path}. "
            "Rebuild risk basis with scripts/02_build_risk_space.py or scripts/04_stage1_5_analysis.py."
        )
    paired_baselines = _paired_delta_baselines(metadata) if paired_delta_mode else {}

    rows: List[Dict[str, Any]] = []
    layer_cols = [f"R_layer_{layer}" for layer in target_layers]
    for idx, meta in enumerate(metadata):
        row = {
            "sample_id": meta.get("id"),
            "split": meta.get("split", split),
            "sample_type": meta.get("sample_type"),
            "pair_id": meta.get("pair_id"),
            "category": meta.get("category"),
            "keyword": meta.get("keyword"),
            "image_path": meta.get("image_path"),
            "instruction": meta.get("instruction"),
            "score_mode": score_mode,
        }
        total = 0.0
        for layer in target_layers:
            h = _layer_tensor(hidden_states, layer)[idx].to(dtype=torch.float32)
            basis = _layer_tensor(risk_basis, layer).to(dtype=torch.float32)
            if paired_delta_mode:
                baseline_idx = paired_baselines.get(idx)
                if baseline_idx is None:
                    row[f"R_layer_{layer}"] = float("nan")
                    continue
                h_base = _layer_tensor(hidden_states, layer)[baseline_idx].to(dtype=torch.float32)
                h_eval = h - h_base
            elif center_required:
                center = _layer_tensor(safe_center, layer).to(dtype=torch.float32)
                h_eval = h - center
            else:
                h_eval = h
            proj = h_eval @ basis.T
            score = _score_projection(proj, score_mode)
            row[f"R_layer_{layer}"] = score
            if not pd.isna(score):
                total += score
        row["R_total"] = total if any(f"R_layer_{layer}" in row and not pd.isna(row[f"R_layer_{layer}"]) for layer in target_layers) else float("nan")
        rows.append(row)

    df = pd.DataFrame(rows)
    metrics_dir = ensure_dir(Path(output_dir) if output_dir is not None else resolve_path(config, config["outputs"]["metrics_dir"]))
    csv_path = metrics_dir / f"{_output_stem(split, 'risk_scores', score_mode, output_suffix)}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info("Saved risk scores to %s", csv_path)
    if output_dir is None and output_suffix is None and score_mode == "raw":
        legacy_csv_path = metrics_dir / f"{split}_risk_scores.csv"
        df.to_csv(legacy_csv_path, index=False, encoding="utf-8-sig")
        logger.info("Saved legacy raw risk scores to %s", legacy_csv_path)

    eval_cfg = config.get("evaluation", {})
    stats_by_type = {
        sample_type: _summary(df.loc[df["sample_type"] == sample_type, "R_total"])
        for sample_type in ["harmful_trigger", "safe_neighbor", "retain"]
    }
    eval_result: Dict[str, Any] = {
        "split": split,
        "score_mode": score_mode,
        "target_layers": target_layers,
        "risk_basis_path": str(basis_path),
        "risk_target": basis_data.get("risk_target", {"mode": "safe_neighbor", "safe_weight": 1.0, "retain_weight": 0.0}),
        "num_samples": int(len(df)),
        "stats_by_sample_type": stats_by_type,
        "auc": {},
        "paired_analysis": {},
        "category_analysis": {},
    }
    if bool(eval_cfg.get("compute_auc", True)):
        eval_result["auc"]["harmful_trigger_vs_safe_neighbor"] = _auc(df, "harmful_trigger", "safe_neighbor")
        eval_result["auc"]["harmful_trigger_vs_retain"] = _auc(df, "harmful_trigger", "retain")
    if bool(eval_cfg.get("compute_paired_stats", True)):
        eval_result["paired_analysis"] = _paired_analysis(df, compute_tests=True)
    eval_result["category_analysis"] = _category_analysis(df)

    json_path = metrics_dir / f"{_output_stem(split, 'risk_space_eval', score_mode, output_suffix)}.json"
    save_json(eval_result, json_path)
    logger.info("Saved evaluation metrics to %s", json_path)
    if output_dir is None and output_suffix is None and score_mode == "raw":
        legacy_json_path = metrics_dir / f"{split}_risk_space_eval.json"
        save_json(eval_result, legacy_json_path)
        logger.info("Saved legacy raw evaluation metrics to %s", legacy_json_path)

    if generate_figures:
        out_figures_dir = ensure_dir(Path(figures_dir) if figures_dir is not None else resolve_path(config, config["outputs"]["figures_dir"]))
        fig_split = f"{split}_{score_mode}" if output_suffix is None else f"{split}_{output_suffix}_{score_mode}"
        generate_all_figures(df, fig_split, layer_cols, out_figures_dir)
        logger.info("Saved figures to %s", out_figures_dir)
    return df
