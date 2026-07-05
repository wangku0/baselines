from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from .utils import ensure_dir, ensure_output_dirs, logger, resolve_path, save_json


def _ensure_torch_runtime() -> None:
    if not hasattr(torch, "load") or not hasattr(torch, "linalg"):
        raise ImportError("PyTorch is not fully installed. Install dependencies with: pip install -r requirements.txt")


def _layer_tensor(hidden_states: Dict[Any, torch.Tensor], layer: int) -> torch.Tensor:
    if layer in hidden_states:
        return hidden_states[layer]
    if str(layer) in hidden_states:
        return hidden_states[str(layer)]
    raise KeyError(f"Layer {layer} not found in hidden_states.")


def _paired_indices(metadata: List[Dict[str, Any]]) -> Tuple[List[str], List[int], List[int]]:
    harmful: Dict[str, int] = {}
    safe: Dict[str, int] = {}
    for idx, meta in enumerate(metadata):
        pair_id = meta.get("pair_id")
        sample_type = meta.get("sample_type")
        if not pair_id:
            continue
        if sample_type == "harmful_trigger" and pair_id not in harmful:
            harmful[pair_id] = idx
        elif sample_type == "safe_neighbor" and pair_id not in safe:
            safe[pair_id] = idx

    pair_ids = sorted(set(harmful) & set(safe))
    missing_harmful = sorted(set(safe) - set(harmful))
    missing_safe = sorted(set(harmful) - set(safe))
    if missing_harmful:
        logger.warning("Pairs missing harmful samples: %s", missing_harmful[:10])
    if missing_safe:
        logger.warning("Pairs missing safe_neighbor samples: %s", missing_safe[:10])
    return pair_ids, [harmful[pair_id] for pair_id in pair_ids], [safe[pair_id] for pair_id in pair_ids]


def _sample_type_indices(metadata: List[Dict[str, Any]], sample_type: str) -> List[int]:
    indices = [idx for idx, meta in enumerate(metadata) if meta.get("sample_type") == sample_type]
    if not indices:
        raise ValueError(f"No train {sample_type} samples found; cannot compute risk-space center.")
    return indices


def _risk_target_config(config: Dict[str, Any]) -> Dict[str, float | str]:
    target_cfg = config.get("risk_space", {}).get("target", {})
    mode_raw = str(target_cfg.get("mode", "safe_neighbor")).lower()
    aliases = {
        "safe": "safe_neighbor",
        "safenb": "safe_neighbor",
        "safe_nb": "safe_neighbor",
        "safe_neighbor": "safe_neighbor",
        "retain": "retain",
        "mixed": "mixed",
        "mix": "mixed",
        "alpha_safenb_beta_retain": "mixed",
        "safe_retain_mix": "mixed",
    }
    mode = aliases.get(mode_raw)
    if mode is None:
        raise ValueError(
            f"Unsupported risk_space.target.mode={target_cfg.get('mode')!r}. "
            "Use safe_neighbor, retain, or mixed."
        )

    safe_weight = float(target_cfg.get("safe_weight", target_cfg.get("alpha_safe", 0.5)))
    retain_weight = float(target_cfg.get("retain_weight", target_cfg.get("beta_retain", 0.5)))
    if mode == "safe_neighbor":
        safe_weight, retain_weight = 1.0, 0.0
    elif mode == "retain":
        safe_weight, retain_weight = 0.0, 1.0
    else:
        denom = safe_weight + retain_weight
        if denom <= 0:
            raise ValueError("mixed risk target requires safe_weight + retain_weight > 0.")
        safe_weight, retain_weight = safe_weight / denom, retain_weight / denom
    return {"mode": mode, "safe_weight": safe_weight, "retain_weight": retain_weight}


def _choose_retain_indices(metadata: List[Dict[str, Any]], harmful_idx: List[int]) -> List[int]:
    retain_indices = _sample_type_indices(metadata, "retain")
    retain_by_sample_index: Dict[Any, List[int]] = defaultdict(list)
    for idx in retain_indices:
        retain_by_sample_index[metadata[idx].get("sample_index")].append(idx)

    chosen: List[int] = []
    for pos, h_idx in enumerate(harmful_idx):
        same_record = retain_by_sample_index.get(metadata[h_idx].get("sample_index")) or []
        chosen.append(same_record[pos % len(same_record)] if same_record else retain_indices[pos % len(retain_indices)])
    return chosen


def build_risk_space(
    config: Dict[str, Any],
    k_override: Optional[int] = None,
    output_dir: Optional[str | Path] = None,
    metrics_dir: Optional[str | Path] = None,
):
    _ensure_torch_runtime()
    ensure_output_dirs(config)
    risk_cfg = config.get("risk_space", {})
    split = risk_cfg.get("build_split", "train")
    if split != "train":
        logger.warning(
            "risk-space center is intended to be computed from train target samples, but build_split=%s.",
            split,
        )
    hidden_path = resolve_path(config, config["outputs"]["hidden_states_dir"]) / f"{split}_hidden_states.pt"
    if not hidden_path.exists():
        raise FileNotFoundError(
            f"Hidden state file not found: {hidden_path}. Run scripts/01_extract_hidden_states.py first."
        )

    data = torch.load(hidden_path, map_location="cpu", weights_only=False)
    metadata = data["metadata"]
    hidden_states = data["hidden_states"]
    target_layers = [int(layer) for layer in data["target_layers"]]

    pair_ids, harmful_idx, safe_idx = _paired_indices(metadata)
    if not pair_ids:
        raise ValueError("No paired harmful_trigger/safe_neighbor samples found in hidden-state metadata.")
    target_cfg = _risk_target_config(config)
    target_mode = str(target_cfg["mode"])
    retain_idx = _choose_retain_indices(metadata, harmful_idx) if target_mode in {"retain", "mixed"} else []
    center_idx = (
        _sample_type_indices(metadata, "retain")
        if target_mode == "retain"
        else _sample_type_indices(metadata, "safe_neighbor")
        if target_mode == "safe_neighbor"
        else []
    )

    requested_k = int(k_override if k_override is not None else risk_cfg.get("k", 8))
    center_delta = bool(risk_cfg.get("center_delta", True))
    normalize_delta = bool(risk_cfg.get("normalize_delta", False))
    logger.info(
        "Building risk space from %d paired samples, requested k=%d, target=%s (safe_weight=%.3f, retain_weight=%.3f)",
        len(pair_ids),
        requested_k,
        target_mode,
        float(target_cfg["safe_weight"]),
        float(target_cfg["retain_weight"]),
    )

    risk_basis: Dict[int, torch.Tensor] = {}
    safe_center: Dict[int, torch.Tensor] = {}
    mean_delta: Dict[int, torch.Tensor] = {}
    basis_signs: Dict[int, torch.Tensor] = {}
    singular_values: Dict[int, torch.Tensor] = {}
    actual_k_by_layer: Dict[str, int] = {}
    svd_stats: Dict[str, Any] = {
        "requested_k": requested_k,
        "num_pairs": len(pair_ids),
        "center_delta": center_delta,
        "normalize_delta": normalize_delta,
        "risk_target": target_cfg,
        "layers": {},
    }

    for layer in target_layers:
        h = _layer_tensor(hidden_states, layer).to(dtype=torch.float32)
        harmful_h = h[harmful_idx]
        safe_h = h[safe_idx]
        if target_mode == "safe_neighbor":
            target_h = safe_h
            safe_center[layer] = h[center_idx].mean(dim=0).to("cpu", dtype=torch.float32)
        elif target_mode == "retain":
            retain_h = h[retain_idx]
            target_h = retain_h
            safe_center[layer] = h[center_idx].mean(dim=0).to("cpu", dtype=torch.float32)
        else:
            retain_h = h[retain_idx]
            target_h = float(target_cfg["safe_weight"]) * safe_h + float(target_cfg["retain_weight"]) * retain_h
            safe_center[layer] = target_h.mean(dim=0).to("cpu", dtype=torch.float32)
        raw_delta = harmful_h - target_h
        mean_delta[layer] = raw_delta.mean(dim=0).to("cpu", dtype=torch.float32)
        delta = raw_delta
        if center_delta:
            delta = delta - delta.mean(dim=0, keepdim=True)
        if normalize_delta:
            delta = delta / delta.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)

        num_pairs, hidden_dim = delta.shape
        actual_k = min(requested_k, num_pairs, hidden_dim)
        if actual_k < requested_k:
            logger.warning(
                "Layer %s has only num_pairs=%d, hidden_dim=%d; using actual_k=%d instead of requested_k=%d.",
                layer,
                num_pairs,
                hidden_dim,
                actual_k,
                requested_k,
            )

        try:
            _, s, vh = torch.linalg.svd(delta, full_matrices=False)
        except RuntimeError as exc:
            raise RuntimeError(f"SVD failed at layer {layer}: {exc}") from exc

        basis = vh[:actual_k].contiguous().to(dtype=torch.float32)
        # SVD basis vectors are sign-indeterminate. Orient every vector so its
        # projection on the empirical harmful -> target contrast is positive.
        orientation = basis @ mean_delta[layer].to(device=basis.device, dtype=basis.dtype)
        signs = torch.where(orientation < 0, -torch.ones_like(orientation), torch.ones_like(orientation))
        basis = (basis * signs[:, None]).to("cpu", dtype=torch.float32)
        basis_signs[layer] = signs.to("cpu", dtype=torch.float32)
        risk_basis[layer] = basis
        singular_values[layer] = s.to("cpu", dtype=torch.float32)
        actual_k_by_layer[str(layer)] = actual_k

        energy = s.pow(2)
        total_energy = float(energy.sum().item())
        explained = energy / energy.sum().clamp_min(1e-12)
        svd_stats["layers"][str(layer)] = {
            "num_pairs": num_pairs,
            "hidden_dim": hidden_dim,
            "requested_k": requested_k,
            "actual_k": actual_k,
            "top_k_singular_values": [float(x) for x in s[:actual_k].tolist()],
            "top_k_explained_ratio": [float(x) for x in explained[:actual_k].tolist()],
            "cumulative_explained_ratio": float(explained[:actual_k].sum().item()) if total_energy > 0 else 0.0,
        }
        logger.info(
            "Layer %s: hidden_dim=%d, actual_k=%d, cumulative explained ratio=%.4f",
            layer,
            hidden_dim,
            actual_k,
            svd_stats["layers"][str(layer)]["cumulative_explained_ratio"],
        )

    output = {
        "risk_basis": risk_basis,
        "safe_center": safe_center,
        "mean_delta": mean_delta,
        "basis_signs": basis_signs,
        "singular_values": singular_values,
        "target_layers": target_layers,
        "k": requested_k,
        "actual_k_by_layer": actual_k_by_layer,
        "num_pairs": len(pair_ids),
        "pair_ids": pair_ids,
        "target_indices": {
            "harmful": harmful_idx,
            "safe_neighbor": safe_idx,
            "retain": retain_idx,
        },
        "risk_target": target_cfg,
        "center_delta": center_delta,
        "normalize_delta": normalize_delta,
        "model_path": data.get("model_path"),
    }

    risk_dir = ensure_dir(Path(output_dir) if output_dir is not None else resolve_path(config, config["outputs"]["risk_space_dir"]))
    basis_path = risk_dir / "risk_basis.pt"
    torch.save(output, basis_path)
    logger.info("Saved risk basis to %s", basis_path)

    out_metrics_dir = ensure_dir(Path(metrics_dir) if metrics_dir is not None else resolve_path(config, config["outputs"]["metrics_dir"]))
    stats_path = out_metrics_dir / "svd_stats.json"
    save_json(svd_stats, stats_path)
    logger.info("Saved SVD stats to %s", stats_path)
    return basis_path
