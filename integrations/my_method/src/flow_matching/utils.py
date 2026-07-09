from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, Optional

import torch

from ..risk_space.recommended_config import RecommendedRiskConfig
from ..utils import resolve_path


def response_mean_last_pool(
    hidden: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    mean_weight: float = 0.5,
    last_weight: float = 0.5,
    debug: bool = False,
) -> torch.Tensor:
    if hidden.ndim != 3 or response_mask.ndim != 2:
        raise ValueError("Expected hidden [B,T,D] and response_mask [B,T].")
    mask = response_mask.to(hidden.device).bool()
    outs = []
    for i in range(hidden.shape[0]):
        idx = mask[i].nonzero(as_tuple=False).flatten()
        if idx.numel() == 0:
            msg = f"Empty response span for batch index {i}; falling back to last non-pad token."
            if debug:
                raise ValueError(msg)
            warnings.warn(msg)
            idx = torch.tensor([hidden.shape[1] - 1], device=hidden.device)
        selected = hidden[i, idx, :]
        outs.append(float(mean_weight) * selected.mean(dim=0) + float(last_weight) * selected[-1])
    return torch.stack(outs, dim=0)


def compute_risk_coefficients(
    x: torch.Tensor,
    layer: int,
    recommended: RecommendedRiskConfig,
    risk_basis: Dict[int, torch.Tensor],
    safe_center: Optional[Dict[int, torch.Tensor]] = None,
) -> torch.Tensor:
    layer = int(layer)
    if layer not in risk_basis:
        raise KeyError(f"Risk basis missing layer {layer}.")
    basis = risk_basis[layer].to(device=x.device, dtype=x.dtype)
    if basis.ndim != 2:
        raise ValueError(f"Risk basis for layer {layer} must be [k,D], got {tuple(basis.shape)}")
    k = min(int(recommended.recommended_k), basis.shape[0])
    x_used = x
    score_mode = str(recommended.recommended_score_mode)
    if score_mode == "centered" or score_mode.startswith("centered_") or score_mode.startswith("paired_delta"):
        if safe_center is None or layer not in safe_center:
            raise KeyError(f"safe_center missing for centered score at layer {layer}.")
        x_used = x - safe_center[layer].to(device=x.device, dtype=x.dtype)
    elif score_mode == "raw" or score_mode.startswith("raw_"):
        pass
    else:
        raise ValueError(f"Unsupported score mode: {recommended.recommended_score_mode}")
    return x_used @ basis[:k].T


def compute_risk_delta_coefficients(
    delta: torch.Tensor,
    layer: int,
    recommended: RecommendedRiskConfig,
    risk_basis: Dict[int, torch.Tensor],
) -> torch.Tensor:
    """Project a counterfactual hidden-state delta onto the oriented risk basis.

    Unlike ``compute_risk_coefficients``, this function does not subtract a
    center. The caller has already removed context through a paired/reference
    hidden-state difference.
    """
    layer = int(layer)
    if layer not in risk_basis:
        raise KeyError(f"Risk basis missing layer {layer}.")
    basis = risk_basis[layer].to(device=delta.device, dtype=delta.dtype)
    if basis.ndim != 2:
        raise ValueError(f"Risk basis for layer {layer} must be [k,D], got {tuple(basis.shape)}")
    k = min(int(recommended.recommended_k), basis.shape[0])
    return delta @ basis[:k].T


def load_stage2_implicit_normalization(config: dict) -> tuple[float, float, bool]:
    metrics_dir = config.get("stage2", {}).get("outputs", {}).get(
        "metrics_dir", "integrations/my_method/outputs/metrics/stage2"
    )
    path = resolve_path(config, str(Path(metrics_dir) / "implicit_normalization.json"))
    if not path.exists():
        raise FileNotFoundError(f"Missing Stage 2 implicit normalization: {path}. Run Stage 2 first.")
    data = json.load(path.open("r", encoding="utf-8"))
    clip = bool(data.get("clip", True) or config.get("stage2", {}).get("normalization", {}).get("clip", True))
    return float(data["lower_value"]), float(data["upper_value"]), clip


def load_dynamic_implicit_normalization(config: dict) -> tuple[Dict[int, float], Dict[int, float], bool]:
    """Load Stage2 implicit-risk normalization for dynamic Flow conditioning.

    The dynamic hidden state is still observed one layer at a time, but its raw
    score and normalization now use the same risk-space definition and global
    train-set bounds as Stage2 ``R_implicit_norm``.
    """
    metrics_dir = config.get("stage2", {}).get("outputs", {}).get(
        "metrics_dir", "integrations/my_method/outputs/metrics/stage2"
    )
    path = resolve_path(config, str(Path(metrics_dir) / "implicit_normalization.json"))
    if not path.exists():
        raise FileNotFoundError(f"Missing Stage2 implicit normalization: {path}. Run Stage 2 first.")
    data = json.load(path.open("r", encoding="utf-8"))
    settings = data.get("implicit_settings") or {}
    layers = [int(x) for x in settings.get("layers", [])]
    if not layers:
        layers = [int(x) for x in config.get("stage3", {}).get("risk_space", {}).get("risk_layers", [])]
    if not layers:
        raise ValueError(f"Stage2 implicit normalization does not record risk layers: {path}")
    lower_value = float(data["lower_value"])
    upper_value = float(data["upper_value"])
    if upper_value <= lower_value:
        raise ValueError(
            f"Invalid Stage2 implicit-risk bounds: lower={lower_value}, upper={upper_value}"
        )
    lower = {int(layer): lower_value for layer in layers}
    upper = {int(layer): upper_value for layer in layers}
    clip = bool(data.get("clip", True) or config.get("stage2", {}).get("normalization", {}).get("clip", True))
    return lower, upper, clip


def normalize_implicit_risk_value(raw: torch.Tensor, lower: float, upper: float, *, clip: bool = True) -> torch.Tensor:
    norm = (raw - float(lower)) / max(float(upper) - float(lower), 1e-6)
    if clip:
        norm = torch.clamp(norm, 0.0, 1.0)
    return norm


def risk_score_from_coefficients(coeff: torch.Tensor, score_mode: str) -> torch.Tensor:
    if score_mode.endswith("_signed"):
        return coeff.mean(dim=-1, keepdim=True)
    if score_mode.endswith("_positive"):
        return torch.linalg.vector_norm(torch.relu(coeff), ord=2, dim=-1, keepdim=True)
    return torch.linalg.vector_norm(coeff, ord=2, dim=-1, keepdim=True)


def dynamic_implicit_risk_norm(
    x: torch.Tensor,
    layer_id: torch.Tensor,
    recommended: RecommendedRiskConfig,
    risk_basis: Dict[int, torch.Tensor],
    safe_center: Optional[Dict[int, torch.Tensor]],
    lower: Dict[int, float],
    upper: Dict[int, float],
    *,
    clip: bool = True,
) -> torch.Tensor:
    """Return per-layer calibrated implicit risk for current flow states.

    ``x`` is the current hidden state on the flow path, so this computes the
    continuous R_imp(t) used to condition the velocity field.
    """
    if x.ndim != 2:
        raise ValueError(f"dynamic_implicit_risk_norm expects x [B,D], got {tuple(x.shape)}")
    layer_id = layer_id.to(device=x.device, dtype=torch.long).flatten()
    if layer_id.numel() == 1 and x.shape[0] != 1:
        layer_id = layer_id.expand(x.shape[0])
    if layer_id.numel() != x.shape[0]:
        raise ValueError(f"layer_id must have B entries, got B={x.shape[0]} layer_id={tuple(layer_id.shape)}")

    out = torch.empty((x.shape[0], 1), device=x.device, dtype=x.dtype)
    for layer in torch.unique(layer_id).tolist():
        layer = int(layer)
        if layer not in lower or layer not in upper:
            raise KeyError(
                f"Dynamic implicit-risk normalization missing layer {layer}; "
                f"available layers={sorted(lower)}"
        )
        mask = layer_id == int(layer)
        coeff = compute_risk_coefficients(x[mask], layer, recommended, risk_basis, safe_center)
        raw = risk_score_from_coefficients(coeff, str(recommended.recommended_score_mode))
        out[mask] = normalize_implicit_risk_value(
            raw,
            lower[layer],
            upper[layer],
            clip=clip,
        ).to(dtype=x.dtype)
    return out


def lambda_flow_ramp(
    step: int,
    total_steps: int,
    max_value: float,
    start_ratio: float,
    end_ratio: float,
    min_warmup_steps: int = 0,
) -> float:
    total_steps = max(1, int(total_steps))
    start = max(float(start_ratio) * total_steps, float(max(0, int(min_warmup_steps))))
    end = max(float(end_ratio) * total_steps, start + 1.0)
    if step < start:
        return 0.0
    if step >= end:
        return float(max_value)
    return float(max_value) * float((step - start) / max(end - start, 1e-6))
