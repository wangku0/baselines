from __future__ import annotations

import warnings
from typing import Dict, Optional

import torch

from ..risk_space.recommended_config import RecommendedRiskConfig


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
