from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch

from src.flow_matching.model import FlowVectorField
from src.flow_matching.utils import (
    compute_risk_coefficients,
    dynamic_implicit_risk_norm,
    load_dynamic_implicit_normalization,
)
from src.model_utils import infer_input_device
from src.risk_space.recommended_config import RecommendedRiskConfig, load_recommended_risk_config
from src.stage3_lora_utils import sync_stage3_layers_with_recommendation
from src.stage3_losses import load_risk_tensors
from src.utils import load_config, resolve_path


@dataclass
class FlowInterventionStats:
    calls: int = 0
    active_calls: int = 0
    mean_gate_sum: float = 0.0
    mean_risk_sum: float = 0.0
    mean_explicit_risk_sum: float = 0.0
    mean_total_risk_sum: float = 0.0
    mean_delta_norm_sum: float = 0.0
    risk_gate_mode: str = "fused"
    trace_dropped: int = 0
    numerical_skips: int = 0
    numerical_retries: int = 0

    def to_dict(self) -> dict:
        denom = max(self.calls, 1)
        return {
            "calls": self.calls,
            "active_calls": self.active_calls,
            "mean_gate": self.mean_gate_sum / denom,
            "mean_implicit_risk": self.mean_risk_sum / denom,
            "mean_explicit_risk": self.mean_explicit_risk_sum / denom,
            "mean_total_risk": self.mean_total_risk_sum / denom,
            "mean_delta_norm": self.mean_delta_norm_sum / denom,
            "risk_gate_mode": self.risk_gate_mode,
            "trace_dropped": self.trace_dropped,
            "numerical_skips": self.numerical_skips,
            "numerical_retries": self.numerical_retries,
        }


def _load_flow_teacher(path: Path, device: torch.device) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Flow teacher not found: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    teacher_cfg = ckpt.get("teacher_cfg", {})
    model = FlowVectorField(
        hidden_dim=int(ckpt["hidden_dim"]),
        cond_dim=int(ckpt["cond_dim"]),
        hidden_width=int(teacher_cfg.get("hidden_width", 1024)),
        time_embedding_dim=int(teacher_cfg.get("time_embedding_dim", 128)),
        layer_embedding_dim=int(teacher_cfg.get("layer_embedding_dim", 16)),
        dropout=float(teacher_cfg.get("dropout", 0.05)),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return {"model": model, "ckpt": ckpt}


def _find_layers_container(model: torch.nn.Module):
    candidates = [
        "language_model.model.layers",
        "language_model.layers",
        "model.layers",
        "transformer.h",
    ]
    for name in candidates:
        try:
            module = model.get_submodule(name)
        except AttributeError:
            module = None
            cur = model
            for part in name.split("."):
                cur = getattr(cur, part, None)
                if cur is None:
                    break
            module = cur
        except Exception:
            module = None
        if module is not None and hasattr(module, "__getitem__") and hasattr(module, "__len__"):
            return module, name
    raise RuntimeError(
        "Could not locate transformer layers. Tried: "
        + ", ".join(candidates)
        + ". Add this model's layer path to infer_time_flow.controller._find_layers_container."
    )


class InferenceTimeFlowController:
    """Hidden-state intervention controller driven by a trained Flow Navigator.

    The training-time Flow teacher uses paired targets. During inference those
    targets are unavailable, so this controller builds an inference-time
    condition from the current hidden state's risk-subspace coordinates and the
    dynamic Stage2-normalized implicit risk R_imp_norm(t).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        config_path: str,
        flow_teacher_path: Optional[str] = None,
        hidden_layers: Optional[Iterable[int]] = None,
        strength: float = 0.25,
        risk_gate_threshold: float = 0.0,
        risk_gate_mode: str = "fused",
        max_delta_norm_ratio: float = 0.20,
        explicit_risk: float = 1.0,
        group_id: int = 0,
        intervene_on_prefill: bool = True,
        intervene_on_decode: bool = True,
        risk_trace_max_records: int = 200000,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.config = sync_stage3_layers_with_recommendation(load_config(config_path))
        self.device = device or infer_input_device(model)
        flow_cfg = self.config.get("stage3", {}).get("flow_distillation", {})
        raw_flow_path = flow_teacher_path or flow_cfg.get("teacher_path", "integrations/my_method/outputs/stage2_5_flow/flow_teacher.pt")
        self.flow_path = resolve_path(self.config, raw_flow_path)
        self.bundle = _load_flow_teacher(self.flow_path, self.device)
        self.teacher: FlowVectorField = self.bundle["model"]
        self.ckpt = self.bundle["ckpt"]
        self.recommended: RecommendedRiskConfig = load_recommended_risk_config(self.config, allow_fallback=False)
        self.hidden_layers = sorted({int(x) for x in (hidden_layers or self.config["stage3"]["risk_space"].get("risk_layers") or self.recommended.recommended_hidden_layers)})
        self.block_layers = {layer - 1: layer for layer in self.hidden_layers if int(layer) > 0}
        if not self.block_layers:
            raise ValueError(f"No valid hidden layers for inference-time intervention: {self.hidden_layers}")
        self.risk_tensors = load_risk_tensors(self.config, self.device)
        self.norm_lower, self.norm_upper, self.norm_clip = load_dynamic_implicit_normalization(self.config)
        self.cond_dim = int(self.ckpt["cond_dim"])
        dynamic_cfg = self.ckpt.get("dynamic_conditioning") or {}
        self.use_dynamic_r_imp = bool(dynamic_cfg.get("R_imp_norm_t", False))
        if self.use_dynamic_r_imp and dynamic_cfg.get("normalization") != "per_layer_safe_harmful_percentile":
            raise RuntimeError(
                "Flow teacher uses the old aggregate Stage2 normalization for dynamic R_imp(t). "
                "Retrain Stage 2.5 before inference-time Flow intervention."
            )
        self.static_cond_dim = int(self.ckpt.get("static_cond_dim", self.cond_dim - 1 if self.use_dynamic_r_imp else self.cond_dim))
        self.strength = float(strength)
        self.risk_gate_threshold = float(risk_gate_threshold)
        self.risk_gate_mode = str(risk_gate_mode).lower()
        if self.risk_gate_mode not in {"fused", "implicit"}:
            raise ValueError(f"Unsupported risk_gate_mode={risk_gate_mode!r}; use 'fused' or 'implicit'.")
        self.max_delta_norm_ratio = float(max_delta_norm_ratio)
        self.explicit_risk = float(explicit_risk)
        self.group_id = int(group_id)
        self._context_explicit_risk = float(explicit_risk)
        self._context_group_id = int(group_id)
        self.intervene_on_prefill = bool(intervene_on_prefill)
        self.intervene_on_decode = bool(intervene_on_decode)
        self.handles: List[Any] = []
        self.stats = FlowInterventionStats()
        self.stats.risk_gate_mode = self.risk_gate_mode
        self.risk_trace_max_records = int(risk_trace_max_records)
        self.risk_trace: List[Dict[str, Any]] = []
        self._enabled = False

    def _append_trace(
        self,
        *,
        call_index: int,
        hidden_layer: int,
        phase: str,
        implicit_risk: float,
        explicit_risk: float,
        gate_risk: float,
        gate: float,
        active: bool,
        delta_norm: float,
        skip_reason: Optional[str] = None,
    ) -> None:
        if self.risk_trace_max_records <= 0:
            return
        if len(self.risk_trace) >= self.risk_trace_max_records:
            self.stats.trace_dropped += 1
            return
        self.risk_trace.append(
            {
                "call_index": int(call_index),
                "hidden_layer": int(hidden_layer),
                "phase": phase,
                "risk_gate_mode": self.risk_gate_mode,
                "R_imp_norm": float(implicit_risk),
                "R_exp": float(explicit_risk),
                "R_gate": float(gate_risk),
                "gate": float(gate),
                "active": bool(active),
                "delta_norm": float(delta_norm),
                "skip_reason": skip_reason,
                "strength": self.strength,
                "risk_gate_threshold": self.risk_gate_threshold,
                "max_delta_norm_ratio": self.max_delta_norm_ratio,
                "group_id": int(self._context_group_id),
            }
        )

    def write_risk_trace(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for row in self.risk_trace:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _static_condition(self, x: torch.Tensor, hidden_layer: int) -> torch.Tensor:
        coeff = compute_risk_coefficients(
            x[None, :],
            hidden_layer,
            self.recommended,
            self.risk_tensors["risk_basis"],
            self.risk_tensors["safe_center"],
        )[0]
        group = torch.zeros(3, dtype=x.dtype, device=x.device)
        group[self._context_group_id] = 1.0
        cond = torch.cat(
            [
                x,
                coeff.to(x.dtype),
                torch.tensor([self._context_explicit_risk], dtype=x.dtype, device=x.device),
                group,
            ],
            dim=0,
        )
        if cond.numel() < self.static_cond_dim:
            cond = torch.nn.functional.pad(cond, (0, self.static_cond_dim - cond.numel()))
        elif cond.numel() > self.static_cond_dim:
            cond = cond[: self.static_cond_dim]
        return cond[None, :]

    def _condition(self, x: torch.Tensor, hidden_layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        cond = self._static_condition(x, hidden_layer)
        layer_id = torch.tensor([hidden_layer], device=x.device, dtype=torch.long)
        risk = dynamic_implicit_risk_norm(
            x[None, :],
            layer_id,
            self.recommended,
            self.risk_tensors["risk_basis"],
            self.risk_tensors["safe_center"],
            self.norm_lower,
            self.norm_upper,
            clip=self.norm_clip,
        )
        if self.use_dynamic_r_imp:
            cond = torch.cat([cond, risk.to(dtype=cond.dtype)], dim=-1)
        return cond, risk

    def _intervene_vector(self, h: torch.Tensor, hidden_layer: int, *, phase: str) -> torch.Tensor:
        original_device = h.device
        original_dtype = h.dtype
        x = h.detach().to(device=self.device, dtype=next(self.teacher.parameters()).dtype)
        cond, risk = self._condition(x, hidden_layer)
        explicit_risk = torch.tensor([[self._context_explicit_risk]], device=risk.device, dtype=risk.dtype)
        call_index = self.stats.calls + 1
        if not torch.isfinite(x).all() or not torch.isfinite(cond).all() or not torch.isfinite(risk).all():
            self._append_trace(
                call_index=call_index,
                hidden_layer=hidden_layer,
                phase=phase,
                implicit_risk=float(risk.item()) if risk.numel() == 1 else float("nan"),
                explicit_risk=float(explicit_risk.item()),
                gate_risk=float("nan"),
                gate=0.0,
                active=False,
                delta_norm=0.0,
                skip_reason="nonfinite_condition",
            )
            self.stats.calls += 1
            self.stats.numerical_skips += 1
            return h
        if self.risk_gate_mode == "implicit":
            total_risk = risk
        else:
            total_risk = 0.5 * explicit_risk + 0.5 * risk
        threshold = self.risk_gate_threshold
        gate = torch.clamp((total_risk - threshold) / max(1.0 - threshold, 1e-6), 0.0, 1.0)
        if float(gate.item()) <= 0.0 or self.strength == 0.0:
            self._append_trace(
                call_index=call_index,
                hidden_layer=hidden_layer,
                phase=phase,
                implicit_risk=float(risk.item()),
                explicit_risk=float(explicit_risk.item()),
                gate_risk=float(total_risk.item()),
                gate=float(gate.item()),
                active=False,
                delta_norm=0.0,
            )
            self.stats.calls += 1
            self.stats.mean_risk_sum += float(risk.item())
            self.stats.mean_explicit_risk_sum += float(explicit_risk.item())
            self.stats.mean_total_risk_sum += float(total_risk.item())
            return h
        t = torch.zeros(1, 1, device=self.device, dtype=x.dtype)
        layer_id = torch.tensor([hidden_layer], device=self.device, dtype=torch.long)
        with torch.no_grad():
            velocity = self.teacher(x[None, :], t, cond, layer_id)[0]
        if not torch.isfinite(velocity).all():
            self._append_trace(
                call_index=call_index,
                hidden_layer=hidden_layer,
                phase=phase,
                implicit_risk=float(risk.item()),
                explicit_risk=float(explicit_risk.item()),
                gate_risk=float(total_risk.item()),
                gate=float(gate.item()),
                active=False,
                delta_norm=0.0,
                skip_reason="nonfinite_velocity",
            )
            self.stats.calls += 1
            self.stats.numerical_skips += 1
            return h
        delta = self.strength * gate[0, 0].to(velocity.dtype) * velocity
        if not torch.isfinite(delta).all():
            self._append_trace(
                call_index=call_index,
                hidden_layer=hidden_layer,
                phase=phase,
                implicit_risk=float(risk.item()),
                explicit_risk=float(explicit_risk.item()),
                gate_risk=float(total_risk.item()),
                gate=float(gate.item()),
                active=False,
                delta_norm=0.0,
                skip_reason="nonfinite_delta",
            )
            self.stats.calls += 1
            self.stats.numerical_skips += 1
            return h
        if self.max_delta_norm_ratio > 0:
            max_norm = self.max_delta_norm_ratio * x.norm().clamp_min(1e-6)
            delta_norm = delta.norm()
            if delta_norm > max_norm:
                delta = delta * (max_norm / delta_norm)
        out = (x + delta).to(device=original_device, dtype=original_dtype)
        if not torch.isfinite(out).all():
            self._append_trace(
                call_index=call_index,
                hidden_layer=hidden_layer,
                phase=phase,
                implicit_risk=float(risk.item()),
                explicit_risk=float(explicit_risk.item()),
                gate_risk=float(total_risk.item()),
                gate=float(gate.item()),
                active=False,
                delta_norm=float(delta.norm().detach().cpu()),
                skip_reason="nonfinite_output",
            )
            self.stats.calls += 1
            self.stats.numerical_skips += 1
            return h
        delta_norm_value = float(delta.norm().detach().cpu())
        self._append_trace(
            call_index=call_index,
            hidden_layer=hidden_layer,
            phase=phase,
            implicit_risk=float(risk.item()),
            explicit_risk=float(explicit_risk.item()),
            gate_risk=float(total_risk.item()),
            gate=float(gate.item()),
            active=True,
            delta_norm=delta_norm_value,
        )
        self.stats.calls += 1
        self.stats.active_calls += 1
        self.stats.mean_gate_sum += float(gate.item())
        self.stats.mean_risk_sum += float(risk.item())
        self.stats.mean_explicit_risk_sum += float(explicit_risk.item())
        self.stats.mean_total_risk_sum += float(total_risk.item())
        self.stats.mean_delta_norm_sum += delta_norm_value
        return out

    def _should_intervene(self, hidden: torch.Tensor) -> bool:
        if hidden.ndim != 3 or hidden.shape[0] != 1:
            return False
        is_decode = hidden.shape[1] == 1
        return self.intervene_on_decode if is_decode else self.intervene_on_prefill

    def _make_hook(self, hidden_layer: int):
        def hook(_module, _inputs, output):
            if not self._enabled:
                return output
            raw_hidden = output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(raw_hidden) or not self._should_intervene(raw_hidden):
                return output
            phase = "decode" if raw_hidden.shape[1] == 1 else "prefill"
            source = raw_hidden[0, -1, :].clone()
            intervened = self._intervene_vector(source, hidden_layer, phase=phase)
            hidden = raw_hidden.clone()
            hidden[0, -1, :] = intervened
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden

        return hook

    def register(self) -> None:
        if self.handles:
            return
        layers, layer_path = _find_layers_container(self.model)
        for block_idx, hidden_layer in self.block_layers.items():
            if block_idx < 0 or block_idx >= len(layers):
                raise IndexError(f"Requested hidden layer {hidden_layer} -> block {block_idx}, but {layer_path} has {len(layers)} blocks")
            self.handles.append(layers[block_idx].register_forward_hook(self._make_hook(hidden_layer)))

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

    @contextmanager
    def context(self, *, explicit_risk: Optional[float] = None, group_id: Optional[int] = None):
        old_explicit = self._context_explicit_risk
        old_group = self._context_group_id
        if explicit_risk is not None:
            self._context_explicit_risk = float(explicit_risk)
        if group_id is not None:
            self._context_group_id = int(group_id)
        try:
            yield self
        finally:
            self._context_explicit_risk = old_explicit
            self._context_group_id = old_group

    @contextmanager
    def enabled(self, *, explicit_risk: Optional[float] = None, group_id: Optional[int] = None):
        self.register()
        old = self._enabled
        self._enabled = True
        try:
            with self.context(explicit_risk=explicit_risk, group_id=group_id):
                yield self
        finally:
            self._enabled = old
