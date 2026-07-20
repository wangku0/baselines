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
    strength: float = 0.0
    decode_strength: Optional[float] = None
    decode_max_steps: Optional[int] = None
    decode_steering_mode: str = "flow"
    prefix_direction_path: Optional[str] = None
    intervention_group_ids: Optional[List[int]] = None
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
            "strength": self.strength,
            "decode_strength": self.decode_strength,
            "decode_max_steps": self.decode_max_steps,
            "decode_steering_mode": self.decode_steering_mode,
            "prefix_direction_path": self.prefix_direction_path,
            "intervention_group_ids": self.intervention_group_ids,
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


def _load_prefix_directions(path: Path, device: torch.device, dtype: torch.dtype) -> Dict[int, torch.Tensor]:
    if not path.exists():
        raise FileNotFoundError(f"Safe-prefix direction file not found: {path}")
    data = torch.load(path, map_location="cpu", weights_only=False)
    raw = data.get("directions") if isinstance(data, dict) else None
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"Safe-prefix direction file has no non-empty 'directions' dict: {path}")
    return {int(layer): direction.to(device=device, dtype=dtype) for layer, direction in raw.items()}


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
        decode_strength: Optional[float] = None,
        risk_gate_threshold: float = 0.0,
        risk_gate_mode: str = "fused",
        max_delta_norm_ratio: float = 0.20,
        explicit_risk: float = 1.0,
        group_id: int = 0,
        intervention_group_ids: Optional[Iterable[int]] = None,
        intervene_on_prefill: bool = True,
        intervene_on_decode: bool = True,
        decode_max_steps: Optional[int] = None,
        decode_steering_mode: str = "flow",
        prefix_direction_path: Optional[str] = None,
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
        if self.ckpt.get("recommended"):
            self.recommended = RecommendedRiskConfig(**self.ckpt["recommended"])
        else:
            self.recommended = load_recommended_risk_config(self.config, allow_fallback=False)
        self.hidden_layers = sorted({int(x) for x in (hidden_layers or self.config["stage3"]["risk_space"].get("risk_layers") or self.recommended.recommended_hidden_layers)})
        self.block_layers = {layer - 1: layer for layer in self.hidden_layers if int(layer) > 0}
        if not self.block_layers:
            raise ValueError(f"No valid hidden layers for inference-time intervention: {self.hidden_layers}")
        self.risk_tensors = load_risk_tensors(self.config, self.device)
        self.norm_lower, self.norm_upper, self.norm_clip = load_dynamic_implicit_normalization(self.config)
        self.cond_dim = int(self.ckpt["cond_dim"])
        dynamic_cfg = self.ckpt.get("dynamic_conditioning") or {}
        self.use_dynamic_r_imp = bool(dynamic_cfg.get("R_imp_norm_t", False))
        if self.use_dynamic_r_imp and dynamic_cfg.get("normalization") != "stage2_sample_risk":
            raise RuntimeError(
                "Flow teacher uses a stale dynamic-risk normalization for R_imp(t). "
                "Retrain Stage 2.5 before inference-time Flow intervention."
            )
        self.static_cond_dim = int(self.ckpt.get("static_cond_dim", self.cond_dim - 1 if self.use_dynamic_r_imp else self.cond_dim))
        self.strength = float(strength)
        self.decode_strength = None if decode_strength is None else float(decode_strength)
        self.risk_gate_threshold = float(risk_gate_threshold)
        self.risk_gate_mode = str(risk_gate_mode).lower()
        if self.risk_gate_mode not in {
            "fused",
            "implicit",
            "prefill_fused_decode_implicit",
            "prefill_fused_decode_fused",
        }:
            raise ValueError(
                f"Unsupported risk_gate_mode={risk_gate_mode!r}; "
                "use 'fused', 'implicit', 'prefill_fused_decode_implicit', "
                "or 'prefill_fused_decode_fused'."
            )
        self.max_delta_norm_ratio = float(max_delta_norm_ratio)
        self.explicit_risk = float(explicit_risk)
        self.group_id = int(group_id)
        self.intervention_group_ids = (
            None
            if intervention_group_ids is None
            else {int(group_id) for group_id in intervention_group_ids}
        )
        self.decode_max_steps = None if decode_max_steps is None else int(decode_max_steps)
        if self.decode_max_steps is not None and self.decode_max_steps < 0:
            raise ValueError("--decode-max-steps must be >= 0 when provided.")
        self.decode_steering_mode = str(decode_steering_mode).lower()
        if self.decode_steering_mode not in {"flow", "safe_prefix"}:
            raise ValueError("--decode-steering-mode must be 'flow' or 'safe_prefix'.")
        self.prefix_direction_path = None
        self.prefix_directions: Dict[int, torch.Tensor] = {}
        if prefix_direction_path:
            self.prefix_direction_path = str(resolve_path(self.config, prefix_direction_path))
            self.prefix_directions = _load_prefix_directions(
                Path(self.prefix_direction_path),
                self.device,
                next(self.teacher.parameters()).dtype,
            )
        if self.decode_steering_mode == "safe_prefix" and not self.prefix_directions:
            raise ValueError("--decode-steering-mode safe_prefix requires --prefix-direction-path.")
        self._decode_step_anchor_layer = min(self.hidden_layers)
        self._decode_step_index = 0
        self._context_explicit_risk = float(explicit_risk)
        self._context_group_id = int(group_id)
        self._batch_explicit_risks: Optional[List[float]] = None
        self._batch_group_ids: Optional[List[int]] = None
        self.intervene_on_prefill = bool(intervene_on_prefill)
        self.intervene_on_decode = bool(intervene_on_decode)
        self.handles: List[Any] = []
        self.stats = FlowInterventionStats()
        self.stats.risk_gate_mode = self.risk_gate_mode
        self.stats.strength = self.strength
        self.stats.decode_strength = self.decode_strength
        self.stats.decode_max_steps = self.decode_max_steps
        self.stats.decode_steering_mode = self.decode_steering_mode
        self.stats.prefix_direction_path = self.prefix_direction_path
        self.stats.intervention_group_ids = (
            None if self.intervention_group_ids is None else sorted(self.intervention_group_ids)
        )
        self.risk_trace_max_records = int(risk_trace_max_records)
        self.risk_trace: List[Dict[str, Any]] = []
        self._enabled = False

    def _phase_strength(self, phase: str) -> float:
        if phase == "decode" and self.decode_strength is not None:
            return self.decode_strength
        return self.strength

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
        phase_strength: Optional[float] = None,
        target_dist: Optional[float] = None,
        proj_ratio: Optional[float] = None,
        target_basis: Optional[str] = None,
        decode_step: Optional[int] = None,
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
                "target_dist": None if target_dist is None else float(target_dist),
                "proj_ratio": None if proj_ratio is None else float(proj_ratio),
                "target_basis": target_basis,
                "skip_reason": skip_reason,
                "strength": float(self.strength if phase_strength is None else phase_strength),
                "base_strength": float(self.strength),
                "decode_strength": None if self.decode_strength is None else float(self.decode_strength),
                "decode_max_steps": self.decode_max_steps,
                "decode_step": None if decode_step is None else int(decode_step),
                "intervention_group_ids": (
                    None if self.intervention_group_ids is None else sorted(self.intervention_group_ids)
                ),
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

    def _intervene_vector(self, h: torch.Tensor, hidden_layer: int, *, phase: str, decode_step: Optional[int] = None) -> torch.Tensor:
        original_device = h.device
        original_dtype = h.dtype
        x = h.detach().to(device=self.device, dtype=next(self.teacher.parameters()).dtype)
        cond, risk = self._condition(x, hidden_layer)
        explicit_risk = torch.tensor([[self._context_explicit_risk]], device=risk.device, dtype=risk.dtype)
        call_index = self.stats.calls + 1
        phase_strength = self._phase_strength(phase)
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
                phase_strength=phase_strength,
                decode_step=decode_step,
                skip_reason="nonfinite_condition",
            )
            self.stats.calls += 1
            self.stats.numerical_skips += 1
            return h
        if self.risk_gate_mode == "implicit" or (
            self.risk_gate_mode == "prefill_fused_decode_implicit" and phase == "decode"
        ):
            total_risk = risk
        else:
            total_risk = 0.5 * explicit_risk + 0.5 * risk
        if (
            self.intervention_group_ids is not None
            and int(self._context_group_id) not in self.intervention_group_ids
        ):
            self._append_trace(
                call_index=call_index,
                hidden_layer=hidden_layer,
                phase=phase,
                implicit_risk=float(risk.item()),
                explicit_risk=float(explicit_risk.item()),
                gate_risk=float(total_risk.item()),
                gate=0.0,
                active=False,
                delta_norm=0.0,
                phase_strength=phase_strength,
                decode_step=decode_step,
                skip_reason="group_not_allowed",
            )
            self.stats.calls += 1
            self.stats.mean_risk_sum += float(risk.item())
            self.stats.mean_explicit_risk_sum += float(explicit_risk.item())
            self.stats.mean_total_risk_sum += float(total_risk.item())
            return h
        threshold = self.risk_gate_threshold
        gate = torch.clamp((total_risk - threshold) / max(1.0 - threshold, 1e-6), 0.0, 1.0)
        if phase == "decode" and self.decode_max_steps is not None and (decode_step or 0) > self.decode_max_steps:
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
                phase_strength=phase_strength,
                decode_step=decode_step,
                skip_reason="decode_step_limit",
            )
            self.stats.calls += 1
            self.stats.mean_risk_sum += float(risk.item())
            self.stats.mean_explicit_risk_sum += float(explicit_risk.item())
            self.stats.mean_total_risk_sum += float(total_risk.item())
            return h
        target_basis = "flow_velocity_proxy"
        if float(gate.item()) <= 0.0 or phase_strength == 0.0:
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
                phase_strength=phase_strength,
                target_basis=target_basis,
                decode_step=decode_step,
            )
            self.stats.calls += 1
            self.stats.mean_risk_sum += float(risk.item())
            self.stats.mean_explicit_risk_sum += float(explicit_risk.item())
            self.stats.mean_total_risk_sum += float(total_risk.item())
            return h
        if phase == "decode" and self.decode_steering_mode == "safe_prefix":
            velocity = self.prefix_directions.get(int(hidden_layer))
            target_basis = "safe_prefix_direction"
            if velocity is None:
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
                    phase_strength=phase_strength,
                    target_basis=target_basis,
                    decode_step=decode_step,
                    skip_reason="missing_prefix_direction",
                )
                self.stats.calls += 1
                self.stats.mean_risk_sum += float(risk.item())
                self.stats.mean_explicit_risk_sum += float(explicit_risk.item())
                self.stats.mean_total_risk_sum += float(total_risk.item())
                return h
        else:
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
                phase_strength=phase_strength,
                target_basis=target_basis,
                decode_step=decode_step,
                skip_reason="nonfinite_velocity",
            )
            self.stats.calls += 1
            self.stats.numerical_skips += 1
            return h
        delta = phase_strength * gate[0, 0].to(velocity.dtype) * velocity
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
                phase_strength=phase_strength,
                target_basis=target_basis,
                decode_step=decode_step,
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
        target_dist_value = float(velocity.norm().detach().cpu())
        proj_ratio_value = float(
            torch.dot(delta.flatten(), velocity.flatten()).div(velocity.norm().pow(2).clamp_min(1e-12)).detach().cpu()
        )
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
                phase_strength=phase_strength,
                target_dist=target_dist_value,
                proj_ratio=proj_ratio_value,
                target_basis=target_basis,
                decode_step=decode_step,
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
            phase_strength=phase_strength,
            target_dist=target_dist_value,
            proj_ratio=proj_ratio_value,
            target_basis=target_basis,
            decode_step=decode_step,
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
        if hidden.ndim != 3 or hidden.shape[0] < 1:
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
            decode_step = None
            if phase == "decode":
                if hidden_layer == self._decode_step_anchor_layer:
                    self._decode_step_index += 1
                decode_step = self._decode_step_index
            hidden = raw_hidden.clone()
            for batch_idx in range(raw_hidden.shape[0]):
                source = raw_hidden[batch_idx, -1, :].clone()
                with self._batch_item_context(batch_idx):
                    hidden[batch_idx, -1, :] = self._intervene_vector(source, hidden_layer, phase=phase, decode_step=decode_step)
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
    def _batch_item_context(self, batch_idx: int):
        if self._batch_explicit_risks is None and self._batch_group_ids is None:
            yield self
            return
        old_explicit = self._context_explicit_risk
        old_group = self._context_group_id
        try:
            if self._batch_explicit_risks is not None:
                if batch_idx >= len(self._batch_explicit_risks):
                    raise IndexError(
                        f"Batch explicit-risk context has {len(self._batch_explicit_risks)} items, "
                        f"but model hidden batch index {batch_idx} was requested."
                    )
                self._context_explicit_risk = float(self._batch_explicit_risks[batch_idx])
            if self._batch_group_ids is not None:
                if batch_idx >= len(self._batch_group_ids):
                    raise IndexError(
                        f"Batch group-id context has {len(self._batch_group_ids)} items, "
                        f"but model hidden batch index {batch_idx} was requested."
                    )
                self._context_group_id = int(self._batch_group_ids[batch_idx])
            yield self
        finally:
            self._context_explicit_risk = old_explicit
            self._context_group_id = old_group

    @contextmanager
    def batch_context(
        self,
        *,
        explicit_risks: Optional[Iterable[float]] = None,
        group_ids: Optional[Iterable[int]] = None,
    ):
        old_explicit = self._batch_explicit_risks
        old_group = self._batch_group_ids
        self._batch_explicit_risks = [float(x) for x in explicit_risks] if explicit_risks is not None else None
        self._batch_group_ids = [int(x) for x in group_ids] if group_ids is not None else None
        if (
            self._batch_explicit_risks is not None
            and self._batch_group_ids is not None
            and len(self._batch_explicit_risks) != len(self._batch_group_ids)
        ):
            raise ValueError("explicit_risks and group_ids must have the same length.")
        try:
            yield self
        finally:
            self._batch_explicit_risks = old_explicit
            self._batch_group_ids = old_group

    @contextmanager
    def enabled(self, *, explicit_risk: Optional[float] = None, group_id: Optional[int] = None):
        self.register()
        old = self._enabled
        old_decode_step = self._decode_step_index
        self._enabled = True
        self._decode_step_index = 0
        try:
            with self.context(explicit_risk=explicit_risk, group_id=group_id):
                yield self
        finally:
            self._decode_step_index = old_decode_step
            self._enabled = old

    @contextmanager
    def enabled_batch(
        self,
        *,
        explicit_risks: Optional[Iterable[float]] = None,
        group_ids: Optional[Iterable[int]] = None,
    ):
        self.register()
        old = self._enabled
        old_decode_step = self._decode_step_index
        self._enabled = True
        self._decode_step_index = 0
        try:
            with self.batch_context(explicit_risks=explicit_risks, group_ids=group_ids):
                yield self
        finally:
            self._decode_step_index = old_decode_step
            self._enabled = old
