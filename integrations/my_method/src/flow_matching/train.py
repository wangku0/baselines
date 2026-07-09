from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
import yaml

from ..risk_space.recommended_config import RecommendedRiskConfig
from ..utils import ensure_dir, logger, resolve_path, save_json, set_seed
from .features import _load_risk_basis, build_flow_features
from .model import FlowVectorField, euler_integrate_flow_trainable
from .utils import dynamic_implicit_risk_norm, load_dynamic_implicit_normalization


def _batch(examples, indices, device):
    x0 = torch.stack([examples[i]["x0"] for i in indices]).to(device)
    x1 = torch.stack([examples[i]["x1"] for i in indices]).to(device)
    cond = torch.stack([examples[i]["cond"] for i in indices]).to(device)
    layer = torch.tensor([int(examples[i]["layer"]) for i in indices], dtype=torch.long, device=device)
    kind = [examples[i]["kind"] for i in indices]
    return x0, x1, cond, layer, kind


def _append_dynamic_risk_condition(
    x: torch.Tensor,
    cond_static: torch.Tensor,
    layer: torch.Tensor,
    recommended: RecommendedRiskConfig,
    risk_basis: Dict[int, torch.Tensor],
    safe_center: Dict[int, torch.Tensor],
    lower: Dict[int, float],
    upper: Dict[int, float],
    clip: bool,
) -> torch.Tensor:
    r_imp_norm_t = dynamic_implicit_risk_norm(
        x,
        layer,
        recommended,
        risk_basis,
        safe_center,
        lower,
        upper,
        clip=clip,
    )
    return torch.cat([cond_static.to(x.dtype), r_imp_norm_t.to(x.dtype)], dim=-1)


def train_flow_teacher(
    config: Dict[str, Any],
    *,
    split: str = "train",
    max_pairs: Optional[int] = None,
    steps: Optional[int] = None,
    batch_size: Optional[int] = None,
    device: Optional[str] = None,
    debug: bool = False,
    recommended_config_path: Optional[str] = None,
) -> Path:
    flow_cfg = config.get("flow_matching", {})
    out_dir = ensure_dir(resolve_path(config, flow_cfg.get("output_dir", "integrations/my_method/outputs/stage2_5_flow")))
    set_seed(int(config.get("stage3", {}).get("training", {}).get("seed", 42)))
    feature_path = build_flow_features(config, split=split, max_pairs=max_pairs, debug=debug, recommended_config_path=recommended_config_path)
    data = torch.load(feature_path, map_location="cpu", weights_only=False)
    examples = data["examples"]
    rec = data["recommended"]
    rec_cfg = RecommendedRiskConfig(**rec)
    risk_basis, safe_center = _load_risk_basis(Path(rec_cfg.risk_basis_path))
    lower, upper, norm_clip = load_dynamic_implicit_normalization(config)
    teacher_cfg = flow_cfg.get("teacher", {})
    losses_cfg = flow_cfg.get("teacher_losses", {})
    hidden_dim = int(data["hidden_dim"])
    cond_dim = int(data["cond_dim"])
    steps = int(steps if steps is not None else teacher_cfg.get("steps", 8000))
    batch_size = int(batch_size if batch_size is not None else teacher_cfg.get("batch_size", 64))
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = FlowVectorField(
        hidden_dim=hidden_dim,
        cond_dim=cond_dim,
        hidden_width=int(teacher_cfg.get("hidden_width", 1024)),
        time_embedding_dim=int(teacher_cfg.get("time_embedding_dim", 128)),
        layer_embedding_dim=int(teacher_cfg.get("layer_embedding_dim", 16)),
        dropout=float(teacher_cfg.get("dropout", 0.05)),
    ).to(device_t)
    opt = torch.optim.AdamW(model.parameters(), lr=float(teacher_cfg.get("lr", 1e-3)), weight_decay=float(teacher_cfg.get("weight_decay", 1e-4)))
    ode_steps = int(teacher_cfg.get("ode_steps", 8))
    grad_clip = float(teacher_cfg.get("grad_clip", 1.0))
    log_path = out_dir / "train_log.csv"
    fields = ["step", "loss", "loss_velocity", "loss_endpoint", "loss_identity", "endpoint_noop", "delta_cos"]
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        n = len(examples)
        for step in range(1, steps + 1):
            idx = torch.randint(0, n, (min(batch_size, n),)).tolist()
            x0, x1, cond, layer, kind = _batch(examples, idx, device_t)
            t = torch.rand(x0.shape[0], 1, device=device_t)
            xt = (1 - t) * x0 + t * x1
            target_v = x1 - x0
            cond_t = _append_dynamic_risk_condition(
                xt,
                cond,
                layer,
                rec_cfg,
                risk_basis,
                safe_center,
                lower,
                upper,
                norm_clip,
            )
            pred_v = model(xt, t, cond_t, layer)
            identity_mask = torch.tensor([k == "identity" for k in kind], device=device_t)
            pair_mask = ~identity_mask
            loss_velocity = F.mse_loss(pred_v[pair_mask], target_v[pair_mask]) if pair_mask.any() else pred_v.sum() * 0
            xhat = euler_integrate_flow_trainable(
                model,
                x0,
                cond,
                layer_id=layer,
                steps=ode_steps,
                cond_fn=lambda x, _t, c, lid: _append_dynamic_risk_condition(
                    x,
                    c,
                    lid if lid is not None else layer,
                    rec_cfg,
                    risk_basis,
                    safe_center,
                    lower,
                    upper,
                    norm_clip,
                ),
            )
            loss_endpoint = F.mse_loss(xhat[pair_mask], x1[pair_mask]) if pair_mask.any() else pred_v.sum() * 0
            loss_identity = (pred_v[identity_mask].pow(2).mean() + F.mse_loss(xhat[identity_mask], x0[identity_mask])) if identity_mask.any() else pred_v.sum() * 0
            loss = (
                float(losses_cfg.get("velocity", 1.0)) * loss_velocity
                + float(losses_cfg.get("endpoint", 0.5)) * loss_endpoint
                + float(losses_cfg.get("identity_safe_retain", 0.5)) * loss_identity
            )
            if not torch.isfinite(loss):
                torch.save(
                    {
                        "x0": x0.detach().cpu(),
                        "x1": x1.detach().cpu(),
                        "cond_static": cond.detach().cpu(),
                        "cond_t": cond_t.detach().cpu(),
                    },
                    out_dir / "nan_batch_debug.pt",
                )
                raise RuntimeError("Flow teacher loss became NaN/Inf. Saved nan_batch_debug.pt")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            if step == 1 or step % int(teacher_cfg.get("eval_every", 500)) == 0 or step == steps:
                with torch.no_grad():
                    noop = F.mse_loss(x0[pair_mask], x1[pair_mask]) if pair_mask.any() else torch.tensor(0.0, device=device_t)
                    cos = F.cosine_similarity((xhat - x0)[pair_mask], (x1 - x0)[pair_mask], dim=-1).mean() if pair_mask.any() else torch.tensor(0.0, device=device_t)
                row = {
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "loss_velocity": float(loss_velocity.detach().cpu()),
                    "loss_endpoint": float(loss_endpoint.detach().cpu()),
                    "loss_identity": float(loss_identity.detach().cpu()),
                    "endpoint_noop": float(noop.detach().cpu()),
                    "delta_cos": float(cos.detach().cpu()),
                }
                writer.writerow(row)
                f.flush()
                print(f"Flow step {step} | loss={row['loss']:.4f} endpoint={row['loss_endpoint']:.4f} noop={row['endpoint_noop']:.4f} cos={row['delta_cos']:.4f}")
            if step % int(teacher_cfg.get("save_every", 1000)) == 0:
                _save(model, out_dir, data, rec, teacher_cfg)
    _save(model, out_dir, data, rec, teacher_cfg)
    save_json({"recommended": rec, "train_log": str(log_path)}, out_dir / "eval_summary.json")
    with (out_dir / "flow_config_resolved.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "flow_matching": flow_cfg,
                "recommended": rec,
                "hidden_dim": hidden_dim,
                "cond_dim": cond_dim,
                "static_cond_dim": int(data.get("static_cond_dim", cond_dim - 1)),
                "dynamic_conditioning": data.get("dynamic_conditioning"),
                "flow_target": data.get("flow_target"),
                "representation_pooling": data.get("representation_pooling"),
                "requested_representation_pooling": data.get("requested_representation_pooling"),
            },
            f,
            allow_unicode=True,
            sort_keys=False,
        )
    return out_dir / "flow_teacher.pt"


def _save(model, out_dir: Path, data: Dict[str, Any], rec: Dict[str, Any], teacher_cfg: Dict[str, Any]) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hidden_dim": int(data["hidden_dim"]),
            "cond_dim": int(data["cond_dim"]),
            "static_cond_dim": int(data.get("static_cond_dim", int(data["cond_dim"]) - 1)),
            "dynamic_conditioning": data.get("dynamic_conditioning"),
            "recommended": rec,
            "teacher_cfg": dict(teacher_cfg),
            "flow_target": data.get("flow_target"),
        },
        out_dir / "flow_teacher.pt",
    )
