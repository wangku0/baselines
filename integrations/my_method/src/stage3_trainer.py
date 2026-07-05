from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import torch
import yaml

from .flow_matching.stage3_integration import compute_flow_stage3_losses, load_flow_teacher
from .risk_space.recommended_config import RecommendedRiskConfig, load_recommended_risk_config, save_resolved_recommended_config
from .stage3_data import build_stage3_triplets
from .stage3_lora_utils import (
    add_lora_adapter,
    get_trainable_parameter_summary,
    load_base_model_and_processor,
    save_lora_adapter,
    sync_stage3_layers_with_recommendation,
)
from .stage3_losses import compute_triplet_losses, load_risk_tensors
from .stage3_visualize import plot_loss_curve
from .utils import cuda_oom_help, ensure_dir, logger, resolve_path, save_json, set_seed


def _override(
    config: Dict[str, Any],
    *,
    max_steps=None,
    max_train_samples=None,
    learning_rate=None,
    output_dir=None,
    gradient_accumulation_steps=None,
):
    cfg = copy.deepcopy(config)
    if max_steps is not None:
        cfg["stage3"]["training"]["max_steps"] = int(max_steps)
    if max_train_samples is not None:
        cfg["stage3"]["data"]["max_train_samples"] = int(max_train_samples)
    if learning_rate is not None:
        cfg["stage3"]["training"]["learning_rate"] = float(learning_rate)
    if output_dir is not None:
        cfg["stage3"]["training"]["output_dir"] = output_dir
        cfg["stage3"]["outputs"]["adapter_dir"] = str(Path(output_dir) / "adapter")
    if gradient_accumulation_steps is not None:
        cfg["stage3"]["training"]["gradient_accumulation_steps"] = int(gradient_accumulation_steps)
    return cfg


def _resolved_recommended_for_stage3(config: Dict[str, Any]) -> RecommendedRiskConfig:
    base = load_recommended_risk_config(
        config,
        allow_fallback=bool(config.get("flow_matching", {}).get("recommended_config", {}).get("allow_fallback", False)),
    )
    selected_hidden = [int(x) for x in config["stage3"]["risk_space"].get("risk_layers", base.recommended_hidden_layers)]
    selected_lora = [int(x) for x in config["stage3"]["lora"].get("train_layers", base.lora_train_layers)]
    return RecommendedRiskConfig(
        recommended_k=base.recommended_k,
        recommended_score_mode=base.recommended_score_mode,
        recommended_hidden_layers=selected_hidden,
        lora_train_layers=selected_lora,
        risk_basis_path=config["stage3"]["risk_space"].get("risk_basis_path") or base.risk_basis_path,
        normalization_config=base.normalization_config,
        source_path=base.source_path,
        recommended_config_path=base.recommended_config_path,
    )


def train_stage3_lora(
    config: Dict[str, Any],
    *,
    max_steps: Optional[int] = None,
    max_train_samples: Optional[int] = None,
    learning_rate: Optional[float] = None,
    output_dir: Optional[str] = None,
    gradient_accumulation_steps: Optional[int] = None,
    debug: bool = False,
    resume_from_adapter: Optional[str] = None,
) -> Dict[str, Any]:
    cfg = _override(
        config,
        max_steps=max_steps,
        max_train_samples=max_train_samples,
        learning_rate=learning_rate,
        output_dir=output_dir,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    cfg = sync_stage3_layers_with_recommendation(cfg)
    if (
        cfg.get("stage3", {}).get("flow_distillation", {}).get("enabled", False)
        and output_dir is None
        and cfg.get("stage3", {}).get("outputs", {}).get("adapter_dir", "").endswith("lora_unlearned/adapter")
    ):
        configured_output = Path(cfg["stage3"]["training"]["output_dir"])
        flow_output = configured_output.parent / "lora_flow_unlearned"
        cfg["stage3"]["training"]["output_dir"] = str(flow_output)
        cfg["stage3"]["outputs"]["adapter_dir"] = str(flow_output / "adapter")
    train_cfg = cfg["stage3"]["training"]
    out_cfg = cfg["stage3"]["outputs"]
    metrics_dir = ensure_dir(resolve_path(cfg, out_cfg["metrics_dir"]))
    figures_dir = ensure_dir(resolve_path(cfg, out_cfg["figures_dir"]))
    adapter_dir = resolve_path(cfg, out_cfg["adapter_dir"])
    set_seed(int(train_cfg.get("seed", 42)))

    triplets = build_stage3_triplets(cfg, max_train_samples=cfg["stage3"]["data"].get("max_train_samples"))
    configured_max_steps = train_cfg.get("max_steps")
    if configured_max_steps is not None and int(configured_max_steps) == 0:
        logger.info("max_steps=0: data construction test complete; no model loaded and no training performed.")
        return {"triplets": len(triplets), "adapter_dir": str(adapter_dir), "trained_steps": 0}

    print("Stage 3 LoRA unlearning started.")
    model, processor = load_base_model_and_processor(cfg)
    if resume_from_adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, resume_from_adapter, is_trainable=True)
    else:
        model = add_lora_adapter(model, cfg)
    if hasattr(model, "config"):
        model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing for Stage 3 training.")
        except Exception as exc:
            logger.warning("Could not enable gradient checkpointing: %s", exc)
    model.train()
    summary = get_trainable_parameter_summary(model)
    print(f"Trainable params: {summary}")
    save_json(summary, metrics_dir / "trainable_params.json")

    device = next((p.device for p in model.parameters() if p.device.type != "meta"), torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    risk_tensors = load_risk_tensors(cfg, device)
    recommended = _resolved_recommended_for_stage3(cfg)
    save_resolved_recommended_config(recommended, metrics_dir / "recommended_config_resolved.json")
    flow_bundle = load_flow_teacher(cfg, device)
    print(f"Risk layers: {sorted(risk_tensors['risk_basis'])}")
    print(f"LoRA layers: {cfg['stage3']['lora'].get('train_layers', [20])}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(train_cfg.get("learning_rate", 2e-5)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    grad_accum = max(1, int(train_cfg.get("gradient_accumulation_steps", 1)))
    max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
    epochs = int(train_cfg.get("num_train_epochs", 1))
    max_steps_total = int(configured_max_steps) if configured_max_steps is not None else epochs * len(triplets)
    log_rows = []
    global_step = 0
    optimizer_steps = 0
    pending_backward_steps = 0
    optimizer.zero_grad(set_to_none=True)
    debug_label_path = metrics_dir / "label_mask_debug.json"

    try:
        for epoch in range(epochs):
            for idx, triplet in enumerate(triplets):
                if global_step >= max_steps_total:
                    break
                losses = compute_triplet_losses(
                    model,
                    processor,
                    triplet,
                    cfg,
                    risk_tensors,
                    debug_label_path=debug_label_path,
                )
                flow_losses = compute_flow_stage3_losses(
                    model,
                    processor,
                    triplet,
                    cfg,
                    risk_tensors,
                    flow_bundle,
                    recommended,
                    global_step=global_step,
                    total_steps=max_steps_total,
                )
                losses["loss_total"] = (
                    losses["loss_total"]
                    + flow_losses["loss_flow_distill"]
                    + flow_losses["loss_flow_identity"]
                )
                if not torch.isfinite(losses["loss_total"]).all():
                    raise FloatingPointError(
                        f"Non-finite Stage 3 loss at step {global_step + 1}: "
                        f"total={float(losses['loss_total'].detach().float().cpu())}, "
                        f"npo={float(losses['loss_npo'].detach().float().cpu())}, "
                        f"po={float(losses['loss_po'].detach().float().cpu())}, "
                        f"flow={float(flow_losses['loss_flow_distill'].detach().float().cpu())}."
                    )
                (losses["loss_total"] / grad_accum).backward()
                pending_backward_steps += 1
                grad_norm_value = None
                optimizer_step_happened = False
                is_last_configured_step = (global_step + 1) >= max_steps_total
                is_last_data_step = epoch == epochs - 1 and idx == len(triplets) - 1
                if pending_backward_steps >= grad_accum or is_last_configured_step or is_last_data_step:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        max_grad_norm,
                        error_if_nonfinite=True,
                    )
                    grad_norm_value = float(grad_norm.detach().cpu())
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_steps += 1
                    pending_backward_steps = 0
                    optimizer_step_happened = True
                row = {
                    "step": global_step + 1,
                    "epoch": epoch,
                    **{k: float(v.detach().float().cpu()) for k, v in losses.items()},
                    **{
                        k: (float(v.detach().float().cpu()) if torch.is_tensor(v) else float(v))
                        for k, v in flow_losses.items()
                    },
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "grad_norm": grad_norm_value,
                    "optimizer_step": optimizer_steps if optimizer_step_happened else None,
                    "pending_backward_steps": pending_backward_steps,
                    "harmful_sample_id": triplet["harmful"].get("sample_id"),
                    "safe_sample_id": triplet["safe"].get("sample_id"),
                    "retain_sample_ids": ",".join([r.get("sample_id", "") for r in triplet.get("retains", [])]),
                }
                log_rows.append(row)
                if (global_step + 1) % int(train_cfg.get("logging_steps", 1)) == 0:
                    print(
                        f"Step {global_step + 1} | total={row['loss_total']:.4f} "
                        f"harm_ce={row['loss_safe_ce']:.4f} safe_nb_ce={row['loss_safe_neighbor_ce']:.4f} "
                        f"npo={row['loss_npo']:.4f} npo_ratio={row['npo_log_ratio']:.4f} "
                        f"po={row['loss_po']:.4f} po_harm={row['loss_po_harmful_ce']:.4f} "
                        f"po_retain={row['loss_po_retain_ce']:.4f} "
                        f"align={row['loss_align']:.4f} implicit={row['loss_implicit']:.4f} "
                        f"safe_kl={row['loss_safe_kl']:.4f} retain_kl={row['loss_retain_kl']:.4f} "
                        f"hidden={row['loss_retain_hidden']:.4f} flow={row['loss_flow_distill']:.4f} "
                        f"lambda_flow={row['lambda_flow']:.3f} opt_step={row['optimizer_step']}"
                    )
                if (global_step + 1) % int(train_cfg.get("save_steps", 50)) == 0:
                    save_lora_adapter(model, adapter_dir)
                global_step += 1
            if global_step >= max_steps_total:
                break
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            raise RuntimeError(f"{exc}\n{cuda_oom_help()}") from exc
        raise
    finally:
        if log_rows:
            pd.DataFrame(log_rows).to_csv(metrics_dir / "train_loss_log.csv", index=False, encoding="utf-8-sig")
            plot_loss_curve(metrics_dir / "train_loss_log.csv", figures_dir)
        with (metrics_dir / "stage3_train_config_snapshot.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    save_lora_adapter(model, adapter_dir)
    print("Stage 3 training finished.")
    print(f"Adapter saved to {adapter_dir}")
    return {"trained_steps": global_step, "optimizer_steps": optimizer_steps, "adapter_dir": str(adapter_dir), "trainable": summary}
