import argparse
import json
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.risk_space.recommended_config import RecommendedRiskConfig, load_recommended_risk_config, save_resolved_recommended_config
from src.cli_overrides import add_batch_size_arg, add_model_memory_args, apply_model_memory_override, positive_batch_size
from src.stage3_lora_utils import sync_stage3_layers_with_recommendation
from src.stage3_trainer import train_stage3_lora
from src.reuse_existing import reuse_if_exists
from src.utils import add_dataset_argument, apply_dataset_preset, ensure_dir, load_config, resolve_path, save_json


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


def _extract_layers_from_names(names):
    layers = set()
    pattern = re.compile(r"(?:^|\.)(?:layers|h)\.(\d+)\.")
    for name in names or []:
        match = pattern.search(str(name))
        if match:
            layers.add(int(match.group(1)))
    return sorted(layers)


def _materialize_stage3_adapter_metadata(config, *, output_dir=None):
    """Use an existing adapter and regenerate lightweight Stage 3 metadata."""
    cfg = sync_stage3_layers_with_recommendation(config)
    if output_dir is not None:
        cfg["stage3"]["training"]["output_dir"] = output_dir
        cfg["stage3"]["outputs"]["adapter_dir"] = str(Path(output_dir) / "adapter")
    metrics_dir = ensure_dir(resolve_path(cfg, cfg["stage3"]["outputs"]["metrics_dir"]))
    adapter_dir = resolve_path(cfg, cfg["stage3"]["outputs"]["adapter_dir"])
    adapter_config = adapter_dir / "adapter_config.json"
    if not adapter_config.exists():
        raise FileNotFoundError(f"Cannot materialize Stage 3 metadata; missing adapter config: {adapter_config}")

    base = load_recommended_risk_config(
        cfg,
        allow_fallback=bool(cfg.get("flow_matching", {}).get("recommended_config", {}).get("allow_fallback", False)),
    )
    selected_hidden = [int(x) for x in cfg["stage3"]["risk_space"].get("risk_layers", base.recommended_hidden_layers)]
    selected_lora = [int(x) for x in cfg["stage3"]["lora"].get("train_layers", base.lora_train_layers)]
    resolved = RecommendedRiskConfig(
        recommended_k=base.recommended_k,
        recommended_score_mode=base.recommended_score_mode,
        recommended_hidden_layers=selected_hidden,
        lora_train_layers=selected_lora,
        risk_basis_path=cfg["stage3"]["risk_space"].get("risk_basis_path") or base.risk_basis_path,
        normalization_config=base.normalization_config,
        source_path=base.source_path,
        recommended_config_path=base.recommended_config_path,
    )
    save_resolved_recommended_config(resolved, metrics_dir / "recommended_config_resolved.json")

    with adapter_config.open("r", encoding="utf-8") as f:
        adapter_data = json.load(f)
    target_modules = adapter_data.get("target_modules") or cfg["stage3"]["lora"].get("target_modules", [])
    actual_layers = _extract_layers_from_names(target_modules) or selected_lora
    actual = {
        "strategy": "existing_adapter_metadata",
        "requested_layers": selected_lora,
        "actual_layers": actual_layers,
        "target_modules": target_modules,
        "adapter_dir": str(adapter_dir),
        "note": "Materialized from existing adapter_config.json; train_loss_log.csv is only available from an actual training run.",
    }
    save_json(actual, metrics_dir / "actual_lora_modules.json")
    print("Stage 3 adapter exists; regenerated resolved config and adapter metadata.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Stage 3 LoRA unlearning, optionally with flow distillation.")
    parser.add_argument("--config", default="integrations/my_method/configs/safeeraser_llava.yaml")
    add_dataset_argument(parser)
    parser.add_argument("--flow_teacher_path", default=None)
    parser.add_argument("--model_path", default=None, help="Override stage3.base_model.model_path (local path or Hugging Face repo id).")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--metrics_dir", default=None)
    parser.add_argument("--figures_dir", default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    add_batch_size_arg(parser, help_text="Override stage3.training.batch_size.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument(
        "--safe_ce_weight",
        type=float,
        default=None,
        help="Override harmful-prompt behavior CE weight (stage3.loss_weights.safe_ce).",
    )
    parser.add_argument(
        "--safe_neighbor_ce_weight",
        type=float,
        default=None,
        help="Override safe-neighbor answer CE weight (stage3.loss_weights.safe_neighbor_ce).",
    )
    parser.add_argument(
        "--safe_response_mode",
        choices=["paired_safe_response", "safenb", "safe_neighbor", "fallback_template", "refusal_template", "retain_template", "retain"],
        default=None,
        help="Override stage3.safe_response.mode for harmful safe_ce targets.",
    )
    parser.add_argument(
        "--align_target",
        choices=["safe_neighbor", "safenb", "safe", "retain", "mixed", "mix"],
        default=None,
        help="Override stage3.align_target.mode. Use retain for Flow Retain ablations.",
    )
    parser.add_argument("--align_target_safe_weight", type=float, default=None)
    parser.add_argument("--align_target_retain_weight", type=float, default=None)
    parser.add_argument("--align_weight", type=float, default=None, help="Override stage3.loss_weights.align.")
    parser.add_argument("--implicit_weight", type=float, default=None, help="Override stage3.loss_weights.implicit.")
    parser.add_argument("--npo_weight", type=float, default=None, help="Override stage3.loss_weights.npo.")
    parser.add_argument("--npo_beta", type=float, default=None, help="Override stage3.npo.beta.")
    parser.add_argument("--po_weight", type=float, default=None, help="Override stage3.loss_weights.po.")
    parser.add_argument("--po_prompt_path", default=None, help="Override stage3.po.prompt_path (SafeEraser prompt.json).")
    parser.add_argument("--safe_kl_weight", type=float, default=None, help="Override stage3.loss_weights.safe_kl.")
    parser.add_argument("--retain_kl_weight", type=float, default=None, help="Override stage3.loss_weights.retain_kl.")
    parser.add_argument(
        "--retain_hidden_weight",
        type=float,
        default=None,
        help="Override stage3.loss_weights.retain_hidden.",
    )
    parser.add_argument(
        "--flow_enabled",
        type=_parse_bool,
        default=None,
        help="Override stage3.flow_distillation.enabled with true/false.",
    )
    parser.add_argument(
        "--flow_identity_weight",
        type=float,
        default=None,
        help="Override stage3.flow_distillation.lambda_identity.",
    )
    parser.add_argument(
        "--layer_selection_method",
        choices=["stage1_5_ablation", "stage1_5_recommended", "risk_transport_influence", "module_risk_transport_influence"],
        default=None,
        help="Override stage3.layer_selection.method without editing the YAML config.",
    )
    parser.add_argument(
        "--transport_target",
        choices=["safe_neighbor", "safenb", "safe", "retain", "mixed", "mix"],
        default=None,
        help="Override stage3.layer_selection.transport_target.mode for risk_transport_influence.",
    )
    parser.add_argument("--transport_target_safe_weight", type=float, default=None)
    parser.add_argument("--transport_target_retain_weight", type=float, default=None)
    parser.add_argument(
        "--layer_selection_top_n",
        default=None,
        help="Override stage3.layer_selection.top_n for risk_transport_influence. Use an integer or 'auto'.",
    )
    parser.add_argument("--debug", action="store_true")
    add_model_memory_args(parser)
    parser.add_argument("--reuse_existing", action="store_true", help="Skip training if adapter_config.json already exists.")
    parser.add_argument("--force", action="store_true", help="Force retraining even when --reuse_existing products exist.")
    args = parser.parse_args()

    config = load_config(args.config)
    apply_dataset_preset(config, args.dataset)
    if args.batch_size is not None:
        config.setdefault("stage3", {}).setdefault("training", {})["batch_size"] = positive_batch_size(args.batch_size)
    apply_model_memory_override(config, args, sections=["stage3.base_model"])
    if args.model_path:
        config.setdefault("stage3", {}).setdefault("base_model", {})["model_path"] = args.model_path
    if args.flow_teacher_path:
        config.setdefault("stage3", {}).setdefault("flow_distillation", {})["teacher_path"] = args.flow_teacher_path
        config["stage3"]["flow_distillation"]["enabled"] = True
    if args.metrics_dir:
        config.setdefault("stage3", {}).setdefault("outputs", {})["metrics_dir"] = args.metrics_dir
    if args.figures_dir:
        config.setdefault("stage3", {}).setdefault("outputs", {})["figures_dir"] = args.figures_dir
    if args.safe_ce_weight is not None:
        config.setdefault("stage3", {}).setdefault("loss_weights", {})["safe_ce"] = float(args.safe_ce_weight)
    if args.safe_neighbor_ce_weight is not None:
        config.setdefault("stage3", {}).setdefault("loss_weights", {})["safe_neighbor_ce"] = float(args.safe_neighbor_ce_weight)
    if args.align_weight is not None:
        config.setdefault("stage3", {}).setdefault("loss_weights", {})["align"] = float(args.align_weight)
    if args.implicit_weight is not None:
        config.setdefault("stage3", {}).setdefault("loss_weights", {})["implicit"] = float(args.implicit_weight)
    if args.npo_weight is not None:
        if float(args.npo_weight) < 0:
            parser.error("--npo_weight must be non-negative.")
        config.setdefault("stage3", {}).setdefault("loss_weights", {})["npo"] = float(args.npo_weight)
    if args.npo_beta is not None:
        if float(args.npo_beta) <= 0:
            parser.error("--npo_beta must be positive.")
        config.setdefault("stage3", {}).setdefault("npo", {})["beta"] = float(args.npo_beta)
    if args.po_weight is not None:
        if float(args.po_weight) < 0:
            parser.error("--po_weight must be non-negative.")
        config.setdefault("stage3", {}).setdefault("loss_weights", {})["po"] = float(args.po_weight)
    if args.po_prompt_path is not None:
        config.setdefault("stage3", {}).setdefault("po", {})["prompt_path"] = str(args.po_prompt_path)
    if args.safe_kl_weight is not None:
        config.setdefault("stage3", {}).setdefault("loss_weights", {})["safe_kl"] = float(args.safe_kl_weight)
    if args.retain_kl_weight is not None:
        config.setdefault("stage3", {}).setdefault("loss_weights", {})["retain_kl"] = float(args.retain_kl_weight)
    if args.retain_hidden_weight is not None:
        config.setdefault("stage3", {}).setdefault("loss_weights", {})["retain_hidden"] = float(args.retain_hidden_weight)
    if args.flow_enabled is not None:
        config.setdefault("stage3", {}).setdefault("flow_distillation", {})["enabled"] = bool(args.flow_enabled)
    if args.flow_identity_weight is not None:
        config.setdefault("stage3", {}).setdefault("flow_distillation", {})["lambda_identity"] = float(args.flow_identity_weight)
    if args.safe_response_mode is not None:
        config.setdefault("stage3", {}).setdefault("safe_response", {})["mode"] = args.safe_response_mode
    if args.align_target is not None:
        config.setdefault("stage3", {}).setdefault("align_target", {})["mode"] = args.align_target
    if args.align_target_safe_weight is not None:
        config.setdefault("stage3", {}).setdefault("align_target", {})["safe_weight"] = float(args.align_target_safe_weight)
    if args.align_target_retain_weight is not None:
        config.setdefault("stage3", {}).setdefault("align_target", {})["retain_weight"] = float(args.align_target_retain_weight)
    if args.layer_selection_method:
        config.setdefault("stage3", {}).setdefault("layer_selection", {})["method"] = args.layer_selection_method
    if args.force and config.get("stage3", {}).get("layer_selection", {}).get("method") == "module_risk_transport_influence":
        config.setdefault("stage3", {}).setdefault("layer_selection", {})["reuse_module_selection"] = False
    if args.transport_target is not None:
        config.setdefault("stage3", {}).setdefault("layer_selection", {}).setdefault("transport_target", {})["mode"] = args.transport_target
    if args.transport_target_safe_weight is not None:
        config.setdefault("stage3", {}).setdefault("layer_selection", {}).setdefault("transport_target", {})["safe_weight"] = float(args.transport_target_safe_weight)
    if args.transport_target_retain_weight is not None:
        config.setdefault("stage3", {}).setdefault("layer_selection", {}).setdefault("transport_target", {})["retain_weight"] = float(args.transport_target_retain_weight)
    if args.layer_selection_top_n is not None:
        raw_top_n = str(args.layer_selection_top_n)
        config.setdefault("stage3", {}).setdefault("layer_selection", {})["top_n"] = (
            raw_top_n if raw_top_n.lower() == "auto" else int(raw_top_n)
        )
    adapter_dir = Path(args.output_dir) / "adapter" if args.output_dir else Path(config["stage3"]["outputs"]["adapter_dir"])
    if reuse_if_exists(
        config,
        [adapter_dir / "adapter_config.json"],
        label="Stage 3 LoRA adapter",
        reuse_existing=args.reuse_existing,
        force=args.force,
    ):
        _materialize_stage3_adapter_metadata(config, output_dir=args.output_dir)
        return
    max_train_samples = args.max_train_samples if args.max_train_samples is not None else args.max_samples
    result = train_stage3_lora(
        config,
        max_steps=args.max_steps,
        max_train_samples=max_train_samples,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        debug=args.debug,
    )
    print(result)


if __name__ == "__main__":
    main()
