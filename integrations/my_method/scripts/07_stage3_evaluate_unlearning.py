import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.stage3_evaluator import evaluate_stage3_unlearning
from src.reuse_existing import reuse_if_exists
from src.utils import add_dataset_argument, apply_dataset_preset, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Stage 3 LoRA unlearning adapter.")
    parser.add_argument("--config", default="integrations/my_method/configs/safeeraser_llava.yaml")
    add_dataset_argument(parser)
    parser.add_argument("--adapter_path", default="integrations/my_method/outputs/stage3/lora_unlearned/adapter")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_per_group", type=int, default=None)
    parser.set_defaults(generate_base_baseline=None)
    parser.add_argument("--generate_base_baseline", dest="generate_base_baseline", action="store_true")
    parser.add_argument("--skip_base_baseline", dest="generate_base_baseline", action="store_false")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--metrics_dir", default=None)
    parser.add_argument("--figures_dir", default=None)
    parser.add_argument("--generations_dir", default=None)
    parser.add_argument(
        "--layer_selection_method",
        choices=["stage1_5_ablation", "stage1_5_recommended", "risk_transport_influence", "module_risk_transport_influence"],
        default=None,
        help="Override stage3.layer_selection.method for evaluation without editing the YAML config.",
    )
    parser.add_argument(
        "--layer_selection_top_n",
        default=None,
        help="Override stage3.layer_selection.top_n for risk_transport_influence. Use an integer or 'auto'.",
    )
    parser.add_argument(
        "--transport_target",
        choices=["safe_neighbor", "safenb", "safe", "retain", "mixed", "mix"],
        default=None,
        help="Override stage3.layer_selection.transport_target.mode during evaluation.",
    )
    parser.add_argument("--transport_target_safe_weight", type=float, default=None)
    parser.add_argument("--transport_target_retain_weight", type=float, default=None)
    parser.add_argument("--reuse_existing", action="store_true", help="Skip evaluation if unlearned summary and scores already exist.")
    parser.add_argument("--force", action="store_true", help="Force reevaluation even when --reuse_existing products exist.")
    args = parser.parse_args()

    config = load_config(args.config)
    apply_dataset_preset(config, args.dataset)
    if args.metrics_dir:
        config.setdefault("stage3", {}).setdefault("outputs", {})["metrics_dir"] = args.metrics_dir
    if args.figures_dir:
        config.setdefault("stage3", {}).setdefault("outputs", {})["figures_dir"] = args.figures_dir
    if args.generations_dir:
        config.setdefault("stage3", {}).setdefault("outputs", {})["generations_dir"] = args.generations_dir
    if args.layer_selection_method:
        config.setdefault("stage3", {}).setdefault("layer_selection", {})["method"] = args.layer_selection_method
    if args.force and config.get("stage3", {}).get("layer_selection", {}).get("method") == "module_risk_transport_influence":
        config.setdefault("stage3", {}).setdefault("layer_selection", {})["reuse_module_selection"] = False
    if args.layer_selection_top_n is not None:
        raw_top_n = str(args.layer_selection_top_n)
        config.setdefault("stage3", {}).setdefault("layer_selection", {})["top_n"] = (
            raw_top_n if raw_top_n.lower() == "auto" else int(raw_top_n)
        )
    if args.transport_target is not None:
        config.setdefault("stage3", {}).setdefault("layer_selection", {}).setdefault("transport_target", {})["mode"] = args.transport_target
    if args.transport_target_safe_weight is not None:
        config.setdefault("stage3", {}).setdefault("layer_selection", {}).setdefault("transport_target", {})[
            "safe_weight"
        ] = float(args.transport_target_safe_weight)
    if args.transport_target_retain_weight is not None:
        config.setdefault("stage3", {}).setdefault("layer_selection", {}).setdefault("transport_target", {})[
            "retain_weight"
        ] = float(args.transport_target_retain_weight)
    metrics_dir = config["stage3"]["outputs"]["metrics_dir"]
    if reuse_if_exists(
        config,
        [
            f"{metrics_dir}/{args.split}_unlearned_risk_scores.csv",
            f"{metrics_dir}/{args.split}_unlearned_summary.json",
            f"{metrics_dir}/{args.split}_before_after_summary.json",
            f"{metrics_dir}/{args.split}_unified_evaluation_summary.json",
        ],
        label=f"Stage 3 evaluation ({args.split})",
        reuse_existing=args.reuse_existing,
        force=args.force,
    ):
        return
    evaluate_stage3_unlearning(
        config,
        adapter_path=args.adapter_path,
        split=args.split,
        max_samples=args.max_samples,
        max_per_group=args.max_per_group,
        generate_base_baseline=args.generate_base_baseline,
        model_path_override=args.model_path,
        max_new_tokens_override=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
