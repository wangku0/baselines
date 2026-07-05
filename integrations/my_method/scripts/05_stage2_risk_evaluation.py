import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.stage2_risk_evaluation import run_stage2_risk_evaluation
from src.reuse_existing import reuse_if_exists
from src.utils import add_dataset_argument, apply_dataset_preset, load_config


def _print_validation_summary(outputs):
    df = outputs.get("val")
    if df is None or df.empty:
        return
    means = df.groupby("sample_type")[["R_explicit", "R_implicit_norm", "R_total"]].mean()
    harmful = means.loc["harmful_trigger"] if "harmful_trigger" in means.index else None
    safe = means.loc["safe_neighbor"] if "safe_neighbor" in means.index else None
    retain = means.loc["retain"] if "retain" in means.index else None
    print("\nValidation summary:")
    if harmful is not None:
        print(f"  harmful R_explicit_mean: {harmful['R_explicit']:.4f}")
        print(f"  harmful R_implicit_norm_mean: {harmful['R_implicit_norm']:.4f}")
        print(f"  harmful R_total_mean: {harmful['R_total']:.4f}")
    if safe is not None:
        print(f"  safe R_explicit_mean: {safe['R_explicit']:.4f}")
        print(f"  safe R_implicit_norm_mean: {safe['R_implicit_norm']:.4f}")
        print(f"  safe R_total_mean: {safe['R_total']:.4f}")
    if retain is not None:
        print(f"  retain R_explicit_mean: {retain['R_explicit']:.4f}")
        print(f"  retain R_implicit_norm_mean: {retain['R_implicit_norm']:.4f}")
        print(f"  retain R_total_mean: {retain['R_total']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Run Stage 2 explicit/implicit risk evaluation.")
    parser.add_argument("--config", default="integrations/my_method/configs/safeeraser_llava.yaml")
    add_dataset_argument(parser)
    parser.add_argument("--split", choices=["train", "val", "both"], default="both")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--skip_generation", action="store_true")
    parser.add_argument("--force_regenerate", action="store_true")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--alpha_explicit", type=float, default=None)
    parser.add_argument("--beta_implicit", type=float, default=None)
    parser.add_argument(
        "--layer_selection_method",
        choices=["stage1_5_ablation", "stage1_5_recommended", "risk_transport_influence"],
        default=None,
        help="Use the same selected risk layers for Stage 2 normalization/evaluation. "
        "Use risk_transport_influence to compute the unified evaluation subspace after Stage 2.4.",
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
        help="Override stage3.layer_selection.transport_target.mode for unified Stage 2 implicit risk.",
    )
    parser.add_argument("--transport_target_safe_weight", type=float, default=None)
    parser.add_argument("--transport_target_retain_weight", type=float, default=None)
    parser.add_argument("--reuse_existing", action="store_true", help="Skip split(s) if Stage 2 risk score CSVs already exist.")
    parser.add_argument("--force", action="store_true", help="Force recomputation even when --reuse_existing products exist.")
    args = parser.parse_args()

    config = load_config(args.config)
    apply_dataset_preset(config, args.dataset)
    if args.layer_selection_method:
        config.setdefault("stage3", {}).setdefault("layer_selection", {})["method"] = args.layer_selection_method
        if args.layer_selection_method == "risk_transport_influence":
            config.setdefault("stage2", {}).setdefault("implicit_risk", {})["use_stage3_selected_layers"] = True
    if args.layer_selection_top_n is not None:
        raw_top_n = str(args.layer_selection_top_n)
        config.setdefault("stage3", {}).setdefault("layer_selection", {})["top_n"] = (
            raw_top_n if raw_top_n.lower() == "auto" else int(raw_top_n)
        )
    if args.transport_target is not None:
        config.setdefault("stage3", {}).setdefault("layer_selection", {}).setdefault("transport_target", {})[
            "mode"
        ] = args.transport_target
    if args.transport_target_safe_weight is not None:
        config.setdefault("stage3", {}).setdefault("layer_selection", {}).setdefault("transport_target", {})[
            "safe_weight"
        ] = float(args.transport_target_safe_weight)
    if args.transport_target_retain_weight is not None:
        config.setdefault("stage3", {}).setdefault("layer_selection", {}).setdefault("transport_target", {})[
            "retain_weight"
        ] = float(args.transport_target_retain_weight)
    splits = ("train", "val") if args.split == "both" else (args.split,)
    if reuse_if_exists(
        config,
        [f"integrations/my_method/outputs/metrics/stage2/{split}_stage2_risk_scores.csv" for split in splits]
        + [f"integrations/my_method/outputs/metrics/stage2/{split}_stage2_summary.json" for split in splits]
        + ["integrations/my_method/outputs/metrics/stage2/implicit_normalization.json"],
        label=f"Stage 2 risk scores ({','.join(splits)})",
        reuse_existing=args.reuse_existing,
        force=args.force,
    ):
        return
    outputs = run_stage2_risk_evaluation(
        config,
        splits=splits,
        max_samples=args.max_samples,
        skip_generation=args.skip_generation,
        force_regenerate=args.force_regenerate or args.force,
        model_path_override=args.model_path,
        alpha_explicit=args.alpha_explicit,
        beta_implicit=args.beta_implicit,
    )
    print("\nStage 2 risk evaluation finished.")
    _print_validation_summary(outputs)
    print("\nNext step:")
    print("  Use outputs/metrics/stage2/train_training_weights.csv")
    print("  and outputs/metrics/stage2/val_stage2_risk_scores.csv")
    print("  for Stage 3 risk-guided unlearning.")


if __name__ == "__main__":
    main()
