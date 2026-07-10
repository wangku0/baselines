import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.stage1_5_analysis import run_stage1_5_analysis
from src.reuse_existing import reuse_if_exists
from src.utils import add_dataset_argument, apply_dataset_preset, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 1.5 risk-space calibration without model loading.")
    parser.add_argument("--config", default="integrations/my_method/configs/safeeraser_llava.yaml")
    add_dataset_argument(parser)
    parser.add_argument("--skip_k_sweep", action="store_true")
    parser.add_argument("--skip_layer_ablation", action="store_true")
    parser.add_argument("--k_values", nargs="+", type=int, default=None)
    parser.add_argument(
        "--score_modes",
        nargs="+",
        choices=[
            "raw",
            "centered",
            "raw_positive",
            "centered_positive",
            "raw_signed",
            "centered_signed",
            "paired_delta",
            "paired_delta_positive",
            "paired_delta_signed",
        ],
        default=None,
    )
    parser.add_argument(
        "--risk_target",
        choices=["safe_neighbor", "safe", "safenb", "retain", "mixed", "mix"],
        default=None,
        help="Override risk_space.target.mode used when rebuilding k-specific risk bases.",
    )
    parser.add_argument("--risk_target_safe_weight", type=float, default=None)
    parser.add_argument("--risk_target_retain_weight", type=float, default=None)
    parser.add_argument("--reuse_existing", action="store_true", help="Skip if Stage 1.5 recommended config/layers already exist.")
    parser.add_argument("--force", action="store_true", help="Force recomputation even when --reuse_existing products exist.")
    args = parser.parse_args()

    config = load_config(args.config)
    apply_dataset_preset(config, args.dataset)
    if args.risk_target is not None:
        config.setdefault("risk_space", {}).setdefault("target", {})["mode"] = args.risk_target
    if args.risk_target_safe_weight is not None:
        config.setdefault("risk_space", {}).setdefault("target", {})["safe_weight"] = args.risk_target_safe_weight
    if args.risk_target_retain_weight is not None:
        config.setdefault("risk_space", {}).setdefault("target", {})["retain_weight"] = args.risk_target_retain_weight
    if reuse_if_exists(
        config,
        [
            "integrations/my_method/outputs/metrics/stage1_5/recommended_config.json",
            "integrations/my_method/outputs/metrics/stage1_5/recommended_layers.json",
            "integrations/my_method/outputs/metrics/stage1_5/k_sweep_summary.csv",
            "integrations/my_method/outputs/metrics/stage1_5/stage1_5_final_summary.json",
        ],
        label="Stage 1.5 recommended config",
        reuse_existing=args.reuse_existing,
        force=args.force,
    ):
        return
    result = run_stage1_5_analysis(
        config,
        skip_k_sweep=args.skip_k_sweep,
        skip_layer_ablation=args.skip_layer_ablation,
        k_values=args.k_values,
        score_modes=args.score_modes,
    )

    rec = result["recommended_config"]
    layers = result.get("recommended_layers") or {}
    val_metrics = layers.get("val_metrics") or rec.get("val_metrics") or {}
    print("\nStage 1.5 finished.\n")
    print("Recommended configuration:")
    print(f"  k: {layers.get('recommended_k', rec['recommended_k'])}")
    print(f"  score_mode: {layers.get('recommended_score_mode', rec['recommended_score_mode'])}")
    print(f"  layers: {layers.get('recommended_layers', 'not evaluated')}")
    print(f"  val harmful_vs_safe_auc: {val_metrics.get('harmful_vs_safe_auc')}")
    print(f"  val harmful_vs_retain_auc: {val_metrics.get('harmful_vs_retain_auc')}")
    print(f"  val balanced_auc: {val_metrics.get('balanced_auc')}")
    print(f"  val paired_mean_diff: {val_metrics.get('paired_mean_diff')}")
    print("  note: centered score is compared with raw score to check whether safe centering reduces retain inflation.")


if __name__ == "__main__":
    main()
