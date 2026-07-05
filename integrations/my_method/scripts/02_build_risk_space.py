import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.build_risk_space import build_risk_space
from src.reuse_existing import reuse_if_exists
from src.utils import add_dataset_argument, apply_dataset_preset, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Build counterfactual risk subspace with SVD.")
    parser.add_argument("--config", default="integrations/my_method/configs/safeeraser_llava.yaml")
    add_dataset_argument(parser)
    parser.add_argument(
        "--risk_target",
        choices=["safe_neighbor", "safe", "safenb", "retain", "mixed", "mix"],
        default=None,
        help="Override risk_space.target.mode for this run.",
    )
    parser.add_argument("--risk_target_safe_weight", type=float, default=None)
    parser.add_argument("--risk_target_retain_weight", type=float, default=None)
    parser.add_argument("--risk_output_dir", default=None, help="Optional output directory for risk_basis.pt.")
    parser.add_argument("--metrics_dir", default=None, help="Optional directory for svd_stats.json.")
    parser.add_argument("--reuse_existing", action="store_true", help="Skip if risk_basis.pt already exists.")
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
    risk_dir = args.risk_output_dir or config.get("outputs", {}).get("risk_space_dir", "integrations/my_method/outputs/risk_space")
    if reuse_if_exists(
        config,
        [f"{risk_dir}/risk_basis.pt"],
        label="Stage 1 risk basis",
        reuse_existing=args.reuse_existing,
        force=args.force,
    ):
        return
    build_risk_space(config, output_dir=args.risk_output_dir, metrics_dir=args.metrics_dir)


if __name__ == "__main__":
    main()
