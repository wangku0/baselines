import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate_risk_space import evaluate_risk_space
from src.reuse_existing import reuse_if_exists
from src.utils import add_dataset_argument, apply_dataset_preset, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate stage-1 counterfactual risk subspace.")
    parser.add_argument("--config", default="integrations/my_method/configs/safeeraser_llava.yaml")
    add_dataset_argument(parser)
    parser.add_argument("--split", choices=["train", "val"], required=True)
    parser.add_argument("--reuse_existing", action="store_true", help="Skip if the split risk-space evaluation JSON already exists.")
    parser.add_argument("--force", action="store_true", help="Force recomputation even when --reuse_existing products exist.")
    args = parser.parse_args()

    config = load_config(args.config)
    apply_dataset_preset(config, args.dataset)
    if reuse_if_exists(
        config,
        [f"{config.get('outputs', {}).get('metrics_dir', 'outputs/metrics')}/{args.split}_risk_space_eval_raw.json"],
        label=f"Stage 1 risk-space evaluation ({args.split})",
        reuse_existing=args.reuse_existing,
        force=args.force,
    ):
        return
    evaluate_risk_space(config, split=args.split)


if __name__ == "__main__":
    main()
