import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.extract_hidden_states import extract_hidden_states_for_split
from src.reuse_existing import reuse_if_exists
from src.utils import add_dataset_argument, apply_dataset_preset, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Qwen-VL hidden states for stage-1 risk-space validation.")
    parser.add_argument("--config", default="integrations/my_method/configs/safeeraser_llava.yaml")
    add_dataset_argument(parser)
    parser.add_argument("--split", choices=["train", "val"], required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--model_path", default=None, help="Override config model.local_path")
    parser.add_argument("--reuse_existing", action="store_true", help="Skip extraction if the split hidden-state file already exists.")
    parser.add_argument("--force", action="store_true", help="Force recomputation even when --reuse_existing products exist.")
    args = parser.parse_args()

    config = load_config(args.config)
    apply_dataset_preset(config, args.dataset)
    if reuse_if_exists(
        config,
        [f"{config.get('outputs', {}).get('hidden_states_dir', 'outputs/hidden_states')}/{args.split}_hidden_states.pt"],
        label=f"Stage 1 hidden states ({args.split})",
        reuse_existing=args.reuse_existing,
        force=args.force,
    ):
        return
    extract_hidden_states_for_split(
        config,
        split=args.split,
        max_samples=args.max_samples,
        model_path_override=args.model_path,
    )


if __name__ == "__main__":
    main()
