import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.extract_hidden_states import extract_hidden_states_for_split
from src.cli_overrides import add_batch_size_arg, add_model_memory_args, apply_model_memory_override, positive_batch_size
from src.reuse_existing import reuse_if_exists
from src.utils import add_dataset_argument, apply_dataset_preset, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Qwen-VL hidden states for stage-1 risk-space validation.")
    parser.add_argument("--config", default="integrations/my_method/configs/safeeraser_llava.yaml")
    add_dataset_argument(parser)
    parser.add_argument("--split", choices=["train", "val"], required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--model_path", default=None, help="Override config model.local_path")
    parser.add_argument(
        "--hidden-states-dir",
        "--hidden_states_dir",
        default=None,
        help="Override outputs.hidden_states_dir for this extraction run.",
    )
    parser.add_argument(
        "--safeeraser-checkpoint",
        "--safeeraser_checkpoint",
        default=None,
        help="Optional SafeEraser checkpoint.pt merged before hidden-state extraction.",
    )
    parser.add_argument("--safeeraser-lora-r", type=int, default=32)
    parser.add_argument("--safeeraser-lora-alpha", type=int, default=256)
    parser.add_argument(
        "--all_layers",
        action="store_true",
        help="Extract all transformer hidden layers, excluding hidden_states[0] embedding output.",
    )
    add_batch_size_arg(
        parser,
        help_text="Override hidden_states.batch_size for true batched extraction. Use 2 on 4090, 4-8 on A800 as a starting point.",
    )
    add_model_memory_args(parser)
    parser.add_argument("--reuse_existing", action="store_true", help="Skip extraction if the split hidden-state file already exists.")
    parser.add_argument("--force", action="store_true", help="Force recomputation even when --reuse_existing products exist.")
    args = parser.parse_args()

    config = load_config(args.config)
    apply_dataset_preset(config, args.dataset)
    if args.hidden_states_dir:
        config.setdefault("outputs", {})["hidden_states_dir"] = args.hidden_states_dir
    if args.all_layers:
        config.setdefault("hidden_states", {})["all_layers"] = True
    if args.batch_size is not None:
        config.setdefault("hidden_states", {})["batch_size"] = positive_batch_size(args.batch_size)
    apply_model_memory_override(config, args, sections=["model"])
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
        safeeraser_checkpoint_path=args.safeeraser_checkpoint,
        safeeraser_lora_r=args.safeeraser_lora_r,
        safeeraser_lora_alpha=args.safeeraser_lora_alpha,
    )


if __name__ == "__main__":
    main()
