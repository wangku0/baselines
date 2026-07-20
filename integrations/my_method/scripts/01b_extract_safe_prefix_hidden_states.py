import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.extract_safe_prefix_hidden_states import extract_safe_prefix_hidden_states_for_split
from src.reuse_existing import reuse_if_exists
from src.utils import add_dataset_argument, apply_dataset_preset, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract safe answer prefix hidden states for Flow target training.")
    parser.add_argument("--config", default="integrations/my_method/configs/safeeraser_llava.yaml")
    add_dataset_argument(parser)
    parser.add_argument("--split", default="train", choices=["train", "val", "both"])
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--model_path", default=None, help="Override model.local_path for this extraction run.")
    parser.add_argument(
        "--safe_prefix_tokens",
        type=int,
        default=None,
        help="Number of safe answer tokens to include before taking the final prefix-token hidden state.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override hidden_states.batch_size for this extraction run.",
    )
    parser.add_argument(
        "--all_layers",
        action="store_true",
        help="Override hidden_states.all_layers=true for this extraction run.",
    )
    parser.add_argument("--gpu_memory", default=None, help="Set model.max_memory for CUDA device 0, e.g. 75GiB.")
    parser.add_argument("--a800-75g", action="store_true", help="Shortcut for --gpu_memory 75GiB.")
    parser.add_argument("--reuse_existing", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    apply_dataset_preset(config, args.dataset)
    if args.batch_size is not None:
        config.setdefault("hidden_states", {})["batch_size"] = int(args.batch_size)
    if args.all_layers:
        config.setdefault("hidden_states", {})["all_layers"] = True
    gpu_memory = "75GiB" if args.a800_75g else args.gpu_memory
    if gpu_memory:
        config.setdefault("model", {})["max_memory"] = {0: str(gpu_memory)}
    if args.safe_prefix_tokens is not None:
        config.setdefault("flow_matching", {}).setdefault("target", {})["safe_answer_prefix_tokens"] = int(args.safe_prefix_tokens)

    splits = ["train", "val"] if args.split == "both" else [args.split]
    for split in splits:
        output_dir = config.get("flow_matching", {}).get("target", {}).get("safe_prefix_hidden_states_dir") or config.get("outputs", {}).get("safe_prefix_hidden_states_dir")
        if output_dir:
            out_rel = f"{output_dir}/{split}_safe_prefix_hidden_states.pt"
        else:
            hidden_dir = Path(str(config.get("outputs", {}).get("hidden_states_dir", "integrations/my_method/outputs/hidden_states")))
            out_rel = str(hidden_dir.parent / "safe_prefix_hidden_states" / f"{split}_safe_prefix_hidden_states.pt")
        if reuse_if_exists(
            config,
            [out_rel],
            label=f"Safe answer prefix hidden states ({split})",
            reuse_existing=args.reuse_existing,
            force=args.force,
        ):
            continue
        path = extract_safe_prefix_hidden_states_for_split(
            config,
            split=split,
            max_samples=args.max_samples,
            model_path_override=args.model_path,
            safe_prefix_tokens=args.safe_prefix_tokens,
        )
        print(f"Safe answer prefix hidden states saved to {path}")


if __name__ == "__main__":
    main()
