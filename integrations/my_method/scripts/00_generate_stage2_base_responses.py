"""Generate Base-model responses consumed by Stage 2 risk evaluation."""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.generate_responses import generate_responses_for_split, write_placeholder_responses_for_split
from src.cli_overrides import add_batch_size_arg, add_model_memory_args, apply_model_memory_override, positive_batch_size
from src.utils import add_dataset_argument, apply_dataset_preset, load_config


def release_model_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Base LLaVA responses for Stage 2 train/val scoring."
    )
    parser.add_argument(
        "--config",
        default="integrations/my_method/configs/safeeraser_llava.yaml",
    )
    add_dataset_argument(parser)
    parser.add_argument("--split", choices=["train", "val", "both"], default="both")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--generation_batch_size", type=int, default=None)
    add_batch_size_arg(parser, help_text="Alias for --generation_batch_size; overrides stage2.generation.generation_batch_size.")
    add_model_memory_args(parser)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate even when a matching generated-response JSONL exists.",
    )
    parser.add_argument(
        "--placeholder_responses",
        action="store_true",
        help=(
            "Do not run base-model generation. Write empty placeholder responses instead; "
            "Stage 2 explicit risk will treat them as safe/R_explicit=0."
        ),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    apply_dataset_preset(config, args.dataset)
    batch_override = args.generation_batch_size if args.generation_batch_size is not None else args.batch_size
    if batch_override is not None:
        config.setdefault("stage2", {}).setdefault("generation", {})["generation_batch_size"] = positive_batch_size(batch_override)
    apply_model_memory_override(config, args, sections=["model"])
    response_source = str(config.get("stage2", {}).get("response_source", "dataset"))
    if response_source != "generate":
        raise ValueError(
            "Stage 2 Base generation requires stage2.response_source='generate', "
            f"got {response_source!r}."
        )
    if args.placeholder_responses:
        config.setdefault("stage2", {}).setdefault("generation", {})["placeholder_responses"] = True

    splits = ("train", "val") if args.split == "both" else (args.split,)
    outputs = {}
    model_bundle = {}
    for split in splits:
        if args.placeholder_responses:
            outputs[split] = write_placeholder_responses_for_split(
                config,
                split,
                max_samples=args.max_samples,
                force_regenerate=args.force,
                model_path_override=args.model_path,
            )
        else:
            outputs[split] = generate_responses_for_split(
                config,
                split,
                max_samples=args.max_samples,
                force_regenerate=args.force,
                model_path_override=args.model_path,
                model_bundle=model_bundle,
            )
    model_bundle.clear()
    release_model_memory()

    label = "placeholder response files written" if args.placeholder_responses else "Base responses generated"
    print(f"Stage 2 {label}:")
    for split, path in outputs.items():
        print(f"  {split}: {path}")


if __name__ == "__main__":
    main()
