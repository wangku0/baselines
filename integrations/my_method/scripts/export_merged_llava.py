"""Merge a my_method PEFT adapter into an FP16 LLaVA model for SafeEraser inference."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoProcessor, LlavaForConditionalGeneration


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("integrations/my_method/outputs/export/merged_llava"),
    )
    parser.add_argument("--device-map", default="cpu", choices=["cpu", "auto"])
    args = parser.parse_args()

    if not (args.adapter_path / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Missing adapter_config.json: {args.adapter_path}")
    device_map = "auto" if args.device_map == "auto" else {"": "cpu"}
    base = LlavaForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map=device_map,
    )
    model = PeftModel.from_pretrained(base, str(args.adapter_path), is_trainable=False)
    merged = model.merge_and_unload(safe_merge=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(args.output_dir, safe_serialization=True, max_shard_size="5GB")
    processor = AutoProcessor.from_pretrained(args.base_model)
    processor.save_pretrained(args.output_dir)
    print(f"Merged LLaVA model saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
