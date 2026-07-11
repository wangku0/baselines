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
    parser.add_argument("--max-memory-per-gpu", default=None)
    parser.add_argument("--gpu-memory", default=None, help="Alias for --max-memory-per-gpu.")
    parser.add_argument("--a800-75g", action="store_true", help="Convenience preset: --max-memory-per-gpu 75GiB.")
    args = parser.parse_args()
    if args.a800_75g:
        args.max_memory_per_gpu = "75GiB"
    if args.gpu_memory is not None:
        args.max_memory_per_gpu = str(args.gpu_memory)

    if not (args.adapter_path / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Missing adapter_config.json: {args.adapter_path}")
    device_map = "auto" if args.device_map == "auto" else {"": "cpu"}
    max_memory = None
    if args.device_map == "auto" and args.max_memory_per_gpu:
        max_memory = {gpu_index: args.max_memory_per_gpu for gpu_index in range(torch.cuda.device_count())}
    model_kwargs = {
        "torch_dtype": torch.float16,
        "low_cpu_mem_usage": True,
        "device_map": device_map,
    }
    if max_memory is not None:
        model_kwargs["max_memory"] = max_memory
    base = LlavaForConditionalGeneration.from_pretrained(
        args.base_model,
        **model_kwargs,
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
