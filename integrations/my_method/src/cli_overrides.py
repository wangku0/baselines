from __future__ import annotations

import argparse
from typing import Any, Dict, Iterable, MutableMapping


def add_model_memory_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--gpu_memory",
        default=None,
        help="Override model max_memory for logical CUDA device 0, for example 23GiB or 75GiB.",
    )
    parser.add_argument(
        "--cpu_memory",
        default="120GiB",
        help="CPU max_memory value used together with --gpu_memory or --a800_75g.",
    )
    parser.add_argument(
        "--a800_75g",
        action="store_true",
        help="Convenience preset: max_memory={0: 75GiB, cpu: --cpu_memory}.",
    )


def resolve_gpu_memory(args: argparse.Namespace) -> str | None:
    if getattr(args, "a800_75g", False):
        return "75GiB"
    value = getattr(args, "gpu_memory", None)
    return str(value) if value is not None else None


def memory_map(args: argparse.Namespace) -> Dict[Any, str] | None:
    gpu_memory = resolve_gpu_memory(args)
    if gpu_memory is None:
        return None
    return {0: gpu_memory, "cpu": str(getattr(args, "cpu_memory", "120GiB"))}


def apply_model_memory_override(config: MutableMapping[str, Any], args: argparse.Namespace, sections: Iterable[str]) -> None:
    override = memory_map(args)
    if override is None:
        return
    for section in sections:
        cursor: MutableMapping[str, Any] = config
        parts = section.split(".")
        for part in parts:
            cursor = cursor.setdefault(part, {})
        cursor["max_memory"] = dict(override)


def add_batch_size_arg(parser: argparse.ArgumentParser, *, help_text: str) -> None:
    parser.add_argument("--batch_size", type=int, default=None, help=help_text)


def positive_batch_size(value: int | None, name: str = "--batch_size") -> int | None:
    if value is None:
        return None
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be >= 1.")
    return value
