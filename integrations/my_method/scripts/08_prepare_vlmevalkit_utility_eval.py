#!/usr/bin/env python
"""Prepare VLMEvalKit configs for external utility evaluation.

The script writes a VLMEvalKit JSON config for MME/MMBench style evaluation and
optionally installs a tiny Qwen2-VL wrapper into a local VLMEvalKit checkout so
PEFT adapters can be evaluated with the same base model.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import ssl
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utility_eval.vlmevalkit_wrapper import wrapper_module_text


DEFAULT_DATASETS = ["MME", "MMBench_DEV_EN"]
SPECIFICITY_DATASET_FLAGS = {
    "GQA": "GQA",
    # User-facing alias. VLMEvalKit commonly spells this benchmark as VizWiz.
    "VISWIZ": "VizWiz",
    "VIZWIZ": "VizWiz",
    # ScienceQA image split in VLMEvalKit.
    "SQA": "ScienceQA_VAL",
    # VQA-v2 validation split in VLMEvalKit.
    "VQA": "VQAv2_VAL",
    "POPE": "POPE",
    "MMVET": "MMVet",
    "MM_VET": "MMVet",
    "MMB_EN": "MMBench_DEV_EN",
    "MMB_CN": "MMBench_DEV_CN",
}
DATASET_SOURCE_URLS = {
    "MME": "https://opencompass.openxlab.space/utils/VLMEval/MME.tsv",
    "MMBench_DEV_EN": "https://opencompass.openxlab.space/utils/benchmarks/MMBench/MMBench_DEV_EN.tsv",
}


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _project_path(path: Optional[str], project_root: Path) -> Optional[str]:
    if path in (None, "", "none", "None", "null", "NULL"):
        return None
    p = Path(str(path)).expanduser()
    if not p.is_absolute():
        p = project_root / p
    return str(p.resolve())


def _parse_model_spec(specs: Iterable[str]) -> List[Tuple[str, Optional[str]]]:
    parsed: List[Tuple[str, Optional[str]]] = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"--model_spec must be NAME=ADAPTER_PATH_OR_NONE, got: {spec}")
        name, adapter = spec.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Empty model name in --model_spec: {spec}")
        parsed.append((name, adapter.strip() or None))
    return parsed


def _selected_specificity_datasets(args: argparse.Namespace) -> List[str]:
    selected: List[str] = []
    for attr, dataset in (
        ("GQA", "GQA"),
        ("GAQ", "GQA"),
        ("VisWiz", "VizWiz"),
        ("VizWiz", "VizWiz"),
        ("SQA", "ScienceQA_VAL"),
        ("VQA", "VQAv2_VAL"),
        ("POPE", "POPE"),
        ("MMVet", "MMVet"),
        ("MM_Vet", "MMVet"),
        ("MMB_EN", "MMBench_DEV_EN"),
        ("MMB_CN", "MMBench_DEV_CN"),
    ):
        if getattr(args, attr, False):
            selected.append(dataset)
    if getattr(args, "all_specificity", False):
        selected.extend(SPECIFICITY_DATASET_FLAGS.values())
    deduped: List[str] = []
    for dataset in selected:
        if dataset not in deduped:
            deduped.append(dataset)
    return deduped


def _dataset_config(dataset: str) -> Dict[str, str]:
    # VLMEvalKit config mode needs the dataset class for image datasets.
    if dataset.startswith("MMBench"):
        return {"class": "ImageMCQDataset", "dataset": dataset}
    if dataset.startswith("MME") or dataset.startswith("YORN"):
        return {"class": "ImageYORNDataset", "dataset": dataset}
    return {"dataset": dataset}


def _lmu_data_root() -> Path:
    env_root = os.environ.get("LMUData")
    if env_root and Path(env_root).expanduser().exists():
        return Path(env_root).expanduser().resolve()
    root = Path.home() / "LMUData"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _mini_dataset_name(dataset: str) -> str:
    if dataset == "MME":
        # Avoid the substring "MME" in the mini dataset name. VLMEvalKit uses
        # that substring to trigger the official MME rating, which assumes the
        # full paired/category structure and is not valid for a tiny smoke set.
        return "YORN_VLMEVAL_MINI"
    return f"{dataset}_MINI"


def _read_tsv_with_ssl_fallback(dataset: str, source: str, data_root: Path) -> pd.DataFrame:
    cache_path = data_root / f"{dataset}.tsv"
    if cache_path.exists():
        return pd.read_csv(cache_path, sep="\t")

    try:
        frame = pd.read_csv(source, sep="\t")
        frame.to_csv(cache_path, sep="\t", index=False)
        return frame
    except Exception as first_error:
        if "certificate" not in str(first_error).lower():
            raise
        print(f"SSL verification failed while downloading {dataset}; retrying without certificate verification.")
        context = ssl._create_unverified_context()
        with urllib.request.urlopen(source, context=context, timeout=120) as response:
            content = response.read()
        cache_path.write_bytes(content)
        return pd.read_csv(cache_path, sep="\t")


def _materialize_image_references(frame: pd.DataFrame, source_frame: pd.DataFrame) -> pd.DataFrame:
    """Replace compact image references with the referenced base64 payload.

    VLMEvalKit TSV files sometimes store `image=<another index>` to avoid
    duplicating large base64 strings. A random mini subset can drop the referenced
    row, so we materialize those references before saving the mini TSV.
    """
    if "image" not in frame.columns or "image" not in source_frame.columns or "index" not in source_frame.columns:
        return frame

    image_map = {
        str(idx): "" if pd.isna(image) else str(image)
        for idx, image in zip(source_frame["index"], source_frame["image"])
    }

    def resolve(value):
        if pd.isna(value):
            return value
        text = str(value)
        seen = set()
        while 0 < len(text) <= 64 and text in image_map and text not in seen:
            seen.add(text)
            next_text = image_map.get(text, "")
            if not next_text:
                break
            text = next_text
        return text

    frame = frame.copy()
    frame["image"] = [resolve(value) for value in frame["image"]]
    return frame


def _write_mini_tsv(dataset: str, n: int, data_root: Path, seed: int) -> str:
    if dataset.endswith("_MINI"):
        return dataset
    if dataset not in DATASET_SOURCE_URLS:
        raise ValueError(f"No built-in mini dataset source URL for {dataset!r}.")
    source = DATASET_SOURCE_URLS[dataset]
    mini_name = _mini_dataset_name(dataset)
    target = data_root / f"{mini_name}.tsv"
    frame = _read_tsv_with_ssl_fallback(dataset, source, data_root)
    source_frame = frame
    if n > 0 and len(frame) > n:
        if "category" in frame.columns:
            parts = []
            per_group = max(1, n // max(1, frame["category"].nunique()))
            for _, group in frame.groupby("category", sort=False):
                parts.append(group.sample(n=min(len(group), per_group), random_state=seed))
            mini = pd.concat(parts)
            if len(mini) < n:
                rest = frame.drop(index=mini.index, errors="ignore")
                if len(rest):
                    mini = pd.concat(
                        [mini, rest.sample(n=min(len(rest), n - len(mini)), random_state=seed)],
                    )
            frame = mini.head(n).reset_index(drop=True)
        else:
            frame = frame.sample(n=n, random_state=seed).sort_index().reset_index(drop=True)
    frame = _materialize_image_references(frame, source_frame)
    frame.to_csv(target, sep="\t", index=False)
    print(f"Mini dataset written: {target} ({len(frame)} samples from {dataset})")
    return mini_name


def _install_wrapper(vlmevalkit_path: Path, force: bool = False) -> Path:
    vlm_dir = vlmevalkit_path / "vlmeval" / "vlm"
    init_path = vlm_dir / "__init__.py"
    if not init_path.exists():
        raise FileNotFoundError(
            f"Cannot find VLMEvalKit vlmeval/vlm/__init__.py under: {vlmevalkit_path}"
        )

    wrapper_path = vlm_dir / "safenav_qwen2vl.py"
    if wrapper_path.exists() and not force:
        current = wrapper_path.read_text(encoding="utf-8")
        if "class SafeNavQwen2VLChat" not in current:
            raise RuntimeError(f"Refusing to overwrite unexpected file: {wrapper_path}")
    wrapper_path.write_text(wrapper_module_text(), encoding="utf-8")

    init_text = init_path.read_text(encoding="utf-8")
    import_line = "from .safenav_qwen2vl import SafeNavQwen2VLChat"
    if import_line not in init_text:
        backup = init_path.with_suffix(".py.safenav_bak")
        if not backup.exists():
            shutil.copy2(init_path, backup)
        init_path.write_text(init_text.rstrip() + "\n" + import_line + "\n", encoding="utf-8")
    return wrapper_path


def build_config(
    *,
    project_root: Path,
    config: Dict[str, Any],
    model_specs: List[Tuple[str, Optional[str]]],
    datasets: List[str],
    max_new_tokens: Optional[int],
    min_pixels: Optional[int],
    max_pixels: Optional[int],
    attn_implementation: Optional[str],
    device_map: Optional[str],
    do_sample: bool,
) -> Dict[str, Any]:
    model_cfg = config.get("model", {})
    stage3_eval = config.get("stage3", {}).get("evaluation", {})
    utility_cfg = config.get("utility_eval", {})
    utility_model_cfg = utility_cfg.get("model", {})

    base_model = (
        utility_model_cfg.get("base_model_path")
        or model_cfg.get("local_path")
        or model_cfg.get("name")
        or config.get("stage3", {}).get("base_model", {}).get("local_path")
        or "Qwen/Qwen2-VL-2B-Instruct"
    )
    final_max_new_tokens = int(
        max_new_tokens
        or utility_model_cfg.get("max_new_tokens")
        or stage3_eval.get("max_new_tokens")
        or 128
    )

    common = {
        "class": "SafeNavQwen2VLChat",
        "model_path": base_model,
        "max_new_tokens": final_max_new_tokens,
        "attn_implementation": attn_implementation or utility_model_cfg.get("attn_implementation", "sdpa"),
        "device_map": device_map or utility_model_cfg.get("device_map", "auto"),
        "print_device_map": True,
        "do_sample": bool(do_sample),
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "use_custom_prompt": True,
    }
    if min_pixels is not None:
        common["min_pixels"] = int(min_pixels)
    elif utility_model_cfg.get("min_pixels") is not None:
        common["min_pixels"] = int(utility_model_cfg["min_pixels"])
    if max_pixels is not None:
        common["max_pixels"] = int(max_pixels)
    elif utility_model_cfg.get("max_pixels") is not None:
        common["max_pixels"] = int(utility_model_cfg["max_pixels"])

    models: Dict[str, Dict[str, Any]] = {}
    for name, adapter in model_specs:
        item = dict(common)
        item["adapter_path"] = _project_path(adapter, project_root)
        models[name] = item

    return {
        "model": models,
        "data": {dataset: _dataset_config(dataset) for dataset in datasets},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare VLMEvalKit utility evaluation config.")
    parser.add_argument("--config", default="integrations/my_method/configs/safeeraser_llava.yaml")
    parser.add_argument("--vlmevalkit_path", default=None, help="Local VLMEvalKit checkout path.")
    parser.add_argument("--install_wrapper", action="store_true")
    parser.add_argument("--force_install_wrapper", action="store_true")
    parser.add_argument("--output_dir", default="integrations/my_method/outputs/utility_eval/vlmevalkit")
    parser.add_argument("--output_config", default=None)
    parser.add_argument("--dataset", action="append", default=None, help="Dataset name. Repeatable.")
    parser.add_argument("--GQA", action="store_true", help="Add the GQA benchmark.")
    parser.add_argument("--GAQ", action="store_true", help="Alias for --GQA.")
    parser.add_argument("--VisWiz", "--VISWIZ", action="store_true", help="Add the VizWiz/VisWiz benchmark.")
    parser.add_argument("--VizWiz", "--VIZWIZ", action="store_true", help="Add the VizWiz benchmark.")
    parser.add_argument("--SQA", action="store_true", help="Add the ScienceQA image benchmark.")
    parser.add_argument("--VQA", action="store_true", help="Add the VQAv2 benchmark.")
    parser.add_argument("--POPE", action="store_true", help="Add the POPE benchmark.")
    parser.add_argument("--MMVet", "--MM-Vet", "--MM_VET", action="store_true", help="Add the MMVet benchmark.")
    parser.add_argument("--MM_Vet", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--MMB_EN", "--MMB-en", action="store_true", help="Add MMBench_DEV_EN.")
    parser.add_argument("--MMB_CN", "--MMB-cn", action="store_true", help="Add MMBench_DEV_CN.")
    parser.add_argument(
        "--all-specificity",
        "--all_specificity",
        action="store_true",
        help="Add GQA, VizWiz, ScienceQA, VQAv2, POPE, MMVet, MMBench EN and MMBench CN.",
    )
    parser.add_argument(
        "--model_spec",
        action="append",
        default=None,
        help="Model spec NAME=ADAPTER_PATH_OR_NONE. Repeatable.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--min_pixels", type=int, default=None)
    parser.add_argument("--max_pixels", type=int, default=None)
    parser.add_argument(
        "--attn_implementation",
        default=None,
        choices=["sdpa", "eager", "flash_attention_2"],
        help="Attention backend for Qwen2-VL in VLMEvalKit. Default comes from utility_eval.model.",
    )
    parser.add_argument(
        "--device_map",
        default=None,
        help="Transformers device_map used by Qwen2-VL. Use 'auto' for GPU auto placement.",
    )
    parser.add_argument(
        "--mini_samples",
        type=int,
        default=None,
        help="Create and evaluate local MINI TSV subsets with this many samples per dataset.",
    )
    parser.add_argument("--mini_seed", type=int, default=42)
    parser.add_argument("--do_sample", action="store_true", help="Enable sampling; default is deterministic.")
    args = parser.parse_args()

    project_root = Path.cwd().resolve()
    cfg = _load_yaml(project_root / args.config)

    utility_cfg = cfg.get("utility_eval", {})
    flagged_datasets = _selected_specificity_datasets(args)
    datasets = args.dataset or flagged_datasets or utility_cfg.get("datasets") or DEFAULT_DATASETS
    mini_data_root = None
    if args.mini_samples is not None:
        mini_data_root = _lmu_data_root()
        datasets = [
            _write_mini_tsv(dataset, int(args.mini_samples), mini_data_root, int(args.mini_seed))
            for dataset in datasets
        ]

    if args.model_spec:
        model_specs = _parse_model_spec(args.model_spec)
    else:
        default_adapter = (
            utility_cfg.get("methods", {}).get("ours", {}).get("adapter_path")
            or cfg.get("stage3", {}).get("output_dir", "integrations/my_method/outputs/stage3/lora_unlearned")
        )
        adapter_dir = Path(default_adapter)
        if adapter_dir.name != "adapter":
            adapter_dir = adapter_dir / "adapter"
        model_specs = [("base", None), ("ours", str(adapter_dir))]

    out_dir = _project_path(args.output_dir, project_root)
    assert out_dir is not None
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    out_config = Path(_project_path(args.output_config, project_root) or (out_dir_path / "vlmevalkit_config.json"))

    vlmeval_config = build_config(
        project_root=project_root,
        config=cfg,
        model_specs=model_specs,
        datasets=list(datasets),
        max_new_tokens=args.max_new_tokens,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        attn_implementation=args.attn_implementation,
        device_map=args.device_map,
        do_sample=args.do_sample,
    )
    out_config.parent.mkdir(parents=True, exist_ok=True)
    out_config.write_text(json.dumps(vlmeval_config, ensure_ascii=False, indent=2), encoding="utf-8")

    installed = None
    if args.install_wrapper:
        if not args.vlmevalkit_path:
            raise ValueError("--install_wrapper requires --vlmevalkit_path")
        installed = _install_wrapper(Path(args.vlmevalkit_path).expanduser().resolve(), args.force_install_wrapper)

    metadata = {
        "vlmevalkit_config": str(out_config),
        "datasets": list(datasets),
        "models": list(vlmeval_config["model"].keys()),
        "wrapper_installed_to": str(installed) if installed else None,
        "mini_samples": args.mini_samples,
        "mini_data_root": str(mini_data_root) if mini_data_root else None,
        "run_command": f"python run.py --config {out_config}",
    }
    (out_dir_path / "prepare_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("VLMEvalKit utility config prepared.")
    print(f"  config: {out_config}")
    if installed:
        print(f"  wrapper: {installed}")
    print("\nRun inside the VLMEvalKit checkout:")
    print(f"  python run.py --config {out_config}")


if __name__ == "__main__":
    main()
