import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_dataset
from src.utils import (
    add_dataset_argument,
    apply_dataset_preset,
    dataset_label,
    load_config,
    logger,
    read_jsonl,
    resolve_path,
)


def _apply_dataset_file_overrides(config, args):
    dataset_cfg = config.setdefault("dataset", {})
    if args.train_file:
        dataset_cfg["train_file"] = args.train_file
    if args.val_file:
        dataset_cfg["val_file"] = args.val_file
    if args.image_root:
        dataset_cfg["image_root"] = args.image_root
def _stage3_model_config(config, model_path_override):
    cfg = dict(config)
    cfg["model"] = dict(config.get("model", {}))
    base = config.get("stage3", {}).get("base_model", {})
    cfg["model"]["local_path"] = model_path_override or base.get("model_path") or cfg["model"].get("local_path")
    cfg["model"]["torch_dtype"] = base.get("torch_dtype", cfg["model"].get("torch_dtype", "auto"))
    cfg["model"]["device_map"] = base.get("device_map", cfg["model"].get("device_map", "auto"))
    cfg["model"]["trust_remote_code"] = base.get("trust_remote_code", cfg["model"].get("trust_remote_code", True))
    for key in ("local_files_only", "cache_dir", "max_memory", "offload_folder"):
        if key in base:
            cfg["model"][key] = base[key]
    return cfg


def _base_generations_path(config, split, dataset_label):
    template = config.get("stage3", {}).get("evaluation", {}).get("base_generations_path_template")
    if template:
        path = str(template).format(split=split, dataset=dataset_label)
        if "{dataset}" not in str(template):
            path_obj = Path(path)
            path = str(path_obj.with_name(f"{dataset_label}_{path_obj.name}"))
        return resolve_path(config, path)
    out_dir = config.get("stage3", {}).get("outputs", {}).get("generations_dir", "integrations/my_method/outputs/generations/stage3")
    return resolve_path(config, out_dir) / f"{dataset_label}_{split}_base_generations.jsonl"


def _stratified_limit(samples, max_per_group):
    if max_per_group is None:
        return samples
    buckets = {"harmful_trigger": [], "safe_neighbor": [], "retain": []}
    for sample in samples:
        stype = sample.get("sample_type")
        if stype in buckets and len(buckets[stype]) < int(max_per_group):
            buckets[stype].append(sample)
    limited = []
    for stype in ["harmful_trigger", "safe_neighbor", "retain"]:
        limited.extend(buckets[stype])
    return limited


def _load_matching_base_generations(path, samples):
    if not path.exists():
        return None
    rows = read_jsonl(path)
    by_id = {str(row.get("sample_id")): row for row in rows}
    ordered = []
    for sample in samples:
        row = by_id.get(str(sample["id"]))
        if (
            row is None
            or row.get("generated_response") in (None, "")
            or str(row.get("instruction") or "") != str(sample.get("instruction") or "")
            or str(row.get("image_path") or "") != str(sample.get("image_path") or "")
            or str(row.get("category") or "") != str(sample.get("category") or "")
        ):
            return None
        ordered.append(row)
    return ordered


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate and persist original/base model responses for Stage 3 before evaluation."
    )
    parser.add_argument("--config", default="integrations/my_method/configs/safeeraser_llava.yaml")
    add_dataset_argument(parser)
    parser.add_argument("--train_file", default=None, help="Override dataset.train_file.")
    parser.add_argument("--val_file", default=None, help="Override dataset.val_file.")
    parser.add_argument("--image_root", default=None, help="Override dataset.image_root.")
    parser.add_argument("--split", choices=["train", "val", "both"], default="val")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_per_group", type=int, default=None)
    parser.add_argument("--model_path", default=None, help="Override base model path/repo id. Alias of --base_model_path.")
    parser.add_argument("--base_model_path", default=None, help="Base model path/repo id used to generate before responses.")
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--generation_batch_size", type=int, default=None)
    parser.add_argument("--generations_path", default=None)
    parser.add_argument("--force", action="store_true", help="Regenerate even if a matching before file exists.")
    args = parser.parse_args()

    config = load_config(args.config)
    apply_dataset_preset(config, args.dataset)
    _apply_dataset_file_overrides(config, args)
    if args.max_new_tokens is not None:
        config.setdefault("stage3", {}).setdefault("evaluation", {})["max_new_tokens"] = int(args.max_new_tokens)
    if args.generation_batch_size is not None:
        config.setdefault("stage3", {}).setdefault("evaluation", {})["generation_batch_size"] = int(args.generation_batch_size)

    resolved_dataset_label = dataset_label(config)
    logger.info(
        "Using dataset=%s train_file=%s val_file=%s",
        resolved_dataset_label,
        config.get("dataset", {}).get("train_file"),
        config.get("dataset", {}).get("val_file"),
    )
    splits = ["train", "val"] if args.split == "both" else [args.split]
    model = processor = None
    for split in splits:
        samples = load_dataset(config, split=split, max_samples=None if args.max_per_group is not None else args.max_samples)
        samples = _stratified_limit(samples, args.max_per_group)
        if args.max_samples is not None and args.max_per_group is None:
            samples = samples[: int(args.max_samples)]

        out_path = (
            resolve_path(config, args.generations_path.format(split=split, dataset=resolved_dataset_label))
            if args.generations_path
            else _base_generations_path(config, split, resolved_dataset_label)
        )
        if not args.force and _load_matching_base_generations(out_path, samples) is not None:
            logger.info("Before responses already exist for split=%s: %s", split, out_path)
            continue

        if model is None or processor is None:
            from src.model_utils import load_model_and_processor

            base_model_path = args.base_model_path or args.model_path
            model_cfg = _stage3_model_config(config, base_model_path)
            model, processor = load_model_and_processor(model_cfg)
        model_path = (
            args.base_model_path
            or args.model_path
            or config.get("stage3", {}).get("base_model", {}).get("model_path")
            or config["model"]["local_path"]
        )
        from src.stage3_generation import generate_for_samples

        generate_for_samples(model, processor, samples, config, split, out_path, "generated_base", model_path=model_path)
        logger.info("Saved persisted before responses for split=%s to %s", split, out_path)


if __name__ == "__main__":
    main()
