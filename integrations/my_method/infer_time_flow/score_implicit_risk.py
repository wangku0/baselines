from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "integrations/my_method"))

import pandas as pd
import torch

from infer_time_flow.controller import InferenceTimeFlowController
from scripts.score_implicit_risk import (
    _infer_sd_eval_file,
    _load_sd_samples,
    _scoring_baselines,
    clearance_summary,
    compare_scores,
    load_variant,
    release_cuda_cache,
    score_model,
    summarize_by_sample_type,
)
from src.data_loader import load_dataset
from src.model_utils import infer_input_device
from src.risk_space.recommended_config import load_recommended_risk_config
from src.stage3_evaluator import _implicit_for_sample, _load_norm
from src.stage3_lora_utils import load_base_model_and_processor, sync_stage3_layers_with_recommendation
from src.stage3_losses import load_risk_tensors
from src.utils import load_config


def _sample_context(sample: dict) -> tuple[float, int]:
    sample_type = sample.get("sample_type")
    if sample_type in {"harmful_trigger", "sd_response"}:
        return 1.0, 0
    if sample_type == "safe_neighbor":
        return 0.0, 1
    if sample_type == "retain":
        return 0.0, 2
    return 0.0, 0


def score_model_with_context(model, processor, controller, config, samples: list[dict], label: str, sample_types: set[str]) -> pd.DataFrame:
    baselines = _scoring_baselines(samples)
    targets = [sample for sample in samples if sample.get("sample_type") in sample_types]
    lower, upper, _ = _load_norm(config)
    recommended = load_recommended_risk_config(config, allow_fallback=False)
    device = infer_input_device(model)
    risk_tensors = load_risk_tensors(config, device)
    rows = []
    for sample in targets:
        baseline = baselines.get(str(sample["id"]))
        if baseline is None:
            raise RuntimeError(f"Missing implicit baseline for {sample['id']} ({sample.get('sample_type')})")
        explicit_risk, group_id = _sample_context(sample)
        with controller.context(explicit_risk=explicit_risk, group_id=group_id):
            raw, before_clip, norm = _implicit_for_sample(
                model,
                processor,
                sample,
                baseline,
                config,
                risk_tensors,
                lower,
                upper,
                recommended,
            )
        rows.append(
            {
                "method": label,
                "sample_id": sample["id"],
                "sample_type": sample.get("sample_type"),
                "pair_id": sample.get("pair_id"),
                "image_path": sample.get("image_path"),
                "instruction": sample.get("instruction"),
                "baseline_sample_id": baseline.get("id"),
                "baseline_sample_type": baseline.get("sample_type"),
                "safeNb_sample_id": baseline.get("id") if baseline.get("sample_type") == "safe_neighbor" else None,
                "R_implicit_raw": raw,
                "R_implicit_norm_before_clip": before_clip,
                "R_implicit_norm": norm,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Implicit-risk scoring for inference-time Flow intervention.")
    parser.add_argument("--config", default="integrations/my_method/configs/safeeraser_llava.yaml")
    parser.add_argument("--flow-teacher-path", default=None)
    parser.add_argument("--method-name", required=True)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--scope", choices=["harmful", "all"], default="all")
    parser.add_argument("--sd-eval-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("integrations/my_method/infer_time_flow/outputs/unified_eval"))
    parser.add_argument("--strength", type=float, default=0.25)
    parser.add_argument("--risk-gate-threshold", type=float, default=0.0)
    parser.add_argument("--max-delta-norm-ratio", type=float, default=0.20)
    parser.add_argument("--no-prefill-intervention", action="store_true")
    parser.add_argument("--no-decode-intervention", action="store_true")
    args = parser.parse_args()

    config = sync_stage3_layers_with_recommendation(load_config(args.config))
    samples = load_dataset(config, split=args.split)
    if args.scope == "all":
        sd_eval_file = args.sd_eval_file or _infer_sd_eval_file(config, args.split)
        samples.extend(_load_sd_samples(config, args.split, sd_eval_file, samples))
        sample_types = {"harmful_trigger", "sd_response", "safe_neighbor", "retain"}
    else:
        sample_types = {"harmful_trigger"}
    expected_scored = sum(sample.get("sample_type") in sample_types for sample in samples)
    if expected_scored == 0:
        raise ValueError("No samples to score.")

    base_model, base_processor, base_lora_stats = load_variant(config, None, None, 32, 256)
    base = score_model(base_model, base_processor, config, samples, "base", sample_types)
    del base_model, base_processor
    release_cuda_cache()

    after_model, after_processor = load_base_model_and_processor(config)
    controller = InferenceTimeFlowController(
        after_model,
        config_path=args.config,
        flow_teacher_path=args.flow_teacher_path,
        strength=args.strength,
        risk_gate_threshold=args.risk_gate_threshold,
        max_delta_norm_ratio=args.max_delta_norm_ratio,
        intervene_on_prefill=not args.no_prefill_intervention,
        intervene_on_decode=not args.no_decode_intervention,
    )
    with controller.enabled():
        after = score_model_with_context(after_model, after_processor, controller, config, samples, args.method_name, sample_types)
    after_stats = {
        "intervention": "infer_time_flow",
        "flow_teacher_path": str(controller.flow_path),
        **controller.stats.to_dict(),
    }
    del after_model, after_processor
    release_cuda_cache()

    merge_keys = [
        "sample_id",
        "sample_type",
        "pair_id",
        "image_path",
        "instruction",
        "baseline_sample_id",
        "baseline_sample_type",
    ]
    merged = base.merge(after, on=merge_keys, suffixes=("_base", "_after"))
    if len(merged) != expected_scored:
        raise RuntimeError(f"Implicit score merge mismatch: expected={expected_scored}, merged={len(merged)}")

    base_by_type = summarize_by_sample_type(base)
    after_by_type = summarize_by_sample_type(after)
    base_summary = base_by_type["harmful_trigger"]
    after_summary = after_by_type["harmful_trigger"]
    before_mean = base_summary["mean"]
    after_mean = after_summary["mean"]
    summary = {
        "definition": "prompt-level paired harmful minus safeNb risk-subspace activation",
        "method": args.method_name,
        "split": args.split,
        "scope": args.scope,
        "intervention": "infer_time_flow",
        "scored_sample_types": sorted(sample_types),
        "base": base_summary,
        "after": after_summary,
        "by_sample_type": {
            "base": base_by_type,
            "after": after_by_type,
            "clearance": clearance_summary(base_by_type, after_by_type),
        },
        "score_delta_diagnostics": compare_scores(merged),
        "lora_diagnostics": {
            "base": base_lora_stats,
            "after": after_stats,
        },
        "implicit_clearance_absolute": (
            float(before_mean - after_mean)
            if before_mean is not None and after_mean is not None
            else None
        ),
        "implicit_clearance_relative": (
            float(1.0 - after_mean / before_mean)
            if before_mean not in (None, 0.0) and after_mean is not None
            else None
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output_dir / f"{args.method_name}_implicit_scores.csv", index=False)
    summary_path = args.output_dir / f"{args.method_name}_implicit_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
