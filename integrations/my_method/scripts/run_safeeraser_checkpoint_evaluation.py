"""Run SafeEraser-style evaluation for a SafeEraser checkpoint.pt.

This wrapper is for PO/PO+PD checkpoints produced by forget.py.  It keeps the
same metric path used by my_method/inference-time Flow reports:
  1. enrich eval JSON with paired safeNb prompts,
  2. run ckpt_infer.py with --checkpoint_path,
  3. run eval_all.py for ASR/RR/SARR_sd/SARR_safeNb,
  4. run score_implicit_risk.py for base/after R_imp and implicit clearance,
  5. combine both reports.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "integrations/my_method"))

from scripts.run_safeeraser_evaluation import enriched_eval_file, validate_predictions


def run(command: list[str], cwd: Path, env: dict[str, str]) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, cwd=cwd, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SafeEraser checkpoint.pt with unified PO metrics.")
    parser.add_argument("--checkpoint-path", "--checkpoint_path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--config", default="integrations/my_method/configs/safeeraser_llava_all.yaml")
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--paired-eval-file", type=Path, default=None)
    parser.add_argument("--method-name", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output-dir", type=Path, default=Path("integrations/my_method/outputs_all/unified_eval"))
    parser.add_argument("--expected-records", type=int, default=600)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--generation-batch-size",
        "--generation_batch_size",
        "--batch-size",
        "--batch_size",
        type=int,
        default=None,
        help="Number of sampled responses generated together for the same prompt/context; passed through to ckpt_infer.py.",
    )
    parser.add_argument(
        "--implicit-batch-size",
        "--implicit_batch_size",
        type=int,
        default=1,
        help="Batch size for score_implicit_risk.py hidden-state forward passes.",
    )
    parser.add_argument("--llama-guard-model-path", default=None)
    parser.add_argument("--llama-guard-cache-dir", default=None)
    parser.add_argument("--skip-inference", action="store_true")
    parser.add_argument("--skip-safeeraser", action="store_true")
    parser.add_argument("--skip-implicit", action="store_true")
    parser.add_argument("--skip-combine", action="store_true")
    parser.add_argument("--safeeraser-lora-r", type=int, default=32)
    parser.add_argument("--safeeraser-lora-alpha", type=int, default=256)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env.setdefault("MPLBACKEND", "Agg")

    checkpoint = Path(args.checkpoint_path).expanduser()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"SafeEraser checkpoint does not exist: {checkpoint}")

    eval_input = enriched_eval_file(Path(args.eval_file).resolve(), args.paired_eval_file, args.output_dir, args.method_name)
    predictions = (args.output_dir / f"{args.method_name}_predictions.json").resolve()
    evaluated = (args.output_dir / f"{args.method_name}_safeeraser_evaluated.json").resolve()
    safeeraser_summary = evaluated.with_suffix(".summary.json")
    implicit_summary = args.output_dir / f"{args.method_name}_implicit_summary.json"

    if not args.skip_inference:
        run(
            [
                args.python,
                "ckpt_infer.py",
                "--eval_file",
                str(eval_input.resolve()),
                "--model_path",
                args.model_path,
                "--checkpoint_path",
                str(checkpoint),
                "--loss_type",
                "idk",
                "--output_file",
                str(predictions),
                "--max_new_tokens",
                str(args.max_new_tokens),
                *(
                    ["--generation_batch_size", str(args.generation_batch_size)]
                    if args.generation_batch_size is not None
                    else []
                ),
            ],
            REPO_ROOT,
            env,
        )

    if not predictions.is_file():
        raise FileNotFoundError(f"Predictions do not exist: {predictions}")
    validate_predictions(predictions, args.expected_records)

    if not args.skip_safeeraser:
        eval_command = [
            args.python,
            "eval_all.py",
            "--input_file",
            str(predictions),
            "--output_file_rr",
            str(evaluated),
            "--file_refer",
            str(eval_input.resolve()),
            "--n_gpu_layers",
            "-1",
        ]
        if args.llama_guard_model_path:
            eval_command.extend(["--llama_guard_model_path", args.llama_guard_model_path])
        if args.llama_guard_cache_dir:
            eval_command.extend(["--cache_dir", args.llama_guard_cache_dir])
        run(eval_command, REPO_ROOT, env)

    if not safeeraser_summary.is_file():
        raise FileNotFoundError(f"SafeEraser summary was not produced: {safeeraser_summary}")

    if not args.skip_implicit:
        run(
            [
                args.python,
                "integrations/my_method/scripts/score_implicit_risk.py",
                "--config",
                args.config,
                "--safeeraser-checkpoint",
                str(checkpoint),
                "--method-name",
                args.method_name,
                "--split",
                "val",
                "--scope",
                "all",
                "--sd-eval-file",
                str(Path(args.eval_file).resolve()),
                "--output-dir",
                str(args.output_dir),
                "--safeeraser-lora-r",
                str(args.safeeraser_lora_r),
                "--safeeraser-lora-alpha",
                str(args.safeeraser_lora_alpha),
                "--implicit-batch-size",
                str(args.implicit_batch_size),
            ],
            REPO_ROOT,
            env,
        )

    if not implicit_summary.is_file():
        raise FileNotFoundError(f"Implicit summary was not produced: {implicit_summary}")

    final_report = args.output_dir / f"{args.method_name}_final_report.json"
    if not args.skip_combine:
        run(
            [
                args.python,
                "integrations/my_method/scripts/combine_reports.py",
                "--safeeraser-summary",
                str(safeeraser_summary),
                "--implicit-summary",
                str(implicit_summary),
                "--method-name",
                args.method_name,
                "--output",
                str(final_report),
            ],
            REPO_ROOT,
            env,
        )

    summary = json.loads(safeeraser_summary.read_text(encoding="utf-8"))
    implicit = json.loads(implicit_summary.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "predictions": str(predictions),
                "safeeraser_summary": str(safeeraser_summary),
                "implicit_summary": str(implicit_summary),
                "final_report": str(final_report),
                "ASR": summary.get("ASR"),
                "RR": summary.get("RR"),
                "SARR_sd": summary.get("SARR_sd"),
                "SARR_safeNb": summary.get("SARR_safeNb"),
                "implicit_clearance_relative": implicit.get("implicit_clearance_relative"),
                "implicit_clearance_absolute": implicit.get("implicit_clearance_absolute"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
