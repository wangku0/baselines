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
    parser = argparse.ArgumentParser(description="Run inference-time Flow intervention with my_method/SafeEraser-aligned metrics.")
    parser.add_argument("--model-path", required=True, help="Base LLaVA model path/repo id.")
    parser.add_argument("--checkpoint-path", "--checkpoint_path", default=None, help="Optional SafeEraser PO LoRA checkpoint.pt to merge before Flow intervention.")
    parser.add_argument("--config", default="integrations/my_method/configs/safeeraser_llava.yaml")
    parser.add_argument("--flow-teacher-path", default=None)
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--paired-eval-file", type=Path, default=None)
    parser.add_argument("--method-name", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output-dir", type=Path, default=Path("integrations/my_method/infer_time_flow/outputs/unified_eval"))
    parser.add_argument("--expected-records", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--strength", type=float, default=0.25)
    parser.add_argument("--risk-gate-threshold", type=float, default=0.0)
    parser.add_argument("--risk-gate-mode", choices=["fused", "implicit"], default="fused")
    parser.add_argument("--max-delta-norm-ratio", type=float, default=0.20)
    parser.add_argument(
        "--numerical-fallback-ratios",
        default="",
        help="Comma-separated max-delta ratios retried after numerical generation errors.",
    )
    parser.add_argument("--risk-trace-max-records", type=int, default=200000)
    parser.add_argument("--no-prefill-intervention", action="store_true")
    parser.add_argument("--no-decode-intervention", action="store_true")
    parser.add_argument("--skip-inference", action="store_true")
    parser.add_argument("--skip-implicit", action="store_true")
    parser.add_argument("--skip-combine", action="store_true")
    parser.add_argument("--llama-guard-model-path", default=None)
    parser.add_argument("--llama-guard-cache-dir", default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
    env.setdefault("MPLBACKEND", "Agg")

    eval_input = enriched_eval_file(Path(args.eval_file).resolve(), args.paired_eval_file, args.output_dir, args.method_name)
    predictions = (args.output_dir / f"{args.method_name}_predictions.json").resolve()
    evaluated = (args.output_dir / f"{args.method_name}_safeeraser_evaluated.json").resolve()

    if not args.skip_inference:
        command = [
            args.python,
            "integrations/my_method/infer_time_flow/infer_safeeraser.py",
            "--eval_file",
            str(eval_input.resolve()),
            "--model_path",
            args.model_path,
            "--output_file",
            str(predictions),
            "--config",
            args.config,
            "--max_new_tokens",
            str(args.max_new_tokens),
            "--strength",
            str(args.strength),
            "--risk_gate_threshold",
            str(args.risk_gate_threshold),
            "--risk_gate_mode",
            args.risk_gate_mode,
            "--max_delta_norm_ratio",
            str(args.max_delta_norm_ratio),
            "--numerical_fallback_ratios",
            args.numerical_fallback_ratios,
            "--risk_trace_max_records",
            str(args.risk_trace_max_records),
        ]
        if args.checkpoint_path:
            command.extend(["--checkpoint_path", args.checkpoint_path])
        if args.flow_teacher_path:
            command.extend(["--flow_teacher_path", args.flow_teacher_path])
        if args.no_prefill_intervention:
            command.append("--no_prefill_intervention")
        if args.no_decode_intervention:
            command.append("--no_decode_intervention")
        run(command, REPO_ROOT, env)

    if not predictions.is_file():
        raise FileNotFoundError(f"Predictions do not exist: {predictions}")
    validate_predictions(predictions, args.expected_records)

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
    safeeraser_summary = evaluated.with_suffix(".summary.json")
    if not safeeraser_summary.is_file():
        raise FileNotFoundError(f"SafeEraser summary was not produced: {safeeraser_summary}")

    implicit_summary = args.output_dir / f"{args.method_name}_implicit_summary.json"
    if not args.skip_implicit:
        implicit_command = [
            args.python,
            "integrations/my_method/infer_time_flow/score_implicit_risk.py",
            "--config",
            args.config,
            "--method-name",
            args.method_name,
            "--split",
            "val",
            "--output-dir",
            str(args.output_dir),
            "--strength",
            str(args.strength),
            "--risk-gate-threshold",
            str(args.risk_gate_threshold),
            "--risk-gate-mode",
            args.risk_gate_mode,
            "--max-delta-norm-ratio",
            str(args.max_delta_norm_ratio),
            "--risk-trace-max-records",
            str(args.risk_trace_max_records),
        ]
        if args.checkpoint_path:
            implicit_command.extend(["--checkpoint-path", args.checkpoint_path])
        if args.flow_teacher_path:
            implicit_command.extend(["--flow-teacher-path", args.flow_teacher_path])
        if args.no_prefill_intervention:
            implicit_command.append("--no-prefill-intervention")
        if args.no_decode_intervention:
            implicit_command.append("--no-decode-intervention")
        run(implicit_command, REPO_ROOT, env)

    if not args.skip_combine and implicit_summary.is_file():
        final_report = args.output_dir / f"{args.method_name}_final_report.json"
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
        print(f"Final report: {final_report}")

    print(json.dumps({"predictions": str(predictions), "safeeraser_summary": str(safeeraser_summary), "implicit_summary": str(implicit_summary)}, indent=2))


if __name__ == "__main__":
    main()
