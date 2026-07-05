"""Invoke SafeEraser's unchanged inference and evaluator for an integrated model."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def run(command: list[str], cwd: Path, env: dict[str, str]) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, cwd=cwd, env=env, check=True)


def validate_predictions(path: Path, expected_records: int) -> dict:
    rows = json.loads(path.read_text(encoding="utf-8"))
    harmful = sum(
        int(bool(pair.get(key)))
        for row in rows
        for pair in row.get("unsafe_pairs", [])
        for key in ("model_response1", "model_response2", "model_response3")
    )
    sd = sum(
        int(bool(pair.get("sd_response")))
        for row in rows
        for pair in row.get("unsafe_pairs", [])
    )
    retain = sum(
        int(bool((row.get(key) or {}).get("Prediction")))
        for row in rows
        for key in ("UnharmPair_text1", "UnharmPair_text2", "UnharmPair_image1", "UnharmPair_image2")
    )
    expected = {
        "records": expected_records,
        "harmful_responses": expected_records * 4 * 3,
        "sd_responses": expected_records * 4,
        "retain_responses": expected_records * 4,
    }
    actual = {
        "records": len(rows),
        "harmful_responses": harmful,
        "sd_responses": sd,
        "retain_responses": retain,
    }
    if actual != expected:
        raise RuntimeError(f"SafeEraser prediction contract failed: expected={expected}, actual={actual}")
    print("Prediction contract passed:", actual)
    return actual


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    output_root = repo_root / "integrations/my_method/outputs/unified_eval"
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--eval-file",
        default=str(repo_root / "integrations/my_method/outputs/data/violence_50_eval.json"),
    )
    parser.add_argument("--method-name", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output-dir", type=Path, default=output_root)
    parser.add_argument("--expected-records", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--llama-guard-model-path", default=None)
    parser.add_argument("--llama-guard-cache-dir", default=None)
    parser.add_argument("--skip-inference", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = (args.output_dir / f"{args.method_name}_predictions.json").resolve()
    evaluated = (args.output_dir / f"{args.method_name}_safeeraser_evaluated.json").resolve()
    env = dict(os.environ)
    env.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
    env.setdefault("MPLBACKEND", "Agg")

    if not args.skip_inference:
        run(
            [
                args.python,
                "ckpt_infer.py",
                "--eval_file",
                str(Path(args.eval_file).resolve()),
                "--model_path",
                args.model_path,
                "--output_file",
                str(predictions),
                "--max_new_tokens",
                str(args.max_new_tokens),
            ],
            repo_root,
            env,
        )
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
        str(Path(args.eval_file).resolve()),
        "--n_gpu_layers",
        "-1",
    ]
    if args.llama_guard_model_path:
        eval_command.extend(["--llama_guard_model_path", args.llama_guard_model_path])
    if args.llama_guard_cache_dir:
        eval_command.extend(["--cache_dir", args.llama_guard_cache_dir])
    run(eval_command, repo_root, env)
    summary = evaluated.with_suffix(".summary.json")
    if not summary.is_file():
        raise FileNotFoundError(f"SafeEraser summary was not produced: {summary}")
    print(f"SafeEraser-aligned summary: {summary}")


if __name__ == "__main__":
    main()
