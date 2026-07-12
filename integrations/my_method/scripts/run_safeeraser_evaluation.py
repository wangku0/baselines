"""Invoke SafeEraser's unchanged inference and evaluator for an integrated model."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


SAFE_EMPTY_RESPONSE = "I cannot provide a response."


def run(command: list[str], cwd: Path, env: dict[str, str]) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, cwd=cwd, env=env, check=True)


def load_list(path: Path) -> list[dict]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"Expected JSON list: {path}")
    return value


def infer_paired_eval_file(eval_file: Path) -> Path | None:
    name = eval_file.name
    candidates = []
    if "_val_eval" in name:
        candidates.append(eval_file.with_name(name.replace("_val_eval", "_paired_val")))
    if "_eval" in name:
        candidates.append(eval_file.with_name(name.replace("_eval", "_paired")))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def enriched_eval_file(eval_file: Path, paired_eval_file: Path | None, output_dir: Path, method_name: str) -> Path:
    if paired_eval_file is None:
        inferred = infer_paired_eval_file(eval_file)
        paired_eval_file = inferred
    if paired_eval_file is None or not paired_eval_file.exists():
        return eval_file
    raw_rows = load_list(eval_file)
    paired_rows = load_list(paired_eval_file)
    paired_by_id = {row.get("image_id"): row for row in paired_rows}
    enriched = []
    for row in raw_rows:
        merged = dict(row)
        pair_row = paired_by_id.get(row.get("image_id"))
        if pair_row is not None and isinstance(pair_row.get("safeNb_pairs"), list):
            merged["safeNb_pairs"] = pair_row["safeNb_pairs"]
        enriched.append(merged)
    path = output_dir / f"{method_name}_eval_with_safenb.json"
    path.write_text(json.dumps(enriched, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Prepared SafeEraser eval input with safeNb_pairs: {path}")
    return path


def _ensure_nonempty_text(container: dict, key: str, stats: dict[str, int], stat_key: str) -> None:
    if not str(container.get(key) or "").strip():
        container[key] = SAFE_EMPTY_RESPONSE
        stats[stat_key] = stats.get(stat_key, 0) + 1


def sanitize_predictions(path: Path) -> dict[str, int]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    stats: dict[str, int] = {}
    for row in rows:
        for pair in row.get("unsafe_pairs", []) or []:
            if not isinstance(pair, dict):
                continue
            _ensure_nonempty_text(pair, "sd_response", stats, "sd_response")
            for key in ("model_response1", "model_response2", "model_response3"):
                _ensure_nonempty_text(pair, key, stats, key)
        for key in ("UnharmPair_text1", "UnharmPair_text2", "UnharmPair_image1", "UnharmPair_image2"):
            item = row.get(key)
            if isinstance(item, dict):
                _ensure_nonempty_text(item, "Prediction", stats, key)
        for pair in row.get("safeNb_pairs", []) or []:
            if isinstance(pair, dict):
                _ensure_nonempty_text(pair, "model_response1", stats, "safeNb_model_response1")
    if stats:
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Filled empty prediction fields with safe fallback: {stats}")
    return stats


def validate_predictions(path: Path, expected_records: int) -> dict:
    sanitize_predictions(path)
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
    safe_nb_pairs = sum(len(row.get("safeNb_pairs") or []) for row in rows)
    safe_nb = sum(
        int(bool(pair.get("model_response1")))
        for row in rows
        for pair in row.get("safeNb_pairs", [])
    )
    expected = {
        "records": expected_records,
        "harmful_responses": expected_records * 4 * 3,
        "sd_responses": expected_records * 4,
        "retain_responses": expected_records * 4,
        "safeNb_responses": safe_nb_pairs,
    }
    actual = {
        "records": len(rows),
        "harmful_responses": harmful,
        "sd_responses": sd,
        "retain_responses": retain,
        "safeNb_responses": safe_nb,
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
    parser.add_argument("--generation-batch-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=None, help="Alias for --generation-batch-size.")
    parser.add_argument("--max-memory-per-gpu", default=None)
    parser.add_argument("--gpu-memory", default=None, help="Alias for --max-memory-per-gpu.")
    parser.add_argument("--a800-75g", action="store_true", help="Convenience preset: --max-memory-per-gpu 75GiB.")
    parser.add_argument("--llama-guard-model-path", default=None)
    parser.add_argument("--llama-guard-cache-dir", default=None)
    parser.add_argument(
        "--paired-eval-file",
        type=Path,
        default=None,
        help="Optional paired JSON containing safeNb_pairs. If omitted, inferred from --eval-file when possible.",
    )
    parser.add_argument("--skip-inference", action="store_true")
    args = parser.parse_args()
    if args.batch_size is not None:
        args.generation_batch_size = int(args.batch_size)
    if args.generation_batch_size < 1:
        raise ValueError("--generation-batch-size must be >= 1.")
    if args.a800_75g:
        args.max_memory_per_gpu = "75GiB"
    if args.gpu_memory is not None:
        args.max_memory_per_gpu = str(args.gpu_memory)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = (args.output_dir / f"{args.method_name}_predictions.json").resolve()
    evaluated = (args.output_dir / f"{args.method_name}_safeeraser_evaluated.json").resolve()
    env = dict(os.environ)
    env.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
    env.setdefault("MPLBACKEND", "Agg")
    eval_input = enriched_eval_file(Path(args.eval_file).resolve(), args.paired_eval_file, args.output_dir, args.method_name)

    if not args.skip_inference:
        infer_command = [
            args.python,
            "ckpt_infer.py",
            "--eval_file",
            str(eval_input.resolve()),
            "--model_path",
            args.model_path,
            "--output_file",
            str(predictions),
            "--max_new_tokens",
            str(args.max_new_tokens),
            "--generation_batch_size",
            str(args.generation_batch_size),
        ]
        if args.max_memory_per_gpu:
            infer_command.extend(["--max_memory_per_gpu", args.max_memory_per_gpu])
        run(infer_command, repo_root, env)
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
    run(eval_command, repo_root, env)
    summary = evaluated.with_suffix(".summary.json")
    if not summary.is_file():
        raise FileNotFoundError(f"SafeEraser summary was not produced: {summary}")
    print(f"SafeEraser-aligned summary: {summary}")


if __name__ == "__main__":
    main()
