"""Combine unchanged SafeEraser metrics with the additional implicit-risk report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--safeeraser-summary", type=Path, required=True)
    parser.add_argument("--implicit-summary", type=Path, required=True)
    parser.add_argument("--method-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    safe = load(args.safeeraser_summary)
    implicit = load(args.implicit_summary)
    report = {
        "method": args.method_name,
        "metric_policy": {
            "explicit": "SafeEraser eval_all.py unchanged",
            "implicit": "paired prompt-level risk-subspace activation",
            "implicit_is_additional": True,
            "safeeraser_metrics_are_not_replaced_or_fused": True,
        },
        "safeeraser_metrics": {
            "ASR": safe.get("ASR"),
            "RR": safe.get("RR"),
            "SARR_sd": safe.get("SARR_sd", safe.get("SARR")),
            "SARR_safeNb": safe.get("SARR_safeNb"),
            "ROUGE-L": safe.get("average_rouge_l_fmeasure"),
            "retain_ROUGE-L": safe.get("retain_rouge_l_fmeasure", safe.get("average_rouge_l_fmeasure")),
            "safeNb_ROUGE-L": safe.get("safeNb_rouge_l_fmeasure"),
            "group_metrics": safe.get("group_metrics"),
            "counts": safe.get("counts"),
            "mode": safe.get("mode"),
        },
        "implicit_risk": implicit,
        "source_files": {
            "safeeraser_summary": str(args.safeeraser_summary),
            "implicit_summary": str(args.implicit_summary),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
