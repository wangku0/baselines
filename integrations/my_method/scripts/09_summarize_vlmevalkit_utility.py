#!/usr/bin/env python
"""Summarize VLMEvalKit external utility outputs into a compact CSV/JSON report."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


def _fmt_float(x: Any) -> Optional[float]:
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def _dataset_from_path(path: Path, datasets: Iterable[str]) -> str:
    lower = path.name.lower()
    for dataset in datasets:
        if dataset.lower() in lower:
            return dataset
    parts = re.split(r"[_\-.]", path.stem)
    return parts[-1] if parts else path.stem


def _method_from_path(path: Path, methods: Iterable[str]) -> str:
    lower_parts = [p.lower() for p in path.parts]
    for method in methods:
        if method.lower() in lower_parts or method.lower() in path.stem.lower():
            return method
    # VLMEvalKit usually stores results under a model-name subdirectory.
    return path.parent.name


def _read_table(path: Path) -> Optional[pd.DataFrame]:
    try:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        if path.suffix.lower() in {".xlsx", ".xls"}:
            return pd.read_excel(path)
    except Exception:
        return None
    return None


def _extract_metrics(path: Path, df: pd.DataFrame, method: str, dataset: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if df.empty:
        return rows

    for col in df.columns:
        if str(col).lower().startswith("unnamed"):
            continue
        numeric_values = [_fmt_float(v) for v in df[col].tolist()]
        numeric_values = [v for v in numeric_values if v is not None]
        if not numeric_values:
            continue

        if len(numeric_values) == 1:
            value = numeric_values[0]
            metric = str(col)
        else:
            value = sum(numeric_values) / len(numeric_values)
            metric = f"{col}_mean"
        rows.append(
            {
                "method": method,
                "dataset": dataset,
                "metric": metric,
                "value": value,
                "source_file": str(path),
            }
        )

    # Also capture common one-row "Overall/Score/Accuracy" tables as explicit cells.
    for idx, row in df.iterrows():
        label_bits = []
        for col in df.columns[:2]:
            val = row.get(col)
            if isinstance(val, str) and val.strip():
                label_bits.append(val.strip())
        label = "_".join(label_bits) or f"row_{idx}"
        for col, val in row.items():
            num = _fmt_float(val)
            if num is None:
                continue
            metric = f"{label}_{col}"
            rows.append(
                {
                    "method": method,
                    "dataset": dataset,
                    "metric": metric,
                    "value": num,
                    "source_file": str(path),
                }
            )
    return rows


def summarize(work_dir: Path, methods: List[str], datasets: List[str]) -> Dict[str, Any]:
    files = list(work_dir.rglob("*.csv")) + list(work_dir.rglob("*.xlsx")) + list(work_dir.rglob("*.xls"))
    metric_rows: List[Dict[str, Any]] = []
    for path in files:
        df = _read_table(path)
        if df is None:
            continue
        method = _method_from_path(path, methods)
        dataset = _dataset_from_path(path, datasets)
        metric_rows.extend(_extract_metrics(path, df, method, dataset))

    metrics_df = pd.DataFrame(metric_rows)
    retention_rows: List[Dict[str, Any]] = []
    if not metrics_df.empty and "base" in set(metrics_df["method"]):
        grouped = metrics_df.groupby(["dataset", "metric"])
        for (dataset, metric), group in grouped:
            base_vals = group[group["method"] == "base"]["value"].dropna()
            if base_vals.empty:
                continue
            base_value = float(base_vals.iloc[0])
            for _, row in group[group["method"] != "base"].iterrows():
                if base_value == 0:
                    retention = None
                else:
                    retention = float(row["value"]) / base_value
                retention_rows.append(
                    {
                        "method": row["method"],
                        "dataset": dataset,
                        "metric": metric,
                        "base_value": base_value,
                        "method_value": float(row["value"]),
                        "utility_retention": retention,
                    }
                )

    return {
        "num_result_files": len(files),
        "metrics": metric_rows,
        "retention": retention_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize VLMEvalKit utility evaluation results.")
    parser.add_argument("--work_dir", required=True, help="VLMEvalKit output/work directory.")
    parser.add_argument("--output_dir", default="integrations/my_method/outputs/utility_eval/vlmevalkit")
    parser.add_argument("--method", action="append", default=["base", "ours"], help="Known method name.")
    parser.add_argument("--dataset", action="append", default=["MME", "MMBench_DEV_EN"])
    args = parser.parse_args()

    work_dir = Path(args.work_dir).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    report = summarize(work_dir, args.method, args.dataset)
    metrics = pd.DataFrame(report["metrics"])
    retention = pd.DataFrame(report["retention"])
    metrics_path = out_dir / "utility_metrics_long.csv"
    retention_path = out_dir / "utility_retention.csv"
    report_path = out_dir / "utility_summary.json"
    metrics.to_csv(metrics_path, index=False)
    retention.to_csv(retention_path, index=False)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Utility evaluation summary written.")
    print(f"  metrics: {metrics_path}")
    print(f"  retention: {retention_path}")
    print(f"  json: {report_path}")
    if not retention.empty:
        print("\n[Utility Retention]")
        print(retention.head(30).to_string(index=False))


if __name__ == "__main__":
    main()

