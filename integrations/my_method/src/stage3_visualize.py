from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import pandas as pd

from .utils import ensure_dir, logger


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def plot_loss_curve(log_path: Path, figures_dir: Path) -> None:
    if not log_path.exists():
        return
    df = pd.read_csv(log_path)
    if df.empty or "loss_total" not in df:
        return
    ensure_dir(figures_dir)
    plt.figure(figsize=(8, 5))
    plt.plot(df["step"], df["loss_total"], label="total")
    for col in ["loss_safe_ce", "loss_npo", "loss_po", "loss_align", "loss_implicit", "loss_safe_kl", "loss_retain_kl", "loss_retain_hidden"]:
        if col in df:
            plt.plot(df["step"], df[col], label=col.replace("loss_", ""))
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Stage 3 Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "train_loss_curve.png", dpi=200)
    plt.close()


def plot_before_after(comparison: Dict[str, Any], figures_dir: Path) -> None:
    ensure_dir(figures_dir)
    by_type = comparison.get("by_sample_type", {})
    if not by_type:
        logger.warning("No before/after data for Stage 3 figures.")
        return

    plot_rows = []
    for sample_type in ["harmful_trigger", "safe_neighbor", "retain"]:
        item = by_type.get(sample_type, {})
        before_value = _finite_float(item.get("R_total_before_mean"))
        after_value = _finite_float(item.get("R_total_after_mean"))
        if int(item.get("count", 0) or 0) > 0 and before_value is not None and after_value is not None:
            plot_rows.append((sample_type, before_value, after_value))
        elif int(item.get("count", 0) or 0) > 0:
            logger.warning("Skipping %s in before/after plot because R_total mean is missing or non-finite.", sample_type)
    if plot_rows:
        sample_types = [row[0] for row in plot_rows]
        before = [row[1] for row in plot_rows]
        after = [row[2] for row in plot_rows]
        x = range(len(sample_types))
        plt.figure(figsize=(8, 5))
        plt.bar([i - 0.2 for i in x], before, width=0.4, label="before")
        plt.bar([i + 0.2 for i in x], after, width=0.4, label="after")
        plt.xticks(list(x), sample_types, rotation=20)
        plt.ylabel("R_total mean")
        plt.title("Before/After R_total by Sample Type")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures_dir / "sample_type_before_after_total_risk.png", dpi=200)
        plt.close()

    if "harmful_trigger" in by_type:
        h = by_type["harmful_trigger"]
        harmful_before = _finite_float(h.get("R_total_before_mean"))
        harmful_after = _finite_float(h.get("R_total_after_mean"))
        if harmful_before is not None and harmful_after is not None:
            plt.figure(figsize=(6, 5))
            plt.bar(["before", "after"], [harmful_before, harmful_after])
            plt.ylabel("R_total mean")
            plt.title("Harmful Before/After Total Risk")
            plt.tight_layout()
            plt.savefig(figures_dir / "harmful_before_after_total_risk.png", dpi=200)
            plt.close()

    clearance = comparison.get("clearance", {})
    if clearance:
        keys = [
            "explicit_clearance",
            "implicit_clearance",
            "fusion_total_risk_clearance",
            "balanced_explicit_implicit_clearance",
        ]
        pairs = [
            (label, _finite_float(clearance.get(key)))
            for key, label in zip(keys, ["explicit", "implicit", "fusion total", "balanced"])
        ]
        pairs = [(label, value) for label, value in pairs if value is not None]
        if pairs:
            plt.figure(figsize=(8, 5))
            plt.bar([pair[0] for pair in pairs], [pair[1] for pair in pairs])
            plt.ylabel("Clearance")
            plt.title("Harmful Clearance Rates")
            plt.xticks(rotation=20)
            plt.tight_layout()
            plt.savefig(figures_dir / "harmful_clearance_rates.png", dpi=200)
            plt.close()

    labels = []
    vals = []
    for stype in ["safe_neighbor", "retain"]:
        if stype in by_type:
            total = _finite_float(by_type[stype].get("R_total_after_mean"))
            high_rate = _finite_float(by_type[stype].get("high_risk_rate_after"))
            if total is not None:
                labels.append(f"{stype}\nR_total")
                vals.append(total)
            if high_rate is not None:
                labels.append(f"{stype}\nhigh_rate")
                vals.append(high_rate)
    if vals:
        plt.figure(figsize=(8, 5))
        plt.bar(labels, vals)
        plt.title("Safe/Retain Preservation Metrics")
        plt.tight_layout()
        plt.savefig(figures_dir / "preservation_metrics.png", dpi=200)
        plt.close()
