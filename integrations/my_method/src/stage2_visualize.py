from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .utils import ensure_dir, logger


TYPE_ORDER = ["harmful_trigger", "safe_neighbor", "retain"]


def _types(df: pd.DataFrame) -> list[str]:
    return [sample_type for sample_type in TYPE_ORDER if (df["sample_type"] == sample_type).any()]


def _boxplot(df: pd.DataFrame, split: str, column: str, ylabel: str, path: Path) -> None:
    types = _types(df)
    values = [df.loc[df["sample_type"] == t, column].dropna().values for t in types]
    if not types or all(len(v) == 0 for v in values):
        logger.warning("Skipping %s boxplot for split=%s due to insufficient data.", column, split)
        return
    ensure_dir(path.parent)
    plt.figure(figsize=(8, 5))
    plt.boxplot(values, labels=types, showmeans=True)
    plt.title(f"{split} {ylabel}")
    plt.xlabel("sample_type")
    plt.ylabel(ylabel)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _scatter(df: pd.DataFrame, split: str, path: Path) -> None:
    subset = df.dropna(subset=["R_explicit", "R_implicit_norm"])
    if subset.empty:
        logger.warning("Skipping explicit-vs-implicit scatter for split=%s.", split)
        return
    ensure_dir(path.parent)
    plt.figure(figsize=(7, 5))
    for sample_type in _types(subset):
        group = subset[subset["sample_type"] == sample_type]
        plt.scatter(group["R_explicit"], group["R_implicit_norm"], label=sample_type, alpha=0.75)
    plt.title(f"{split} explicit vs implicit risk")
    plt.xlabel("R_explicit")
    plt.ylabel("R_implicit_norm")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _refusal_rate(df: pd.DataFrame, split: str, path: Path) -> None:
    types = _types(df)
    if not types:
        return
    values = [(df.loc[df["sample_type"] == t, "refusal"].dropna() >= 0.5).mean() for t in types]
    ensure_dir(path.parent)
    plt.figure(figsize=(8, 5))
    plt.bar(types, values)
    plt.title(f"{split} refusal rate")
    plt.xlabel("sample_type")
    plt.ylabel("refusal_rate")
    plt.ylim(0, 1)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _paired_diff(df: pd.DataFrame, split: str, path: Path) -> None:
    harmful = df[df["sample_type"] == "harmful_trigger"].dropna(subset=["pair_id", "R_total"])
    safe = df[df["sample_type"] == "safe_neighbor"].dropna(subset=["pair_id", "R_total"])
    merged = harmful[["pair_id", "R_total"]].merge(safe[["pair_id", "R_total"]], on="pair_id", suffixes=("_harmful", "_safe"))
    if merged.empty:
        logger.warning("Skipping paired total diff histogram for split=%s.", split)
        return
    diff = merged["R_total_harmful"] - merged["R_total_safe"]
    ensure_dir(path.parent)
    plt.figure(figsize=(8, 5))
    plt.hist(diff.values, bins=min(20, max(5, len(diff))))
    plt.axvline(0.0, linestyle="--", linewidth=1)
    plt.title(f"{split} paired total risk difference")
    plt.xlabel("R_total(harmful) - R_total(safe_neighbor)")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def generate_stage2_figures(df: pd.DataFrame, split: str, figures_dir: Path) -> None:
    if df.empty:
        logger.warning("Skipping Stage 2 figures for empty split=%s.", split)
        return
    ensure_dir(figures_dir)
    _boxplot(df, split, "R_explicit", "explicit risk", figures_dir / f"{split}_explicit_risk_boxplot.png")
    _boxplot(df, split, "R_implicit_norm", "implicit risk", figures_dir / f"{split}_implicit_risk_boxplot.png")
    _boxplot(df, split, "R_total", "total risk", figures_dir / f"{split}_total_risk_boxplot.png")
    _scatter(df, split, figures_dir / f"{split}_explicit_vs_implicit_scatter.png")
    _refusal_rate(df, split, figures_dir / f"{split}_refusal_rate_bar.png")
    _paired_diff(df, split, figures_dir / f"{split}_paired_total_diff_hist.png")
