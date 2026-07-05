from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import pandas as pd

from .utils import ensure_dir, logger


TYPE_ORDER = ["harmful_trigger", "safe_neighbor", "retain"]


def _present_types(df: pd.DataFrame) -> List[str]:
    return [sample_type for sample_type in TYPE_ORDER if (df["sample_type"] == sample_type).any()]


def plot_risk_score_boxplot(df: pd.DataFrame, split: str, figures_dir: Path) -> None:
    types = _present_types(df)
    if not types:
        logger.warning("No sample types available for boxplot.")
        return
    ensure_dir(figures_dir)
    values = [df.loc[df["sample_type"] == sample_type, "R_total"].dropna().values for sample_type in types]
    plt.figure(figsize=(8, 5))
    plt.boxplot(values, labels=types, showmeans=True)
    plt.title(f"{split} total risk score")
    plt.xlabel("Sample type")
    plt.ylabel("R_total")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(figures_dir / f"{split}_risk_score_boxplot.png", dpi=200)
    plt.close()


def plot_risk_score_hist(df: pd.DataFrame, split: str, figures_dir: Path) -> None:
    types = _present_types(df)
    if not types:
        logger.warning("No sample types available for histogram.")
        return
    ensure_dir(figures_dir)
    plt.figure(figsize=(8, 5))
    for sample_type in types:
        values = df.loc[df["sample_type"] == sample_type, "R_total"].dropna().values
        if len(values) == 0:
            continue
        plt.hist(values, bins=min(20, max(5, len(values))), alpha=0.45, label=sample_type)
    plt.title(f"{split} total risk score distribution")
    plt.xlabel("R_total")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / f"{split}_risk_score_hist.png", dpi=200)
    plt.close()


def plot_layerwise_risk_bar(df: pd.DataFrame, split: str, layer_cols: Iterable[str], figures_dir: Path) -> None:
    layer_cols = list(layer_cols)
    types = _present_types(df)
    if not types or not layer_cols:
        logger.warning("Not enough data for layerwise bar plot.")
        return
    ensure_dir(figures_dir)
    means = []
    for sample_type in types:
        means.append([df.loc[df["sample_type"] == sample_type, col].mean() for col in layer_cols])

    x = list(range(len(layer_cols)))
    width = 0.8 / max(1, len(types))
    plt.figure(figsize=(10, 5))
    for i, sample_type in enumerate(types):
        offsets = [pos - 0.4 + width / 2 + i * width for pos in x]
        plt.bar(offsets, means[i], width=width, label=sample_type)
    plt.title(f"{split} mean layerwise risk")
    plt.xlabel("Layer")
    plt.ylabel("Mean R_layer")
    plt.xticks(x, [col.replace("R_layer_", "") for col in layer_cols])
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / f"{split}_layerwise_risk_bar.png", dpi=200)
    plt.close()


def plot_paired_diff_hist(df: pd.DataFrame, split: str, figures_dir: Path) -> None:
    harmful = df[df["sample_type"] == "harmful_trigger"].dropna(subset=["pair_id"])
    safe = df[df["sample_type"] == "safe_neighbor"].dropna(subset=["pair_id"])
    merged = harmful[["pair_id", "R_total"]].merge(
        safe[["pair_id", "R_total"]],
        on="pair_id",
        suffixes=("_harmful", "_safe"),
    )
    if merged.empty:
        logger.warning("No paired harmful/safe samples for paired diff histogram.")
        return
    diffs = merged["R_total_harmful"] - merged["R_total_safe"]
    ensure_dir(figures_dir)
    plt.figure(figsize=(8, 5))
    plt.hist(diffs.values, bins=min(20, max(5, len(diffs))), alpha=0.75)
    plt.axvline(0.0, color="black", linestyle="--", linewidth=1)
    plt.title(f"{split} paired risk difference")
    plt.xlabel("R_total(harmful_trigger) - R_total(safe_neighbor)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(figures_dir / f"{split}_paired_diff_hist.png", dpi=200)
    plt.close()


def generate_all_figures(df: pd.DataFrame, split: str, layer_cols: Iterable[str], figures_dir: Path) -> None:
    if df.empty:
        logger.warning("Empty score dataframe; skipping figures.")
        return
    plot_risk_score_boxplot(df, split, figures_dir)
    plot_risk_score_hist(df, split, figures_dir)
    plot_layerwise_risk_bar(df, split, layer_cols, figures_dir)
    plot_paired_diff_hist(df, split, figures_dir)


def plot_k_sweep_val_auc(summary_df: pd.DataFrame, figures_dir: Path) -> None:
    subset = summary_df[summary_df["split"] == "val"].copy()
    if subset.empty or "harmful_vs_safe_auc" not in subset:
        logger.warning("No validation k-sweep AUC data available; skipping k_sweep_val_auc.png.")
        return
    ensure_dir(figures_dir)
    plt.figure(figsize=(8, 5))
    for score_mode, group in subset.groupby("score_mode"):
        group = group.sort_values("k")
        plt.plot(group["k"], group["harmful_vs_safe_auc"], marker="o", label=score_mode)
    plt.title("Stage 1.5 k sweep: validation harmful vs safe AUC")
    plt.xlabel("k")
    plt.ylabel("Val harmful vs safe AUC")
    plt.ylim(0.0, 1.02)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "k_sweep_val_auc.png", dpi=200)
    plt.close()


def plot_k_sweep_val_means(summary_df: pd.DataFrame, figures_dir: Path) -> None:
    subset = summary_df[summary_df["split"] == "val"].copy()
    if subset.empty:
        logger.warning("No validation k-sweep mean data available; skipping k_sweep_val_means.png.")
        return
    ensure_dir(figures_dir)
    score_modes = list(subset["score_mode"].dropna().unique())
    if not score_modes:
        return
    fig, axes = plt.subplots(1, len(score_modes), figsize=(7 * len(score_modes), 5), squeeze=False)
    for ax, score_mode in zip(axes[0], score_modes):
        group = subset[subset["score_mode"] == score_mode].sort_values("k")
        for col, label in [
            ("harmful_mean", "harmful_trigger"),
            ("safe_mean", "safe_neighbor"),
            ("retain_mean", "retain"),
        ]:
            if col in group:
                ax.plot(group["k"], group[col], marker="o", label=label)
        ax.set_title(f"Val mean R_total ({score_mode})")
        ax.set_xlabel("k")
        ax.set_ylabel("Mean R_total")
        ax.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "k_sweep_val_means.png", dpi=200)
    plt.close()


def plot_single_layer_val_auc(summary_df: pd.DataFrame, figures_dir: Path) -> None:
    subset = summary_df[summary_df["split"] == "val"].copy()
    if subset.empty:
        logger.warning("No validation single-layer data available; skipping single_layer_val_auc.png.")
        return
    ensure_dir(figures_dir)
    subset["first_layer"] = subset["layers"].astype(str).str.split(",").str[0].astype(int)
    plt.figure(figsize=(8, 5))
    for score_mode, group in subset.groupby("score_mode"):
        group = group.sort_values("first_layer")
        plt.plot(group["first_layer"], group["harmful_vs_safe_auc"], marker="o", label=score_mode)
    plt.title("Stage 1.5 single-layer ablation: validation AUC")
    plt.xlabel("Layer")
    plt.ylabel("Val harmful vs safe AUC")
    plt.ylim(0.0, 1.02)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "single_layer_val_auc.png", dpi=200)
    plt.close()


def plot_group_layer_val_auc(summary_df: pd.DataFrame, figures_dir: Path) -> None:
    subset = summary_df[summary_df["split"] == "val"].copy()
    if subset.empty:
        logger.warning("No validation group-layer data available; skipping group_layer_val_auc.png.")
        return
    ensure_dir(figures_dir)
    groups = list(dict.fromkeys(subset["layer_set_name"].tolist()))
    score_modes = list(subset["score_mode"].dropna().unique())
    x = list(range(len(groups)))
    width = 0.8 / max(1, len(score_modes))
    plt.figure(figsize=(10, 5))
    for i, score_mode in enumerate(score_modes):
        values = []
        for group_name in groups:
            row = subset[(subset["layer_set_name"] == group_name) & (subset["score_mode"] == score_mode)]
            values.append(float(row["harmful_vs_safe_auc"].iloc[0]) if not row.empty else float("nan"))
        offsets = [pos - 0.4 + width / 2 + i * width for pos in x]
        plt.bar(offsets, values, width=width, label=score_mode)
    plt.title("Stage 1.5 group-layer ablation: validation AUC")
    plt.xlabel("Layer group")
    plt.ylabel("Val harmful vs safe AUC")
    plt.ylim(0.0, 1.02)
    plt.xticks(x, groups, rotation=20)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "group_layer_val_auc.png", dpi=200)
    plt.close()
