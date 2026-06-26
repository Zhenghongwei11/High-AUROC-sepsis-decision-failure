#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


METHOD_LABELS = {
    "standard_zscore": "Standard z-score",
    "standard_zscore_train_only": "Standard z-score\n(train ref)",
    "robust_scale_train_only": "Robust scale\n(train ref)",
    "sample_rank_zscore": "Sample rank + z-score",
    "cohort_robust_scale": "Cohort robust scale",
    "robust_scale_external_adaptive": "Robust scale\n(external ref)",
}
METHOD_ORDER = [
    "Standard z-score\n(train ref)",
    "Robust scale\n(train ref)",
    "Sample rank + z-score",
    "Robust scale\n(external ref)",
]
METHOD_COLORS = {
    "Standard z-score\n(train ref)": "#6E7F80",
    "Robust scale\n(train ref)": "#9B8E7E",
    "Sample rank + z-score": "#4C6A92",
    "Robust scale\n(external ref)": "#8B5E3C",
}
CLASS_COLORS = {
    "Non-infectious inflammation": "#CFD6DC",
    "Sepsis": "#9A6B53",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the Task B stress-test figure.")
    parser.add_argument(
        "--summary",
        default="results/benchmarks/task_b_preprocessing_benchmark.tsv",
        help="Task B benchmark summary TSV",
    )
    parser.add_argument(
        "--predictions",
        default="results/benchmarks/task_b_preprocessing_predictions.tsv",
        help="Task B prediction TSV",
    )
    parser.add_argument(
        "--outdir",
        default="results/figures",
        help="Output directory",
    )
    return parser.parse_args()


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.18,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def setup_theme() -> None:
    sns.set_theme(
        style="ticks",
        context="paper",
        rc={
            "font.family": "DejaVu Serif",
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
        },
    )


def main() -> int:
    args = parse_args()
    root = Path(".")
    summary = pd.read_csv(root / args.summary, sep="\t")
    predictions = pd.read_csv(root / args.predictions, sep="\t")
    outdir = root / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    setup_theme()

    summary = summary.copy()
    summary["method_label"] = summary["method"].map(METHOD_LABELS)
    external_default = summary[
        (summary["evaluation_type"] == "external") & (summary["threshold_type"] == "default_0_5")
    ].copy()
    internal_default = summary[
        (summary["evaluation_type"] == "internal_cv") & (summary["threshold_type"] == "default_0_5")
    ].copy()
    external_default["balanced_accuracy_pct"] = external_default["balanced_accuracy"] * 100
    external_default["roc_auc_pct"] = external_default["roc_auc"] * 100
    external_default["predicted_positive_rate_pct"] = external_default["predicted_positive_rate"] * 100
    external_default["observed_positive_rate_pct"] = external_default["n_positive"] / external_default["n_samples"] * 100
    internal_default["balanced_accuracy_pct"] = internal_default["balanced_accuracy"] * 100

    predictions = predictions.copy()
    predictions["method_label"] = predictions["method"].map(METHOD_LABELS)
    predictions["class_label"] = predictions["binary_target"].map({0: "Non-infectious inflammation", 1: "Sepsis"})
    external_predictions = predictions[predictions["evaluation_type"] == "external"].copy()

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.4), constrained_layout=True)

    comparison = pd.concat(
        [
            internal_default.assign(split_label="Internal cross-validation"),
            external_default.assign(split_label="External stress test"),
        ],
        ignore_index=True,
    )
    sns.pointplot(
        data=comparison,
        x="method_label",
        y="balanced_accuracy_pct",
        hue="split_label",
        order=METHOD_ORDER,
        palette=["#4C6A92", "#8B5E3C"],
        markers=["o", "s"],
        linestyles=["-", "--"],
        errorbar=None,
        ax=axes[0, 0],
    )
    axes[0, 0].set_title("Balanced accuracy drops with harder negatives")
    axes[0, 0].set_xlabel("")
    axes[0, 0].set_ylabel("Balanced accuracy (%)")
    axes[0, 0].set_ylim(45, 103)
    axes[0, 0].tick_params(axis="x", rotation=10)
    axes[0, 0].legend(title="", frameon=False, loc="lower left")
    add_panel_label(axes[0, 0], "A")

    sns.pointplot(
        data=external_default,
        x="method_label",
        y="roc_auc_pct",
        order=METHOD_ORDER,
        hue="method_label",
        hue_order=METHOD_ORDER,
        palette=METHOD_COLORS,
        markers=["o", "^", "s", "D"],
        linestyles=["-", "-", "-", "--"],
        errorbar=None,
        ax=axes[0, 1],
    )
    axes[0, 1].set_title("External AUROC alone overstates utility")
    axes[0, 1].set_xlabel("")
    axes[0, 1].set_ylabel("AUROC (%)")
    axes[0, 1].set_ylim(50, 103)
    axes[0, 1].tick_params(axis="x", rotation=10)
    if axes[0, 1].legend_ is not None:
        axes[0, 1].legend_.remove()
    add_panel_label(axes[0, 1], "B")

    sns.violinplot(
        data=external_predictions,
        x="method_label",
        y="predicted_probability",
        hue="class_label",
        order=METHOD_ORDER,
        palette=CLASS_COLORS,
        inner="box",
        cut=0,
        linewidth=0.8,
        ax=axes[1, 0],
    )
    axes[1, 0].axhline(0.5, linestyle="--", color="#444444", linewidth=0.9)
    axes[1, 0].set_title("Probability separation depends on preprocessing")
    axes[1, 0].set_xlabel("")
    axes[1, 0].set_ylabel("Predicted probability")
    axes[1, 0].set_ylim(-0.02, 1.02)
    axes[1, 0].tick_params(axis="x", rotation=10)
    axes[1, 0].legend(title="", frameon=False, loc="upper right")
    add_panel_label(axes[1, 0], "C")

    rate_plot = external_default.melt(
        id_vars=["method_label"],
        value_vars=["observed_positive_rate_pct", "predicted_positive_rate_pct"],
        var_name="rate_type",
        value_name="rate_pct",
    )
    rate_plot["rate_type"] = rate_plot["rate_type"].map(
        {
            "observed_positive_rate_pct": "Observed sepsis rate",
            "predicted_positive_rate_pct": "Predicted positive rate",
        }
    )
    sns.barplot(
        data=rate_plot,
        x="method_label",
        y="rate_pct",
        hue="rate_type",
        order=METHOD_ORDER,
        palette=["#445B66", "#BC6C25"],
        ax=axes[1, 1],
    )
    axes[1, 1].set_title("Observed and predicted rates under harder negatives")
    axes[1, 1].set_xlabel("")
    axes[1, 1].set_ylabel("Rate (%)")
    axes[1, 1].set_ylim(0, 100)
    axes[1, 1].tick_params(axis="x", rotation=10)
    axes[1, 1].legend(title="", frameon=False, loc="upper right")
    add_panel_label(axes[1, 1], "D")

    for ax in axes.flat:
        sns.despine(ax=ax)

    png_path = outdir / "figure3_harder_negative_stress_test.png"
    pdf_path = outdir / "figure3_harder_negative_stress_test.pdf"
    source_path = outdir / "figure3_harder_negative_stress_test_source.tsv"
    fig.savefig(png_path, dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(pdf_path, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    external_default.to_csv(source_path, sep="\t", index=False)
    print(f"[wrote] {png_path}")
    print(f"[wrote] {pdf_path}")
    print(f"[wrote] {source_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
