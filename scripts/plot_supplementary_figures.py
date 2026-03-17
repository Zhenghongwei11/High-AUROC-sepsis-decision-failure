#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


METHOD_LABELS = {
    "standard_zscore": "Standard z-score",
    "sample_rank_zscore": "Sample rank + z-score",
    "cohort_robust_scale": "Cohort robust scale",
}
METHOD_ORDER = [
    "Standard z-score",
    "Sample rank + z-score",
    "Cohort robust scale",
]
METHOD_COLORS = {
    "Standard z-score": "#6E7F80",
    "Sample rank + z-score": "#4C6A92",
    "Cohort robust scale": "#8B5E3C",
}
CLASS_COLORS = {
    "Healthy": "#D8DEE5",
    "Sepsis or septic shock": "#9A6B53",
}
DATASET_ORDER = ["GSE65682", "GSE95233", "GSE154918", "GSE28750"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render supplementary academic-style figures.")
    parser.add_argument("--flow", default="results/tables/cohort_flow_summary.tsv", help="Cohort flow TSV")
    parser.add_argument(
        "--taskc-summary",
        default="results/benchmarks/task_c_preprocessing_benchmark.tsv",
        help="Task C preprocessing summary TSV",
    )
    parser.add_argument(
        "--taskc-predictions",
        default="results/benchmarks/task_c_preprocessing_predictions.tsv",
        help="Task C prediction-level TSV",
    )
    parser.add_argument(
        "--calibration",
        default="results/benchmarks/task_a_calibration_benchmark.tsv",
        help="Calibration benchmark TSV",
    )
    parser.add_argument(
        "--coeff-dir",
        default="results/models",
        help="Directory containing exported task coefficient tables",
    )
    parser.add_argument("--outdir", default="results/figures", help="Output directory")
    return parser.parse_args()


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


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.16,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def render_flow_figure(flow: pd.DataFrame, outdir: Path) -> None:
    flow = flow.copy()
    flow = flow.set_index("dataset_id").loc[DATASET_ORDER].reset_index()
    for column in [
        "geo_series_samples",
        "excluded_after_harmonization",
        "mainline_included",
        "task_a_included",
        "task_b_included",
        "task_c_included",
    ]:
        flow[column] = flow[column].astype(int)

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 7.2), constrained_layout=True)

    composition = pd.DataFrame(
        {
            "dataset_id": flow["dataset_id"],
            "Excluded after harmonization": flow["excluded_after_harmonization"],
            "Included in benchmark": flow["mainline_included"],
        }
    ).set_index("dataset_id")
    composition.plot(
        kind="barh",
        stacked=True,
        color=["#D7D2CB", "#6D8A96"],
        ax=axes[0],
        width=0.72,
    )
    axes[0].set_title("Cohort assembly after conservative phenotype harmonization")
    axes[0].set_xlabel("Samples")
    axes[0].set_ylabel("")
    axes[0].legend(title="", frameon=False, loc="lower right")
    add_panel_label(axes[0], "A")

    task_counts = flow.melt(
        id_vars=["dataset_id"],
        value_vars=["task_a_included", "task_b_included", "task_c_included"],
        var_name="task_id",
        value_name="n_samples",
    )
    task_counts["task_id"] = task_counts["task_id"].map(
        {
            "task_a_included": "Task A",
            "task_b_included": "Task B",
            "task_c_included": "Task C",
        }
    )
    sns.barplot(
        data=task_counts,
        x="dataset_id",
        y="n_samples",
        hue="task_id",
        palette=["#4C6A92", "#8B5E3C", "#7A8E55"],
        ax=axes[1],
    )
    axes[1].set_title("Task-specific analyzable samples by cohort")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Samples")
    axes[1].legend(title="", frameon=False, loc="upper right")
    add_panel_label(axes[1], "B")

    for ax in axes:
        sns.despine(ax=ax)

    png_path = outdir / "figureS1_cohort_flow.png"
    pdf_path = outdir / "figureS1_cohort_flow.pdf"
    fig.savefig(png_path, dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(pdf_path, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def render_taskc_figure(taskc_summary: pd.DataFrame, taskc_predictions: pd.DataFrame, calibration: pd.DataFrame, outdir: Path) -> None:
    taskc_summary = taskc_summary.copy()
    taskc_summary["method_label"] = taskc_summary["method"].map(METHOD_LABELS)
    taskc_predictions = taskc_predictions.copy()
    taskc_predictions["method_label"] = taskc_predictions["method"].map(METHOD_LABELS)
    taskc_predictions["class_label"] = taskc_predictions["binary_target"].map({0: "Healthy", 1: "Sepsis or septic shock"})

    gse154918_summary = taskc_summary[
        (taskc_summary["evaluation_type"] == "external")
        & (taskc_summary["threshold_type"] == "default_0_5")
        & (taskc_summary["test_dataset"] == "GSE154918")
    ].copy()
    gse154918_summary["balanced_accuracy_pct"] = gse154918_summary["balanced_accuracy"] * 100
    gse154918_summary["roc_auc_pct"] = gse154918_summary["roc_auc"] * 100
    gse154918_summary["predicted_positive_rate_pct"] = gse154918_summary["predicted_positive_rate"] * 100
    gse154918_summary["observed_positive_rate_pct"] = gse154918_summary["n_positive"] / gse154918_summary["n_samples"] * 100

    gse154918_predictions = taskc_predictions[
        (taskc_predictions["evaluation_type"] == "external") & (taskc_predictions["dataset_id"] == "GSE154918")
    ].copy()

    calibration = calibration.copy()
    calibration = calibration[
        (calibration["evaluation_type"] == "external")
        & (calibration["threshold_type"] == "default_0_5")
        & (calibration["test_dataset"] == "GSE154918")
    ].copy()
    calibration["preprocessing_label"] = calibration["preprocessing_method"].map(METHOD_LABELS)
    calibration["calibration_label"] = calibration["calibration_method"].map(
        {"none": "No post hoc calibration", "sigmoid": "Sigmoid", "isotonic": "Isotonic"}
    )
    calibration["balanced_accuracy_pct"] = calibration["balanced_accuracy"] * 100

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.2), constrained_layout=True)

    metric_plot = gse154918_summary.melt(
        id_vars=["method_label"],
        value_vars=["balanced_accuracy_pct", "roc_auc_pct"],
        var_name="metric",
        value_name="value",
    )
    metric_plot["metric"] = metric_plot["metric"].map(
        {
            "balanced_accuracy_pct": "Balanced accuracy (%)",
            "roc_auc_pct": "AUROC (%)",
        }
    )
    sns.barplot(
        data=metric_plot,
        x="method_label",
        y="value",
        hue="metric",
        order=METHOD_ORDER,
        palette=["#8B5E3C", "#4C6A92"],
        ax=axes[0, 0],
    )
    axes[0, 0].set_title("RNA-seq transfer preserves ranking but not all thresholds")
    axes[0, 0].set_xlabel("")
    axes[0, 0].set_ylabel("Metric value")
    axes[0, 0].tick_params(axis="x", rotation=10)
    axes[0, 0].legend(title="", frameon=False, loc="lower right")
    add_panel_label(axes[0, 0], "A")

    sns.violinplot(
        data=gse154918_predictions,
        x="method_label",
        y="predicted_probability",
        hue="class_label",
        order=METHOD_ORDER,
        palette=CLASS_COLORS,
        inner="box",
        cut=0,
        linewidth=0.8,
        ax=axes[0, 1],
    )
    axes[0, 1].axhline(0.5, linestyle="--", color="#444444", linewidth=0.9)
    axes[0, 1].set_title("Transferred score distributions in the RNA-seq cohort")
    axes[0, 1].set_xlabel("")
    axes[0, 1].set_ylabel("Predicted probability")
    axes[0, 1].set_ylim(-0.02, 1.02)
    axes[0, 1].tick_params(axis="x", rotation=10)
    axes[0, 1].legend(title="", frameon=False, loc="upper right")
    add_panel_label(axes[0, 1], "B")

    separation_plot = gse154918_summary.melt(
        id_vars=["method_label"],
        value_vars=["median_score_negative", "median_score_positive"],
        var_name="score_type",
        value_name="score_value",
    )
    separation_plot["score_type"] = separation_plot["score_type"].map(
        {
            "median_score_negative": "Median healthy score",
            "median_score_positive": "Median sepsis or shock score",
        }
    )
    sns.pointplot(
        data=separation_plot,
        x="method_label",
        y="score_value",
        hue="score_type",
        order=METHOD_ORDER,
        palette=["#8C9AA5", "#9A6B53"],
        markers=["o", "s"],
        linestyles=["-", "--"],
        errorbar=None,
        ax=axes[1, 0],
    )
    axes[1, 0].axhline(0.5, linestyle="--", color="#7A7A7A", linewidth=0.9)
    axes[1, 0].set_title("Median transferred scores separate by preprocessing")
    axes[1, 0].set_xlabel("")
    axes[1, 0].set_ylabel("Median predicted probability")
    axes[1, 0].tick_params(axis="x", rotation=10)
    axes[1, 0].legend(title="", frameon=False, loc="center right")
    add_panel_label(axes[1, 0], "C")

    sns.barplot(
        data=calibration,
        x="preprocessing_label",
        y="balanced_accuracy_pct",
        hue="calibration_label",
        order=METHOD_ORDER,
        palette=["#7B8C8D", "#4C6A92", "#C08A5A"],
        ax=axes[1, 1],
    )
    axes[1, 1].set_title("Post hoc calibration does not rescue poor preprocessing")
    axes[1, 1].set_xlabel("")
    axes[1, 1].set_ylabel("Balanced accuracy (%)")
    axes[1, 1].set_ylim(45, 103)
    axes[1, 1].tick_params(axis="x", rotation=10)
    axes[1, 1].legend(title="", frameon=False, loc="lower right")
    add_panel_label(axes[1, 1], "D")

    for ax in axes.flat:
        sns.despine(ax=ax)

    png_path = outdir / "figureS2_task_c_rnaseq_transfer.png"
    pdf_path = outdir / "figureS2_task_c_rnaseq_transfer.pdf"
    source_path = outdir / "figureS2_task_c_rnaseq_transfer_source.tsv"
    fig.savefig(png_path, dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(pdf_path, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    source = gse154918_summary[
        [
            "train_dataset",
            "test_dataset",
            "method_label",
            "balanced_accuracy_pct",
            "roc_auc_pct",
            "predicted_positive_rate_pct",
            "observed_positive_rate_pct",
            "median_score_negative",
            "median_score_positive",
        ]
    ].copy()
    source["source_block"] = "cross_platform_readout"
    source["calibration_label"] = "No post hoc calibration"

    calibration_export = calibration[
        [
            "train_dataset",
            "test_dataset",
            "preprocessing_label",
            "calibration_label",
            "balanced_accuracy_pct",
            "roc_auc",
            "predicted_positive_rate",
            "n_positive",
            "n_samples",
            "median_score_negative",
            "median_score_positive",
        ]
    ].copy()
    calibration_export = calibration_export.rename(
        columns={
            "preprocessing_label": "method_label",
            "roc_auc": "roc_auc_pct",
            "predicted_positive_rate": "predicted_positive_rate_pct",
            "n_positive": "observed_positive_rate_pct",
        }
    )
    calibration_export["roc_auc_pct"] = calibration_export["roc_auc_pct"] * 100
    calibration_export["predicted_positive_rate_pct"] = calibration_export["predicted_positive_rate_pct"] * 100
    calibration_export["observed_positive_rate_pct"] = (
        calibration_export["observed_positive_rate_pct"] / calibration_export["n_samples"] * 100
    )
    calibration_export["source_block"] = "cross_platform_calibration"
    calibration_export = calibration_export.drop(columns=["n_samples"])

    export = pd.concat([source, calibration_export], ignore_index=True)
    export.to_csv(source_path, sep="\t", index=False)


def render_taska_coefficient_stability(coeff_dir: Path, outdir: Path) -> None:
    method_files = {
        "standard_zscore": coeff_dir / "task_a_standard_zscore_coefficients.tsv.gz",
        "sample_rank_zscore": coeff_dir / "task_a_sample_rank_zscore_coefficients.tsv.gz",
        "cohort_robust_scale": coeff_dir / "task_a_cohort_robust_scale_coefficients.tsv.gz",
    }
    method_label_map = {
        key: METHOD_LABELS[key] for key in method_files
    }

    merged = None
    for method, path in method_files.items():
        df = pd.read_csv(path, sep="\t")[["gene_symbol", "coefficient", "abs_coefficient"]].rename(
            columns={
                "coefficient": method,
                "abs_coefficient": f"{method}_abs",
            }
        )
        merged = df if merged is None else merged.merge(df, on="gene_symbol", how="inner")

    comparisons = [
        ("standard_zscore", "sample_rank_zscore", "A"),
        ("standard_zscore", "cohort_robust_scale", "B"),
        ("sample_rank_zscore", "cohort_robust_scale", "C"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.4), constrained_layout=True)
    flat_axes = [axes[0, 0], axes[0, 1], axes[1, 0]]

    for ax, (left, right, panel_label) in zip(flat_axes, comparisons):
        left_label = method_label_map[left]
        right_label = method_label_map[right]
        ax.hexbin(
            merged[left],
            merged[right],
            gridsize=42,
            mincnt=1,
            cmap="Greys",
            linewidths=0,
        )
        limits = [
            min(merged[left].min(), merged[right].min()),
            max(merged[left].max(), merged[right].max()),
        ]
        ax.plot(limits, limits, linestyle="--", color="#8B5E3C", linewidth=0.9)
        pearson = merged[left].corr(merged[right], method="pearson")
        spearman = merged[left].corr(merged[right], method="spearman")
        sign_agreement = (merged[left] * merged[right] > 0).mean()
        ax.text(
            0.03,
            0.97,
            (
                f"Pearson r = {pearson:.3f}\n"
                f"Spearman rho = {spearman:.3f}\n"
                f"Sign agreement = {sign_agreement:.3f}"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            bbox={"facecolor": "white", "edgecolor": "#DDDDDD", "boxstyle": "round,pad=0.25"},
        )
        ax.set_title(f"{left_label} vs {right_label}")
        ax.set_xlabel(f"{left_label} coefficient")
        ax.set_ylabel(f"{right_label} coefficient")
        add_panel_label(ax, panel_label)

    stable_sets = [
        set(merged.nlargest(50, f"{method}_abs")["gene_symbol"]) for method in method_files
    ]
    stable_genes = sorted(
        set.intersection(*stable_sets),
        key=lambda gene: merged.loc[merged["gene_symbol"] == gene, [f"{m}_abs" for m in method_files]].mean(axis=1).iloc[0],
        reverse=True,
    )
    heatmap_df = merged.set_index("gene_symbol").loc[stable_genes, list(method_files.keys())]
    heatmap_df = heatmap_df.rename(columns=method_label_map)

    sns.heatmap(
        heatmap_df,
        cmap="vlag",
        center=0,
        linewidths=0.5,
        linecolor="#F2F2F2",
        cbar_kws={"label": "Coefficient"},
        ax=axes[1, 1],
    )
    axes[1, 1].set_title("Genes repeatedly prioritized across preprocessing workflows")
    axes[1, 1].set_xlabel("")
    axes[1, 1].set_ylabel("")
    add_panel_label(axes[1, 1], "D")

    for ax in axes.flat:
        sns.despine(ax=ax, left=False, bottom=False)

    png_path = outdir / "figureS3_task_a_coefficient_stability.png"
    pdf_path = outdir / "figureS3_task_a_coefficient_stability.pdf"
    source_path = outdir / "figureS3_task_a_coefficient_stability_source.tsv"
    fig.savefig(png_path, dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(pdf_path, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    export = merged.copy()
    export["top50_all_methods"] = export["gene_symbol"].isin(stable_genes)
    export.to_csv(source_path, sep="\t", index=False)


def main() -> int:
    args = parse_args()
    root = Path(".")
    outdir = root / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    setup_theme()

    flow = pd.read_csv(root / args.flow, sep="\t")
    taskc_summary = pd.read_csv(root / args.taskc_summary, sep="\t")
    taskc_predictions = pd.read_csv(root / args.taskc_predictions, sep="\t")
    calibration = pd.read_csv(root / args.calibration, sep="\t")
    coeff_dir = root / args.coeff_dir

    render_flow_figure(flow, outdir)
    render_taskc_figure(taskc_summary, taskc_predictions, calibration, outdir)
    render_taska_coefficient_stability(coeff_dir, outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
