#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
RANDOM_STATE = 20260313
N_BOOT = 2000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute stratified bootstrap confidence intervals for external benchmark metrics."
    )
    parser.add_argument(
        "--out-long",
        default="results/tables/external_metric_ci_long.tsv",
        help="Long-format CI table.",
    )
    parser.add_argument(
        "--out-wide",
        default="results/tables/external_metric_ci_wide.tsv",
        help="Wide-format CI table.",
    )
    return parser.parse_args()


def score_metrics(y_true: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float]:
    labels = (score >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, score)),
        "average_precision": float(average_precision_score(y_true, score)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, labels)),
        "brier_score": float(brier_score_loss(y_true, score)),
        "predicted_positive_rate": float(labels.mean()),
    }


def stratified_bootstrap_indices(y_true: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    positive_idx = np.flatnonzero(y_true == 1)
    negative_idx = np.flatnonzero(y_true == 0)
    pos_draw = rng.choice(positive_idx, size=positive_idx.size, replace=True)
    neg_draw = rng.choice(negative_idx, size=negative_idx.size, replace=True)
    return np.concatenate([pos_draw, neg_draw])


def bootstrap_ci(
    y_true: np.ndarray,
    score: np.ndarray,
    threshold: float,
    n_boot: int = N_BOOT,
) -> dict[str, tuple[float, float, float]]:
    rng = np.random.default_rng(RANDOM_STATE)
    metrics = score_metrics(y_true, score, threshold)
    boot_values: dict[str, list[float]] = {name: [] for name in metrics}

    for _ in range(n_boot):
        idx = stratified_bootstrap_indices(y_true, rng)
        boot_y = y_true[idx]
        boot_score = score[idx]
        boot_metrics = score_metrics(boot_y, boot_score, threshold)
        for key, value in boot_metrics.items():
            boot_values[key].append(value)

    summary: dict[str, tuple[float, float, float]] = {}
    for key, estimate in metrics.items():
        values = np.asarray(boot_values[key], dtype=float)
        lower, upper = np.quantile(values, [0.025, 0.975])
        summary[key] = (float(estimate), float(lower), float(upper))
    return summary


def collect_groups() -> list[dict[str, str]]:
    groups = []
    task_specs = [
        (
            "task_a",
            ROOT / "results/benchmarks/task_a_preprocessing_predictions.tsv",
            "method",
        ),
        (
            "task_b",
            ROOT / "results/benchmarks/task_b_preprocessing_predictions.tsv",
            "method",
        ),
    ]
    for task_id, path, method_col in task_specs:
        df = pd.read_csv(path, sep="\t")
        df = df.loc[df["evaluation_type"] == "external"].copy()
        if "task_id" not in df.columns:
            df["task_id"] = task_id
        for (dataset_id, method), group in df.groupby(["dataset_id", method_col]):
            groups.append(
                {
                    "task_id": task_id,
                    "dataset_id": dataset_id,
                    "method": method,
                    "group_key": f"{task_id}:{dataset_id}:{method}",
                }
            )
    return groups


def load_group_predictions(task_id: str, dataset_id: str, method: str) -> tuple[np.ndarray, np.ndarray]:
    if task_id == "task_a":
        path = ROOT / "results/benchmarks/task_a_preprocessing_predictions.tsv"
        method_col = "method"
    elif task_id == "task_b":
        path = ROOT / "results/benchmarks/task_b_preprocessing_predictions.tsv"
        method_col = "method"
    else:
        raise ValueError(f"Unsupported task_id: {task_id}")

    df = pd.read_csv(path, sep="\t")
    df = df.loc[
        (df["evaluation_type"] == "external")
        & (df["dataset_id"] == dataset_id)
        & (df[method_col] == method)
    ].copy()
    return (
        df["binary_target"].to_numpy(dtype=int),
        df["predicted_probability"].to_numpy(dtype=float),
    )


def main() -> int:
    args = parse_args()
    rows: list[dict[str, str | int | float]] = []

    for group in collect_groups():
        y_true, score = load_group_predictions(
            task_id=group["task_id"],
            dataset_id=group["dataset_id"],
            method=group["method"],
        )
        ci = bootstrap_ci(y_true=y_true, score=score, threshold=0.5)
        for metric_name, (estimate, ci_lower, ci_upper) in ci.items():
            rows.append(
                {
                    "task_id": group["task_id"],
                    "dataset_id": group["dataset_id"],
                    "method": group["method"],
                    "metric": metric_name,
                    "estimate": estimate,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "n_samples": int(y_true.size),
                    "n_positive": int(y_true.sum()),
                    "n_negative": int((1 - y_true).sum()),
                    "threshold": 0.5,
                    "n_bootstrap": N_BOOT,
                }
            )

    long_df = pd.DataFrame(rows).sort_values(
        ["task_id", "dataset_id", "method", "metric"]
    )
    wide_df = (
        long_df.pivot_table(
            index=["task_id", "dataset_id", "method", "n_samples", "n_positive", "n_negative"],
            columns="metric",
            values=["estimate", "ci_lower", "ci_upper"],
        )
        .sort_index(axis=1)
        .reset_index()
    )
    wide_df.columns = [
        "_".join([part for part in col if str(part) != ""])
        if isinstance(col, tuple)
        else col
        for col in wide_df.columns
    ]

    out_long = ROOT / args.out_long
    out_wide = ROOT / args.out_wide
    out_long.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(out_long, sep="\t", index=False)
    wide_df.to_csv(out_wide, sep="\t", index=False)
    print(f"[wrote] {out_long}")
    print(f"[wrote] {out_wide}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
