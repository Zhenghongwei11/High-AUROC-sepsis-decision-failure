#!/usr/bin/env python3

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from export_classification_diagnostics import classification_metrics, load_prediction_table  # noqa: E402


TASK_IDS = ["task_a", "task_b", "task_c"]
METRICS = [
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "ppv",
    "npv",
    "predicted_positive_rate",
]


def percentile_or_nan(values: np.ndarray, percentile: float) -> float:
    if np.isnan(values).all():
        return float("nan")
    return float(np.nanpercentile(values, percentile))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute paired bootstrap CIs for differences between preprocessing strategies."
    )
    parser.add_argument(
        "--benchmark-dir",
        default="results/benchmarks",
        help="Directory containing <task>_preprocessing_predictions.tsv files.",
    )
    parser.add_argument(
        "--out",
        default="results/tables/paired_bootstrap_differences.tsv",
        help="Output TSV path.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=2000,
        help="Number of paired bootstrap replicates.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260313,
        help="Random seed.",
    )
    return parser.parse_args()


def load_predictions(benchmark_dir: Path) -> pd.DataFrame:
    tables = []
    for task_id in TASK_IDS:
        path = benchmark_dir / f"{task_id}_preprocessing_predictions.tsv"
        tables.append(load_prediction_table(path, task_id))
    return pd.concat(tables, ignore_index=True)


def threshold_for(group: pd.DataFrame, threshold_type: str) -> float:
    if threshold_type == "default_0_5":
        return 0.5
    if threshold_type == "train_cv_optimal":
        return float(group["train_opt_threshold"].iloc[0])
    raise ValueError(f"Unknown threshold_type: {threshold_type}")


def metric_vector(group: pd.DataFrame, indices: np.ndarray, threshold_type: str) -> dict[str, float]:
    sampled = group.iloc[indices]
    metrics = classification_metrics(
        sampled["binary_target"],
        sampled["predicted_probability"],
        threshold_for(group, threshold_type),
    )
    return {metric: float(metrics[metric]) for metric in METRICS}


def paired_bootstrap(
    left: pd.DataFrame,
    right: pd.DataFrame,
    threshold_type: str,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> dict[str, tuple[float, float, float]]:
    merged = left[
        ["sample_id", "binary_target", "predicted_probability", "train_opt_threshold"]
    ].merge(
        right[["sample_id", "predicted_probability", "train_opt_threshold"]],
        on="sample_id",
        suffixes=("_left", "_right"),
        validate="one_to_one",
    )
    y = merged["binary_target"].astype(int).to_numpy()
    positive_idx = np.where(y == 1)[0]
    negative_idx = np.where(y == 0)[0]
    if len(positive_idx) == 0 or len(negative_idx) == 0:
        raise ValueError("Paired bootstrap requires both classes in each external cohort")

    left_group = pd.DataFrame(
        {
            "sample_id": merged["sample_id"],
            "binary_target": merged["binary_target"],
            "predicted_probability": merged["predicted_probability_left"],
            "train_opt_threshold": merged["train_opt_threshold_left"],
        }
    )
    right_group = pd.DataFrame(
        {
            "sample_id": merged["sample_id"],
            "binary_target": merged["binary_target"],
            "predicted_probability": merged["predicted_probability_right"],
            "train_opt_threshold": merged["train_opt_threshold_right"],
        }
    )

    point_left = metric_vector(left_group, np.arange(len(merged)), threshold_type)
    point_right = metric_vector(right_group, np.arange(len(merged)), threshold_type)
    differences = {metric: [] for metric in METRICS}

    for _ in range(n_bootstrap):
        sampled_idx = np.concatenate(
            [
                rng.choice(positive_idx, size=len(positive_idx), replace=True),
                rng.choice(negative_idx, size=len(negative_idx), replace=True),
            ]
        )
        left_metrics = metric_vector(left_group, sampled_idx, threshold_type)
        right_metrics = metric_vector(right_group, sampled_idx, threshold_type)
        for metric in METRICS:
            differences[metric].append(left_metrics[metric] - right_metrics[metric])

    output: dict[str, tuple[float, float, float]] = {}
    for metric in METRICS:
        values = np.asarray(differences[metric], dtype=float)
        output[metric] = (
            float(point_left[metric] - point_right[metric]),
            percentile_or_nan(values, 2.5),
            percentile_or_nan(values, 97.5),
        )
    return output


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    predictions = load_predictions(Path(args.benchmark_dir))
    predictions = predictions.loc[predictions["evaluation_type"] == "external"].copy()

    rows: list[dict[str, object]] = []
    group_cols = ["task_id", "task_name", "dataset_id"]
    for (task_id, task_name, dataset_id), group in predictions.groupby(group_cols, sort=True):
        method_groups = {method: data.copy() for method, data in group.groupby("method", sort=True)}
        for left_method, right_method in itertools.combinations(sorted(method_groups), 2):
            left = method_groups[left_method]
            right = method_groups[right_method]
            left_strategy = left["external_reference_strategy"].iloc[0]
            right_strategy = right["external_reference_strategy"].iloc[0]
            for threshold_type in ["default_0_5", "train_cv_optimal"]:
                diff = paired_bootstrap(
                    left=left,
                    right=right,
                    threshold_type=threshold_type,
                    n_bootstrap=args.n_bootstrap,
                    rng=rng,
                )
                for metric, (estimate, ci_lower, ci_upper) in diff.items():
                    rows.append(
                        {
                            "task_id": task_id,
                            "task_name": task_name,
                            "dataset_id": dataset_id,
                            "left_method": left_method,
                            "right_method": right_method,
                            "left_external_reference_strategy": left_strategy,
                            "right_external_reference_strategy": right_strategy,
                            "threshold_type": threshold_type,
                            "metric": metric,
                            "estimate_left_minus_right": estimate,
                            "ci_lower": ci_lower,
                            "ci_upper": ci_upper,
                            "n_bootstrap": args.n_bootstrap,
                            "bootstrap_type": "paired_stratified_percentile",
                            "degenerate_resample_handling": "positive_and_negative_classes_resampled_separately",
                        }
                    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
    print(f"[wrote] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
