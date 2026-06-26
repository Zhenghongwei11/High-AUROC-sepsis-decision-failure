#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import benchmark_loader as loader  # noqa: E402


TASK_IDS = ["task_a", "task_b", "task_c"]
THRESHOLD_GRID = np.round(np.linspace(0.0, 1.0, 101), 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export confusion-matrix diagnostics and threshold sweeps from benchmark predictions."
    )
    parser.add_argument(
        "--benchmark-dir",
        default="results/benchmarks",
        help="Directory containing <task>_preprocessing_predictions.tsv files.",
    )
    parser.add_argument(
        "--diagnostics-out",
        default="results/tables/classification_diagnostics.tsv",
        help="Output TSV for fixed-threshold diagnostics.",
    )
    parser.add_argument(
        "--threshold-sweep-out",
        default="results/tables/threshold_sweep.tsv",
        help="Output TSV for threshold sweep diagnostics.",
    )
    return parser.parse_args()


def safe_divide(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def classification_metrics(y_true: pd.Series, scores: pd.Series, threshold: float) -> dict[str, float | int]:
    y = y_true.astype(int).to_numpy()
    score = scores.astype(float).to_numpy()
    pred = (score >= threshold).astype(int)

    tp = int(((y == 1) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    n_positive = int((y == 1).sum())
    n_negative = int((y == 0).sum())
    sensitivity = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)

    if np.isnan(sensitivity) or np.isnan(specificity):
        balanced_accuracy = float("nan")
    else:
        balanced_accuracy = float((sensitivity + specificity) / 2)

    return {
        "n_samples": int(len(y)),
        "n_positive": n_positive,
        "n_negative": n_negative,
        "threshold": float(threshold),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": safe_divide(tp, tp + fp),
        "npv": safe_divide(tn, tn + fn),
        "balanced_accuracy": balanced_accuracy,
        "observed_positive_rate": float(n_positive / len(y)) if len(y) else float("nan"),
        "predicted_positive_rate": float(pred.mean()) if len(pred) else float("nan"),
    }


def load_prediction_table(path: Path, task_id: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df["task_id"] = task_id
    df["task_name"] = loader.TASK_DEFINITIONS[task_id]["task_name"]
    for column, fallback in [
        ("strategy", df["method"] if "method" in df.columns else ""),
        ("preprocessing_method", df["method"] if "method" in df.columns else ""),
        ("external_reference_strategy", ""),
    ]:
        if column not in df.columns:
            df[column] = fallback
    return df


def group_columns(df: pd.DataFrame) -> list[str]:
    candidates = [
        "task_id",
        "task_name",
        "method",
        "strategy",
        "preprocessing_method",
        "external_reference_strategy",
        "evaluation_type",
        "dataset_id",
    ]
    return [column for column in candidates if column in df.columns]


def diagnostics_for_group(keys: tuple, group: pd.DataFrame, columns: list[str]) -> list[dict[str, object]]:
    if not isinstance(keys, tuple):
        keys = (keys,)
    base = dict(zip(columns, keys))
    rows: list[dict[str, object]] = []
    threshold_specs = [
        ("default_0_5", 0.5),
        ("train_cv_optimal", float(group["train_opt_threshold"].iloc[0])),
    ]
    for threshold_type, threshold in threshold_specs:
        rows.append(
            {
                **base,
                "threshold_type": threshold_type,
                **classification_metrics(group["binary_target"], group["predicted_probability"], threshold),
            }
        )
    return rows


def sweep_for_group(keys: tuple, group: pd.DataFrame, columns: list[str]) -> list[dict[str, object]]:
    if not isinstance(keys, tuple):
        keys = (keys,)
    base = dict(zip(columns, keys))
    rows: list[dict[str, object]] = []
    for threshold in THRESHOLD_GRID:
        rows.append(
            {
                **base,
                "threshold_type": "grid",
                **classification_metrics(group["binary_target"], group["predicted_probability"], float(threshold)),
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    benchmark_dir = Path(args.benchmark_dir)
    tables = []
    for task_id in TASK_IDS:
        path = benchmark_dir / f"{task_id}_preprocessing_predictions.tsv"
        if not path.exists():
            raise FileNotFoundError(path)
        tables.append(load_prediction_table(path, task_id))

    predictions = pd.concat(tables, ignore_index=True)
    columns = group_columns(predictions)

    diagnostics_rows: list[dict[str, object]] = []
    sweep_rows: list[dict[str, object]] = []
    for keys, group in predictions.groupby(columns, dropna=False, sort=True):
        diagnostics_rows.extend(diagnostics_for_group(keys, group, columns))
        sweep_rows.extend(sweep_for_group(keys, group, columns))

    diagnostics_out = Path(args.diagnostics_out)
    sweep_out = Path(args.threshold_sweep_out)
    diagnostics_out.parent.mkdir(parents=True, exist_ok=True)
    sweep_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(diagnostics_rows).to_csv(diagnostics_out, sep="\t", index=False)
    pd.DataFrame(sweep_rows).to_csv(sweep_out, sep="\t", index=False)
    print(f"[wrote] {diagnostics_out}")
    print(f"[wrote] {sweep_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
