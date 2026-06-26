#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export calibration intercept/slope and calibration-curve source data."
    )
    parser.add_argument(
        "--predictions",
        default="results/benchmarks/task_a_calibration_predictions.tsv",
        help="Calibration prediction TSV.",
    )
    parser.add_argument(
        "--diagnostics-out",
        default="results/tables/calibration_diagnostics.tsv",
        help="Output TSV for calibration diagnostics.",
    )
    parser.add_argument(
        "--curve-out",
        default="results/tables/calibration_curve_source.tsv",
        help="Output TSV for calibration-curve source data.",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=10,
        help="Number of uniform probability bins for calibration curves.",
    )
    return parser.parse_args()


def clipped_logit(scores: pd.Series) -> np.ndarray:
    clipped = np.clip(scores.astype(float).to_numpy(), 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


def calibration_intercept_slope(y_true: pd.Series, scores: pd.Series) -> tuple[float, float]:
    y = y_true.astype(int).to_numpy()
    if len(np.unique(y)) < 2 or scores.nunique(dropna=True) < 2:
        return float("nan"), float("nan")
    model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    model.fit(clipped_logit(scores), y)
    return float(model.intercept_[0]), float(model.coef_[0][0])


def expected_calibration_error(y_true: pd.Series, scores: pd.Series, n_bins: int) -> float:
    curve = calibration_bins(y_true, scores, n_bins)
    if curve.empty:
        return float("nan")
    weights = curve["n"] / curve["n"].sum()
    return float((weights * (curve["observed_fraction"] - curve["mean_predicted"]).abs()).sum())


def calibration_bins(y_true: pd.Series, scores: pd.Series, n_bins: int) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "binary_target": y_true.astype(int),
            "predicted_probability": scores.astype(float),
        }
    )
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    frame["bin"] = pd.cut(
        frame["predicted_probability"],
        bins=bins,
        include_lowest=True,
        right=True,
        duplicates="drop",
    )
    rows = []
    for interval, group in frame.groupby("bin", observed=True):
        rows.append(
            {
                "bin": str(interval),
                "n": int(group.shape[0]),
                "mean_predicted": float(group["predicted_probability"].mean()),
                "observed_fraction": float(group["binary_target"].mean()),
                "bin_min": float(interval.left),
                "bin_max": float(interval.right),
            }
        )
    return pd.DataFrame(rows)


def group_columns(df: pd.DataFrame) -> list[str]:
    candidates = [
        "method",
        "strategy",
        "preprocessing_method",
        "external_reference_strategy",
        "calibration_method",
        "evaluation_type",
        "dataset_id",
    ]
    return [column for column in candidates if column in df.columns]


def main() -> int:
    args = parse_args()
    predictions = pd.read_csv(args.predictions, sep="\t")
    columns = group_columns(predictions)
    diagnostic_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []

    for keys, group in predictions.groupby(columns, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(columns, keys))
        intercept, slope = calibration_intercept_slope(
            group["binary_target"], group["predicted_probability"]
        )
        diagnostic_rows.append(
            {
                **base,
                "n_samples": int(group.shape[0]),
                "n_positive": int(group["binary_target"].sum()),
                "n_negative": int((1 - group["binary_target"].astype(int)).sum()),
                "brier_score": float(
                    brier_score_loss(group["binary_target"], group["predicted_probability"])
                ),
                "calibration_intercept": intercept,
                "calibration_slope": slope,
                "mean_predicted_probability": float(group["predicted_probability"].mean()),
                "observed_positive_rate": float(group["binary_target"].mean()),
                "expected_calibration_error": expected_calibration_error(
                    group["binary_target"], group["predicted_probability"], args.n_bins
                ),
                "calibration_model": "logistic_recalibration_on_logit_predicted_probability",
            }
        )
        curve = calibration_bins(group["binary_target"], group["predicted_probability"], args.n_bins)
        for row in curve.to_dict("records"):
            curve_rows.append({**base, **row})

    diagnostics_out = Path(args.diagnostics_out)
    curve_out = Path(args.curve_out)
    diagnostics_out.parent.mkdir(parents=True, exist_ok=True)
    curve_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(diagnostic_rows).to_csv(diagnostics_out, sep="\t", index=False)
    pd.DataFrame(curve_rows).to_csv(curve_out, sep="\t", index=False)
    print(f"[wrote] {diagnostics_out}")
    print(f"[wrote] {curve_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
