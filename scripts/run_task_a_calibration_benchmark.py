#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_task_a_preprocessing_benchmark import (  # noqa: E402
    EVAL_DATASETS,
    METHODS,
    RANDOM_STATE,
    TRAIN_DATASET,
    load_task_matrix,
    load_task_metadata,
    pick_best_threshold,
    score_predictions,
    transform_for_method,
)


CALIBRATION_METHODS = ["none", "sigmoid", "isotonic"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare probability calibration strategies for Task A."
    )
    parser.add_argument(
        "--task-dir",
        default="results/tasks",
        help="Directory containing task sample tables",
    )
    parser.add_argument(
        "--matrix-dir",
        default="data/processed/task_matrices/task_a",
        help="Directory containing Task A matrices",
    )
    parser.add_argument(
        "--out",
        default="results/benchmarks/task_a_calibration_benchmark.tsv",
        help="Output summary TSV",
    )
    parser.add_argument(
        "--predictions-out",
        default="results/benchmarks/task_a_calibration_predictions.tsv",
        help="Output per-sample prediction TSV",
    )
    return parser.parse_args()


def build_base_model() -> LogisticRegression:
    return LogisticRegression(
        solver="liblinear",
        class_weight="balanced",
        max_iter=5000,
        random_state=RANDOM_STATE,
    )


def build_estimator(calibration_method: str):
    if calibration_method == "none":
        return build_base_model()
    return CalibratedClassifierCV(
        estimator=build_base_model(),
        method=calibration_method,
        cv=3,
        ensemble=False,
    )


def internal_cv_scores(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessing_method: str,
    calibration_method: str,
) -> pd.Series:
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = pd.Series(index=x_train.index, dtype=float)

    for train_idx, val_idx in outer_cv.split(x_train, y_train):
        fold_train = x_train.iloc[train_idx]
        fold_val = x_train.iloc[val_idx]
        fold_y = y_train.iloc[train_idx]
        transformed = transform_for_method(
            method=preprocessing_method,
            x_train=fold_train,
            x_eval=fold_val,
            use_eval_reference=(preprocessing_method == "cohort_robust_scale"),
        )
        estimator = build_estimator(calibration_method)
        estimator.fit(transformed.x_train, fold_y)
        fold_scores = estimator.predict_proba(transformed.x_eval)[:, 1]
        scores.iloc[val_idx] = fold_scores

    return scores


def main() -> int:
    args = parse_args()
    root = Path(".")
    task_dir = root / args.task_dir
    matrix_dir = root / args.matrix_dir
    out_path = root / args.out
    predictions_out = root / args.predictions_out

    metadata = load_task_metadata(task_dir)
    train_meta = metadata.loc[metadata["dataset_id"] == TRAIN_DATASET].copy()
    train_matrix = load_task_matrix(matrix_dir, TRAIN_DATASET).loc[train_meta["sample_id"]]
    y_train = train_meta["binary_target"].astype(int)
    eval_matrices = {dataset_id: load_task_matrix(matrix_dir, dataset_id) for dataset_id in EVAL_DATASETS}

    summary_rows: list[dict[str, str | float | int]] = []
    prediction_rows: list[dict[str, str | float | int]] = []

    for preprocessing_method in METHODS:
        for calibration_method in CALIBRATION_METHODS:
            cv_scores = internal_cv_scores(
                x_train=train_matrix,
                y_train=y_train,
                preprocessing_method=preprocessing_method,
                calibration_method=calibration_method,
            )
            best_threshold, _ = pick_best_threshold(y_train, cv_scores)

            for threshold_type, threshold in [
                ("default_0_5", 0.5),
                ("train_cv_optimal", best_threshold),
            ]:
                metrics = score_predictions(y_train, cv_scores, threshold=threshold)
                summary_rows.append(
                    {
                        "preprocessing_method": preprocessing_method,
                        "calibration_method": calibration_method,
                        "evaluation_type": "internal_cv",
                        "threshold_type": threshold_type,
                        "threshold": float(threshold),
                        "train_dataset": TRAIN_DATASET,
                        "test_dataset": TRAIN_DATASET,
                        "n_samples": int(train_meta.shape[0]),
                        "n_positive": int(y_train.sum()),
                        "n_negative": int((1 - y_train).sum()),
                        **metrics,
                    }
                )

            for sample_id, score in cv_scores.items():
                prediction_rows.append(
                    {
                        "preprocessing_method": preprocessing_method,
                        "calibration_method": calibration_method,
                        "evaluation_type": "internal_cv",
                        "dataset_id": TRAIN_DATASET,
                        "sample_id": sample_id,
                        "binary_target": int(train_meta.loc[train_meta["sample_id"] == sample_id, "binary_target"].iloc[0]),
                        "predicted_probability": float(score),
                        "default_label": int(score >= 0.5),
                        "train_opt_threshold": float(best_threshold),
                        "train_opt_label": int(score >= best_threshold),
                    }
                )

            transformed_train = transform_for_method(
                method=preprocessing_method,
                x_train=train_matrix,
                x_eval=train_matrix,
                use_eval_reference=False,
            )
            estimator = build_estimator(calibration_method)
            estimator.fit(transformed_train.x_train, y_train)

            for dataset_id in EVAL_DATASETS:
                test_meta = metadata.loc[metadata["dataset_id"] == dataset_id].copy()
                test_matrix = eval_matrices[dataset_id].loc[test_meta["sample_id"]]
                y_test = test_meta["binary_target"].astype(int)
                transformed_test = transform_for_method(
                    method=preprocessing_method,
                    x_train=train_matrix,
                    x_eval=test_matrix,
                    use_eval_reference=(preprocessing_method == "cohort_robust_scale"),
                )
                scores = pd.Series(
                    estimator.predict_proba(transformed_test.x_eval)[:, 1],
                    index=test_meta["sample_id"],
                )

                for threshold_type, threshold in [
                    ("default_0_5", 0.5),
                    ("train_cv_optimal", best_threshold),
                ]:
                    metrics = score_predictions(y_test, scores, threshold=threshold)
                    summary_rows.append(
                        {
                            "preprocessing_method": preprocessing_method,
                            "calibration_method": calibration_method,
                            "evaluation_type": "external",
                            "threshold_type": threshold_type,
                            "threshold": float(threshold),
                            "train_dataset": TRAIN_DATASET,
                            "test_dataset": dataset_id,
                            "n_samples": int(test_meta.shape[0]),
                            "n_positive": int(y_test.sum()),
                            "n_negative": int((1 - y_test).sum()),
                            **metrics,
                        }
                    )

                for sample_id, score in scores.items():
                    prediction_rows.append(
                        {
                            "preprocessing_method": preprocessing_method,
                            "calibration_method": calibration_method,
                            "evaluation_type": "external",
                            "dataset_id": dataset_id,
                            "sample_id": sample_id,
                            "binary_target": int(test_meta.loc[test_meta["sample_id"] == sample_id, "binary_target"].iloc[0]),
                            "predicted_probability": float(score),
                            "default_label": int(score >= 0.5),
                            "train_opt_threshold": float(best_threshold),
                            "train_opt_label": int(score >= best_threshold),
                        }
                    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(out_path, sep="\t", index=False)
    pd.DataFrame(prediction_rows).to_csv(predictions_out, sep="\t", index=False)
    print(f"[wrote] {out_path}")
    print(f"[wrote] {predictions_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
