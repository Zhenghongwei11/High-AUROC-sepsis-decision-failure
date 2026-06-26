#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import benchmark_loader as loader  # noqa: E402
from run_task_a_preprocessing_benchmark import (  # noqa: E402
    RANDOM_STATE,
    STRATEGIES,
    TRAIN_DATASET,
    load_task_matrix,
    pick_best_threshold,
    score_predictions,
    transform_for_method,
)


TASK_IDS = ["task_a", "task_b"]
C_VALUES = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate logistic-regression regularization sensitivity for revision benchmarks."
    )
    parser.add_argument(
        "--task-dir",
        default="results/tasks",
        help="Directory containing task sample tables.",
    )
    parser.add_argument(
        "--matrix-root",
        default="data/processed/task_matrices",
        help="Root directory containing task matrices.",
    )
    parser.add_argument(
        "--out",
        default="results/benchmarks/regularization_sensitivity.tsv",
        help="Output TSV path.",
    )
    return parser.parse_args()


def build_model(c_value: float) -> LogisticRegression:
    return LogisticRegression(
        solver="liblinear",
        penalty="l2",
        C=float(c_value),
        class_weight="balanced",
        max_iter=5000,
        random_state=RANDOM_STATE,
    )


def load_task_metadata(task_dir: Path, task_id: str) -> pd.DataFrame:
    return pd.read_csv(task_dir / f"{task_id}_samples.tsv", sep="\t")


def internal_cv_scores(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessing_method: str,
    c_value: float,
) -> pd.Series:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = pd.Series(index=x_train.index, dtype=float)
    for train_idx, val_idx in cv.split(x_train, y_train):
        fold_train = x_train.iloc[train_idx]
        fold_val = x_train.iloc[val_idx]
        fold_y = y_train.iloc[train_idx]
        transformed = transform_for_method(
            method=preprocessing_method,
            x_train=fold_train,
            x_eval=fold_val,
            use_eval_reference=False,
        )
        model = build_model(c_value)
        model.fit(transformed.x_train, fold_y)
        scores.iloc[val_idx] = model.predict_proba(transformed.x_eval)[:, 1]
    return scores


def main() -> int:
    args = parse_args()
    task_dir = Path(args.task_dir)
    matrix_root = Path(args.matrix_root)
    rows: list[dict[str, object]] = []

    for task_id in TASK_IDS:
        task_def = loader.TASK_DEFINITIONS[task_id]
        metadata = load_task_metadata(task_dir, task_id)
        train_meta = metadata.loc[metadata["dataset_id"] == TRAIN_DATASET].copy()
        matrix_dir = matrix_root / task_id
        train_matrix = load_task_matrix(matrix_dir, TRAIN_DATASET).loc[train_meta["sample_id"]]
        y_train = train_meta["binary_target"].astype(int)
        eval_matrices = {
            dataset_id: load_task_matrix(matrix_dir, dataset_id)
            for dataset_id in task_def["dataset_ids"]
            if dataset_id != TRAIN_DATASET
        }

        for c_value in C_VALUES:
            for strategy_def in STRATEGIES:
                strategy = strategy_def["strategy"]
                preprocessing_method = strategy_def["preprocessing_method"]
                cv_scores = internal_cv_scores(train_matrix, y_train, preprocessing_method, c_value)
                best_threshold, _ = pick_best_threshold(y_train, cv_scores)
                for threshold_type, threshold in [
                    ("default_0_5", 0.5),
                    ("train_cv_optimal", best_threshold),
                ]:
                    metrics = score_predictions(y_train, cv_scores, threshold=threshold)
                    rows.append(
                        {
                            "task_id": task_id,
                            "task_name": task_def["task_name"],
                            "method": strategy,
                            "strategy": strategy,
                            "preprocessing_method": preprocessing_method,
                            "external_reference_strategy": "training_fold_only",
                            "regularization_penalty": "l2",
                            "C": float(c_value),
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

                full_transform = transform_for_method(
                    method=preprocessing_method,
                    x_train=train_matrix,
                    x_eval=train_matrix,
                    use_eval_reference=False,
                )
                model = build_model(c_value)
                model.fit(full_transform.x_train, y_train)

                for dataset_id, matrix in eval_matrices.items():
                    test_meta = metadata.loc[metadata["dataset_id"] == dataset_id].copy()
                    test_matrix = matrix.loc[test_meta["sample_id"]]
                    y_test = test_meta["binary_target"].astype(int)
                    external_transform = transform_for_method(
                        method=preprocessing_method,
                        x_train=train_matrix,
                        x_eval=test_matrix,
                        use_eval_reference=bool(strategy_def["use_eval_reference_external"]),
                    )
                    scores = pd.Series(
                        model.predict_proba(external_transform.x_eval)[:, 1],
                        index=test_meta["sample_id"],
                    )
                    for threshold_type, threshold in [
                        ("default_0_5", 0.5),
                        ("train_cv_optimal", best_threshold),
                    ]:
                        metrics = score_predictions(y_test, scores, threshold=threshold)
                        rows.append(
                            {
                                "task_id": task_id,
                                "task_name": task_def["task_name"],
                                "method": strategy,
                                "strategy": strategy,
                                "preprocessing_method": preprocessing_method,
                                "external_reference_strategy": strategy_def["external_reference_strategy"],
                                "regularization_penalty": "l2",
                                "C": float(c_value),
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

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
    print(f"[wrote] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
