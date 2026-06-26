#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import benchmark_loader as loader  # noqa: E402
from run_task_a_preprocessing_benchmark import (  # noqa: E402
    STRATEGIES,
    TRAIN_DATASET,
    build_model,
    internal_cv_scores,
    load_task_matrix,
    pick_best_threshold,
    score_predictions,
    transform_for_method,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the preprocessing benchmark for a task-ready dataset split."
    )
    parser.add_argument(
        "--task-id",
        required=True,
        help="Task identifier, for example task_b or task_c",
    )
    parser.add_argument(
        "--task-dir",
        default="results/tasks",
        help="Directory containing task sample tables",
    )
    parser.add_argument(
        "--matrix-root",
        default="data/processed/task_matrices",
        help="Root directory containing task matrix subdirectories",
    )
    parser.add_argument(
        "--train-dataset",
        default=TRAIN_DATASET,
        help="Primary training dataset ID",
    )
    parser.add_argument(
        "--out",
        help="Output summary TSV; defaults to results/benchmarks/<task>_preprocessing_benchmark.tsv",
    )
    parser.add_argument(
        "--predictions-out",
        help="Output prediction TSV; defaults to results/benchmarks/<task>_preprocessing_predictions.tsv",
    )
    return parser.parse_args()


def load_task_metadata(task_dir: Path, task_id: str) -> pd.DataFrame:
    return pd.read_csv(task_dir / f"{task_id}_samples.tsv", sep="\t")


def main() -> int:
    args = parse_args()
    root = Path(".")
    task_dir = root / args.task_dir
    matrix_dir = root / args.matrix_root / args.task_id
    out_path = root / (
        args.out or f"results/benchmarks/{args.task_id}_preprocessing_benchmark.tsv"
    )
    predictions_out = root / (
        args.predictions_out or f"results/benchmarks/{args.task_id}_preprocessing_predictions.tsv"
    )

    task_def = loader.TASK_DEFINITIONS[args.task_id]
    eval_datasets = [dataset_id for dataset_id in task_def["dataset_ids"] if dataset_id != args.train_dataset]
    metadata = load_task_metadata(task_dir, args.task_id)
    train_meta = metadata.loc[metadata["dataset_id"] == args.train_dataset].copy()
    train_matrix = load_task_matrix(matrix_dir, args.train_dataset).loc[train_meta["sample_id"]]
    y_train = train_meta["binary_target"].astype(int)

    eval_matrices = {
        dataset_id: load_task_matrix(matrix_dir, dataset_id)
        for dataset_id in eval_datasets
    }

    prediction_rows: list[dict[str, str | float | int]] = []
    summary_rows: list[dict[str, str | float | int]] = []

    for strategy_def in STRATEGIES:
        strategy = strategy_def["strategy"]
        method = strategy_def["preprocessing_method"]
        external_reference_strategy = strategy_def["external_reference_strategy"]
        cv_scores = internal_cv_scores(train_matrix, y_train, method=method)
        best_threshold, _ = pick_best_threshold(y_train, cv_scores)

        for threshold_type, threshold in [
            ("default_0_5", 0.5),
            ("train_cv_optimal", best_threshold),
        ]:
            metrics = score_predictions(y_train, cv_scores, threshold=threshold)
            summary_rows.append(
                {
                    "task_id": args.task_id,
                    "task_name": task_def["task_name"],
                    "method": strategy,
                    "strategy": strategy,
                    "preprocessing_method": method,
                    "external_reference_strategy": "training_fold_only",
                    "evaluation_type": "internal_cv",
                    "threshold_type": threshold_type,
                    "threshold": float(threshold),
                    "train_dataset": args.train_dataset,
                    "test_dataset": args.train_dataset,
                    "n_samples": int(train_meta.shape[0]),
                    "n_positive": int(y_train.sum()),
                    "n_negative": int((1 - y_train).sum()),
                    **metrics,
                }
            )

        for sample_id, score in cv_scores.items():
            prediction_rows.append(
                {
                    "task_id": args.task_id,
                    "method": strategy,
                    "strategy": strategy,
                    "preprocessing_method": method,
                    "external_reference_strategy": "training_fold_only",
                    "evaluation_type": "internal_cv",
                    "dataset_id": args.train_dataset,
                    "sample_id": sample_id,
                    "binary_target": int(train_meta.loc[train_meta["sample_id"] == sample_id, "binary_target"].iloc[0]),
                    "predicted_probability": float(score),
                    "default_label": int(score >= 0.5),
                    "train_opt_threshold": float(best_threshold),
                    "train_opt_label": int(score >= best_threshold),
                }
            )

        full_transform = transform_for_method(
            method=method,
            x_train=train_matrix,
            x_eval=train_matrix,
            use_eval_reference=False,
        )
        model = build_model()
        model.fit(full_transform.x_train, y_train)

        for dataset_id in eval_datasets:
            test_meta = metadata.loc[metadata["dataset_id"] == dataset_id].copy()
            test_matrix = eval_matrices[dataset_id].loc[test_meta["sample_id"]]
            y_test = test_meta["binary_target"].astype(int)
            external_transform = transform_for_method(
                method=method,
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
                summary_rows.append(
                    {
                        "task_id": args.task_id,
                        "task_name": task_def["task_name"],
                        "method": strategy,
                        "strategy": strategy,
                        "preprocessing_method": method,
                        "external_reference_strategy": external_reference_strategy,
                        "evaluation_type": "external",
                        "threshold_type": threshold_type,
                        "threshold": float(threshold),
                        "train_dataset": args.train_dataset,
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
                        "task_id": args.task_id,
                        "method": strategy,
                        "strategy": strategy,
                        "preprocessing_method": method,
                        "external_reference_strategy": external_reference_strategy,
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
