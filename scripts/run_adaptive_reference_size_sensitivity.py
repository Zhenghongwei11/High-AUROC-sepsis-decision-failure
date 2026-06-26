#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import benchmark_loader as loader  # noqa: E402
from run_task_a_preprocessing_benchmark import (  # noqa: E402
    RANDOM_STATE,
    TRAIN_DATASET,
    build_model,
    load_task_matrix,
    robust_scale_with_reference,
    score_predictions,
)


TASK_IDS = ["task_a", "task_b", "task_c"]
REFERENCE_SIZES = [5, 10, 20, 40, 80]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assess how adaptive robust scaling depends on external reference cohort size."
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
        default="results/tables/adaptive_reference_size_sensitivity.tsv",
        help="Output TSV path.",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=50,
        help="Random reference-set replicates per reference size.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260313,
        help="Random seed.",
    )
    return parser.parse_args()


def load_task_metadata(task_dir: Path, task_id: str) -> pd.DataFrame:
    return pd.read_csv(task_dir / f"{task_id}_samples.tsv", sep="\t")


def training_transform(x_train: pd.DataFrame) -> tuple[pd.DataFrame, SimpleImputer]:
    imputer = SimpleImputer(strategy="median")
    train_imputed = pd.DataFrame(
        imputer.fit_transform(x_train),
        index=x_train.index,
        columns=x_train.columns,
    )
    train_medians = train_imputed.median(axis=0)
    train_iqr = train_imputed.quantile(0.75, axis=0) - train_imputed.quantile(0.25, axis=0)
    return robust_scale_with_reference(train_imputed, train_medians, train_iqr), imputer


def adaptive_scale_with_reference_subset(
    x_eval: pd.DataFrame,
    imputer: SimpleImputer,
    reference_sample_ids: np.ndarray,
) -> pd.DataFrame:
    eval_imputed = pd.DataFrame(
        imputer.transform(x_eval),
        index=x_eval.index,
        columns=x_eval.columns,
    )
    reference = eval_imputed.loc[reference_sample_ids]
    ref_medians = reference.median(axis=0)
    ref_iqr = reference.quantile(0.75, axis=0) - reference.quantile(0.25, axis=0)
    return robust_scale_with_reference(eval_imputed, ref_medians, ref_iqr)


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
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
        x_train_scaled, imputer = training_transform(train_matrix)
        model = build_model()
        model.fit(x_train_scaled, y_train)

        for dataset_id in [d for d in task_def["dataset_ids"] if d != TRAIN_DATASET]:
            test_meta = metadata.loc[metadata["dataset_id"] == dataset_id].copy()
            test_matrix = load_task_matrix(matrix_dir, dataset_id).loc[test_meta["sample_id"]]
            y_test = test_meta["binary_target"].astype(int)
            sample_ids = test_meta["sample_id"].to_numpy()
            possible_sizes = [size for size in REFERENCE_SIZES if size < len(sample_ids)]
            possible_sizes.append(len(sample_ids))

            for reference_size in possible_sizes:
                repeats = 1 if reference_size == len(sample_ids) else args.n_repeats
                for repeat in range(repeats):
                    if reference_size == len(sample_ids):
                        reference_ids = sample_ids
                    else:
                        reference_ids = rng.choice(sample_ids, size=reference_size, replace=False)
                    x_eval_scaled = adaptive_scale_with_reference_subset(
                        test_matrix,
                        imputer=imputer,
                        reference_sample_ids=reference_ids,
                    )
                    scores = pd.Series(
                        model.predict_proba(x_eval_scaled)[:, 1],
                        index=test_meta["sample_id"],
                    )
                    metrics = score_predictions(y_test, scores, threshold=0.5)
                    rows.append(
                        {
                            "task_id": task_id,
                            "task_name": task_def["task_name"],
                            "train_dataset": TRAIN_DATASET,
                            "test_dataset": dataset_id,
                            "reference_size": int(reference_size),
                            "reference_fraction": float(reference_size / len(sample_ids)),
                            "repeat": int(repeat),
                            "n_samples": int(test_meta.shape[0]),
                            "n_positive": int(y_test.sum()),
                            "n_negative": int((1 - y_test).sum()),
                            "strategy": "robust_scale_external_adaptive_reference_subset",
                            "preprocessing_method": "cohort_robust_scale",
                            "external_reference_strategy": "external_subset_adaptive",
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
