#!/usr/bin/env python3

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import benchmark_loader as loader  # noqa: E402
from run_task_a_preprocessing_benchmark import (  # noqa: E402
    METHODS,
    RANDOM_STATE,
    TRAIN_DATASET,
    build_model,
    load_task_matrix,
    transform_for_method,
)


TASK_IDS = ["task_a", "task_b"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export fold-level coefficient stability summaries for logistic baselines."
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
        "--summary-out",
        default="results/models/fold_coefficient_summary.tsv",
        help="Output TSV for fold-level coefficient summaries.",
    )
    parser.add_argument(
        "--pairwise-out",
        default="results/models/fold_coefficient_pairwise_stability.tsv",
        help="Output TSV for pairwise fold coefficient stability.",
    )
    return parser.parse_args()


def load_task_metadata(task_dir: Path, task_id: str) -> pd.DataFrame:
    return pd.read_csv(task_dir / f"{task_id}_samples.tsv", sep="\t")


def coefficient_summary(coef: pd.Series) -> dict[str, float | int]:
    abs_coef = coef.abs()
    max_abs = float(abs_coef.max())
    return {
        "n_features": int(coef.shape[0]),
        "n_nonzero_abs_gt_1e_12": int((abs_coef > 1e-12).sum()),
        "n_abs_gt_1pct_max": int((abs_coef >= 0.01 * max_abs).sum()) if max_abs > 0 else 0,
        "n_abs_gt_5pct_max": int((abs_coef >= 0.05 * max_abs).sum()) if max_abs > 0 else 0,
        "max_abs_coefficient": max_abs,
        "median_abs_coefficient": float(abs_coef.median()),
        "l1_norm": float(abs_coef.sum()),
        "l2_norm": float(np.sqrt(np.square(coef.to_numpy()).sum())),
    }


def sign_agreement(left: pd.Series, right: pd.Series) -> float:
    left_sign = np.sign(left.to_numpy())
    right_sign = np.sign(right.to_numpy())
    return float((left_sign == right_sign).mean())


def main() -> int:
    args = parse_args()
    task_dir = Path(args.task_dir)
    matrix_root = Path(args.matrix_root)
    summary_rows: list[dict[str, object]] = []
    pairwise_rows: list[dict[str, object]] = []

    for task_id in TASK_IDS:
        task_def = loader.TASK_DEFINITIONS[task_id]
        metadata = load_task_metadata(task_dir, task_id)
        train_meta = metadata.loc[metadata["dataset_id"] == TRAIN_DATASET].copy()
        matrix_dir = matrix_root / task_id
        train_matrix = load_task_matrix(matrix_dir, TRAIN_DATASET).loc[train_meta["sample_id"]]
        y_train = train_meta["binary_target"].astype(int)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

        for method in METHODS:
            fold_coefficients: dict[int, pd.Series] = {}
            for fold_id, (train_idx, _) in enumerate(cv.split(train_matrix, y_train), start=1):
                fold_train = train_matrix.iloc[train_idx]
                fold_y = y_train.iloc[train_idx]
                transformed = transform_for_method(
                    method=method,
                    x_train=fold_train,
                    x_eval=fold_train,
                    use_eval_reference=False,
                )
                model = build_model()
                model.fit(transformed.x_train, fold_y)
                coef = pd.Series(model.coef_.ravel(), index=transformed.x_train.columns)
                fold_coefficients[fold_id] = coef
                summary_rows.append(
                    {
                        "task_id": task_id,
                        "task_name": task_def["task_name"],
                        "train_dataset": TRAIN_DATASET,
                        "preprocessing_method": method,
                        "fold_id": fold_id,
                        "penalty": "l2",
                        "C": 1.0,
                        **coefficient_summary(coef),
                    }
                )

            for left_id, right_id in itertools.combinations(sorted(fold_coefficients), 2):
                left = fold_coefficients[left_id]
                right = fold_coefficients[right_id].loc[left.index]
                pairwise_rows.append(
                    {
                        "task_id": task_id,
                        "task_name": task_def["task_name"],
                        "train_dataset": TRAIN_DATASET,
                        "preprocessing_method": method,
                        "left_fold": left_id,
                        "right_fold": right_id,
                        "pearson_correlation": float(left.corr(right, method="pearson")),
                        "spearman_correlation": float(left.corr(right, method="spearman")),
                        "coefficient_sign_agreement": sign_agreement(left, right),
                    }
                )

    summary_out = Path(args.summary_out)
    pairwise_out = Path(args.pairwise_out)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    pairwise_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(summary_out, sep="\t", index=False)
    pd.DataFrame(pairwise_rows).to_csv(pairwise_out, sep="\t", index=False)
    print(f"[wrote] {summary_out}")
    print(f"[wrote] {pairwise_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
