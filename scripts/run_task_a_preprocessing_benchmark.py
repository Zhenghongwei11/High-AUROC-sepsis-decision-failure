#!/usr/bin/env python3

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 20260313
TRAIN_DATASET = "GSE65682"
EVAL_DATASETS = ["GSE95233", "GSE154918", "GSE28750"]
METHODS = ["standard_zscore", "sample_rank_zscore", "cohort_robust_scale"]


@dataclass
class TransformBundle:
    x_train: pd.DataFrame
    x_eval: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare preprocessing variants for Task A and diagnose threshold drift."
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
        default="results/benchmarks/task_a_preprocessing_benchmark.tsv",
        help="Output summary TSV",
    )
    parser.add_argument(
        "--predictions-out",
        default="results/benchmarks/task_a_preprocessing_predictions.tsv",
        help="Output per-sample prediction TSV",
    )
    return parser.parse_args()


def load_task_metadata(task_dir: Path) -> pd.DataFrame:
    return pd.read_csv(task_dir / "task_a_samples.tsv", sep="\t")


def load_task_matrix(matrix_dir: Path, dataset_id: str) -> pd.DataFrame:
    matrix = pd.read_csv(
        matrix_dir / f"{dataset_id}_samples_x_genes.tsv.gz",
        sep="\t",
        compression="gzip",
    )
    if "sample_id" not in matrix.columns:
        raise ValueError(f"Task matrix for {dataset_id} is missing sample_id")
    return matrix.set_index("sample_id")


def build_model() -> LogisticRegression:
    return LogisticRegression(
        solver="liblinear",
        class_weight="balanced",
        max_iter=5000,
        random_state=RANDOM_STATE,
    )


def rank_transform(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rank(axis=1, method="average", pct=True)


def fit_standard_transform(x_train: pd.DataFrame, x_eval: pd.DataFrame) -> TransformBundle:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train_imputed = pd.DataFrame(
        imputer.fit_transform(x_train),
        index=x_train.index,
        columns=x_train.columns,
    )
    eval_imputed = pd.DataFrame(
        imputer.transform(x_eval),
        index=x_eval.index,
        columns=x_eval.columns,
    )
    train_scaled = pd.DataFrame(
        scaler.fit_transform(train_imputed),
        index=x_train.index,
        columns=x_train.columns,
    )
    eval_scaled = pd.DataFrame(
        scaler.transform(eval_imputed),
        index=x_eval.index,
        columns=x_eval.columns,
    )
    return TransformBundle(train_scaled, eval_scaled)


def fit_sample_rank_transform(x_train: pd.DataFrame, x_eval: pd.DataFrame) -> TransformBundle:
    ranked_train = rank_transform(x_train)
    ranked_eval = rank_transform(x_eval)
    return fit_standard_transform(ranked_train, ranked_eval)


def robust_scale_with_reference(
    frame: pd.DataFrame,
    medians: pd.Series,
    iqrs: pd.Series,
) -> pd.DataFrame:
    safe_iqr = iqrs.replace(0, 1.0).fillna(1.0)
    return (frame - medians) / safe_iqr


def fit_cohort_robust_transform(
    x_train: pd.DataFrame,
    x_eval: pd.DataFrame,
    use_eval_reference: bool,
) -> TransformBundle:
    imputer = SimpleImputer(strategy="median")
    train_imputed = pd.DataFrame(
        imputer.fit_transform(x_train),
        index=x_train.index,
        columns=x_train.columns,
    )
    eval_imputed = pd.DataFrame(
        imputer.transform(x_eval),
        index=x_eval.index,
        columns=x_eval.columns,
    )

    train_medians = train_imputed.median(axis=0)
    train_iqr = train_imputed.quantile(0.75, axis=0) - train_imputed.quantile(0.25, axis=0)
    train_scaled = robust_scale_with_reference(train_imputed, train_medians, train_iqr)

    if use_eval_reference:
        eval_medians = eval_imputed.median(axis=0)
        eval_iqr = eval_imputed.quantile(0.75, axis=0) - eval_imputed.quantile(0.25, axis=0)
        eval_scaled = robust_scale_with_reference(eval_imputed, eval_medians, eval_iqr)
    else:
        eval_scaled = robust_scale_with_reference(eval_imputed, train_medians, train_iqr)

    return TransformBundle(train_scaled, eval_scaled)


def transform_for_method(
    method: str,
    x_train: pd.DataFrame,
    x_eval: pd.DataFrame,
    use_eval_reference: bool = False,
) -> TransformBundle:
    if method == "standard_zscore":
        return fit_standard_transform(x_train, x_eval)
    if method == "sample_rank_zscore":
        return fit_sample_rank_transform(x_train, x_eval)
    if method == "cohort_robust_scale":
        return fit_cohort_robust_transform(x_train, x_eval, use_eval_reference=use_eval_reference)
    raise ValueError(f"Unknown method: {method}")


def pick_best_threshold(y_true: pd.Series, scores: pd.Series) -> tuple[float, float]:
    candidates = sorted(scores.unique())
    best_threshold = 0.5
    best_score = -1.0
    for threshold in candidates:
        bal_acc = balanced_accuracy_score(y_true, (scores >= threshold).astype(int))
        if bal_acc > best_score:
            best_threshold = float(threshold)
            best_score = float(bal_acc)
    return best_threshold, best_score


def score_predictions(y_true: pd.Series, scores: pd.Series, threshold: float) -> dict[str, float]:
    y_array = np.asarray(y_true, dtype=int)
    score_array = np.asarray(scores, dtype=float)
    labels = (score_array >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_array, score_array)),
        "average_precision": float(average_precision_score(y_array, score_array)),
        "balanced_accuracy": float(balanced_accuracy_score(y_array, labels)),
        "brier_score": float(brier_score_loss(y_array, score_array)),
        "predicted_positive_rate": float(labels.mean()),
        "median_score_negative": float(np.median(score_array[y_array == 0])),
        "median_score_positive": float(np.median(score_array[y_array == 1])),
    }


def internal_cv_scores(x_train: pd.DataFrame, y_train: pd.Series, method: str) -> pd.Series:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = pd.Series(index=x_train.index, dtype=float)

    for train_idx, val_idx in cv.split(x_train, y_train):
        fold_train = x_train.iloc[train_idx]
        fold_val = x_train.iloc[val_idx]
        fold_y = y_train.iloc[train_idx]
        transformed = transform_for_method(
            method=method,
            x_train=fold_train,
            x_eval=fold_val,
            use_eval_reference=False,
        )
        model = build_model()
        model.fit(transformed.x_train, fold_y)
        fold_scores = model.predict_proba(transformed.x_eval)[:, 1]
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

    prediction_rows: list[dict[str, str | float | int]] = []
    summary_rows: list[dict[str, str | float | int]] = []

    eval_matrices = {
        dataset_id: load_task_matrix(matrix_dir, dataset_id)
        for dataset_id in EVAL_DATASETS
    }

    for method in METHODS:
        cv_scores = internal_cv_scores(train_matrix, y_train, method=method)
        best_threshold, best_bal_acc = pick_best_threshold(y_train, cv_scores)

        internal_default = score_predictions(y_train, cv_scores, threshold=0.5)
        internal_best = score_predictions(y_train, cv_scores, threshold=best_threshold)
        summary_rows.append(
            {
                "method": method,
                "evaluation_type": "internal_cv",
                "threshold_type": "default_0_5",
                "threshold": 0.5,
                "train_dataset": TRAIN_DATASET,
                "test_dataset": TRAIN_DATASET,
                "n_samples": int(train_meta.shape[0]),
                "n_positive": int(y_train.sum()),
                "n_negative": int((1 - y_train).sum()),
                **internal_default,
            }
        )
        summary_rows.append(
            {
                "method": method,
                "evaluation_type": "internal_cv",
                "threshold_type": "train_cv_optimal",
                "threshold": best_threshold,
                "train_dataset": TRAIN_DATASET,
                "test_dataset": TRAIN_DATASET,
                "n_samples": int(train_meta.shape[0]),
                "n_positive": int(y_train.sum()),
                "n_negative": int((1 - y_train).sum()),
                **internal_best,
            }
        )

        for sample_id, score in cv_scores.items():
            prediction_rows.append(
                {
                    "method": method,
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

        full_transform = transform_for_method(
            method=method,
            x_train=train_matrix,
            x_eval=train_matrix,
            use_eval_reference=False,
        )
        model = build_model()
        model.fit(full_transform.x_train, y_train)

        for dataset_id in EVAL_DATASETS:
            test_meta = metadata.loc[metadata["dataset_id"] == dataset_id].copy()
            test_matrix = eval_matrices[dataset_id].loc[test_meta["sample_id"]]
            y_test = test_meta["binary_target"].astype(int)
            external_transform = transform_for_method(
                method=method,
                x_train=train_matrix,
                x_eval=test_matrix,
                use_eval_reference=(method == "cohort_robust_scale"),
            )
            scores = pd.Series(
                model.predict_proba(external_transform.x_eval)[:, 1],
                index=test_meta["sample_id"],
            )
            default_metrics = score_predictions(y_test, scores, threshold=0.5)
            train_thr_metrics = score_predictions(y_test, scores, threshold=best_threshold)

            summary_rows.append(
                {
                    "method": method,
                    "evaluation_type": "external",
                    "threshold_type": "default_0_5",
                    "threshold": 0.5,
                    "train_dataset": TRAIN_DATASET,
                    "test_dataset": dataset_id,
                    "n_samples": int(test_meta.shape[0]),
                    "n_positive": int(y_test.sum()),
                    "n_negative": int((1 - y_test).sum()),
                    **default_metrics,
                }
            )
            summary_rows.append(
                {
                    "method": method,
                    "evaluation_type": "external",
                    "threshold_type": "train_cv_optimal",
                    "threshold": best_threshold,
                    "train_dataset": TRAIN_DATASET,
                    "test_dataset": dataset_id,
                    "n_samples": int(test_meta.shape[0]),
                    "n_positive": int(y_test.sum()),
                    "n_negative": int((1 - y_test).sum()),
                    **train_thr_metrics,
                }
            )

            for sample_id, score in scores.items():
                prediction_rows.append(
                    {
                        "method": method,
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
