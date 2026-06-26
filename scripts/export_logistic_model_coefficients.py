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
    METHODS,
    TRAIN_DATASET,
    build_model,
    load_task_matrix,
    transform_for_method,
)


ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_TASKS = ["task_a", "task_b"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export full-model logistic regression coefficients for benchmark tasks."
    )
    parser.add_argument(
        "--out-dir",
        default="results/models",
        help="Directory for coefficient exports.",
    )
    return parser.parse_args()


def load_training_data(task_id: str) -> tuple[pd.DataFrame, pd.Series]:
    metadata = pd.read_csv(ROOT / "results/tasks" / f"{task_id}_samples.tsv", sep="\t")
    train_meta = metadata.loc[metadata["dataset_id"] == TRAIN_DATASET].copy()
    matrix = load_task_matrix(ROOT / "data/processed/task_matrices" / task_id, TRAIN_DATASET)
    x_train = matrix.loc[train_meta["sample_id"]]
    y_train = train_meta["binary_target"].astype(int)
    return x_train, y_train


def main() -> int:
    args = parse_args()
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, str | int | float]] = []
    for task_id in SUPPORTED_TASKS:
        x_train, y_train = load_training_data(task_id)
        for method in METHODS:
            transformed = transform_for_method(
                method=method,
                x_train=x_train,
                x_eval=x_train,
                use_eval_reference=False,
            )
            model = build_model()
            model.fit(transformed.x_train, y_train)

            coef_df = pd.DataFrame(
                {
                    "task_id": task_id,
                    "train_dataset": TRAIN_DATASET,
                    "method": method,
                    "gene_symbol": transformed.x_train.columns,
                    "coefficient": model.coef_.ravel(),
                }
            )
            coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
            coef_df = coef_df.sort_values("abs_coefficient", ascending=False)

            out_path = out_dir / f"{task_id}_{method}_coefficients.tsv.gz"
            coef_df.to_csv(out_path, sep="\t", index=False, compression="gzip")

            manifest_rows.append(
                {
                    "task_id": task_id,
                    "task_name": loader.TASK_DEFINITIONS[task_id]["task_name"],
                    "train_dataset": TRAIN_DATASET,
                    "method": method,
                    "n_features": int(transformed.x_train.shape[1]),
                    "n_samples": int(transformed.x_train.shape[0]),
                    "intercept": float(model.intercept_[0]),
                    "coefficient_relpath": str(out_path.relative_to(ROOT)),
                }
            )

    manifest = pd.DataFrame(manifest_rows).sort_values(["task_id", "method"])
    manifest_path = out_dir / "logistic_model_manifest.tsv"
    manifest.to_csv(manifest_path, sep="\t", index=False)
    print(f"[wrote] {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
