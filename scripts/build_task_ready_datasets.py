#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import benchmark_loader as loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build task-ready sample tables, shared gene lists, and matrices."
    )
    parser.add_argument(
        "--task-dir",
        default="results/tasks",
        help="Directory for task metadata and summaries",
    )
    parser.add_argument(
        "--matrix-dir",
        default="data/processed/task_matrices",
        help="Directory for per-task sample-by-gene matrices",
    )
    parser.add_argument(
        "--expression-dir",
        default="data/processed/expression_gene",
        help="Directory containing gene-level matrices",
    )
    parser.add_argument(
        "--metadata",
        default="results/metadata/sample_inclusion.tsv",
        help="Included-sample metadata TSV",
    )
    parser.add_argument(
        "task_ids",
        nargs="*",
        help="Optional subset of task IDs",
    )
    return parser.parse_args()


def write_task_gene_list(genes: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(genes) + "\n", encoding="utf-8")


def write_task_matrix(matrix: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(path, sep="\t", compression="gzip")


def build_task_outputs(
    task_id: str,
    task_dir: Path,
    matrix_dir: Path,
    expression_dir: Path,
    metadata_path: Path,
) -> tuple[list[dict[str, str | int]], dict[str, str | int]]:
    metadata = loader.build_task_metadata(task_id=task_id, metadata_path=metadata_path)
    shared_genes = loader.task_shared_gene_space(task_id=task_id, expression_dir=expression_dir)
    task = loader.TASK_DEFINITIONS[task_id]

    sample_out = task_dir / f"{task_id}_samples.tsv"
    gene_out = task_dir / f"{task_id}_genes.txt"
    task_dir.mkdir(parents=True, exist_ok=True)
    matrix_dir.mkdir(parents=True, exist_ok=True)
    metadata.sort_values(["dataset_id", "binary_target", "sample_id"]).to_csv(
        sample_out,
        sep="\t",
        index=False,
    )
    write_task_gene_list(shared_genes, gene_out)

    summary_rows: list[dict[str, str | int]] = []
    for dataset_id in task["dataset_ids"]:
        dataset_meta = metadata.loc[metadata["dataset_id"] == dataset_id].copy()
        if dataset_meta.empty:
            continue
        matrix = loader.load_gene_matrix(dataset_id, expression_dir=expression_dir, transpose=True)
        sample_order = dataset_meta["sample_id"].tolist()
        dataset_matrix = matrix.loc[sample_order, shared_genes]
        dataset_matrix.index.name = "sample_id"
        matrix_out = matrix_dir / task_id / f"{dataset_id}_samples_x_genes.tsv.gz"
        write_task_matrix(dataset_matrix, matrix_out)

        summary_rows.append(
            {
                "task_id": task_id,
                "task_name": task["task_name"],
                "dataset_id": dataset_id,
                "dataset_role": task["dataset_roles"][dataset_id],
                "n_samples": int(dataset_meta.shape[0]),
                "n_positive": int(dataset_meta["binary_target"].sum()),
                "n_negative": int((1 - dataset_meta["binary_target"]).sum()),
                "shared_genes": len(shared_genes),
                "matrix_relpath": str(matrix_out),
            }
        )

    task_summary = {
        "task_id": task_id,
        "task_name": task["task_name"],
        "n_datasets": len(summary_rows),
        "n_samples": int(metadata.shape[0]),
        "n_positive": int(metadata["binary_target"].sum()),
        "n_negative": int((1 - metadata["binary_target"]).sum()),
        "shared_genes": len(shared_genes),
        "sample_relpath": str(sample_out),
        "gene_relpath": str(gene_out),
    }
    return summary_rows, task_summary


def main() -> int:
    args = parse_args()
    root = Path(".")
    task_dir = root / args.task_dir
    matrix_dir = root / args.matrix_dir
    expression_dir = root / args.expression_dir
    metadata_path = root / args.metadata

    task_ids = args.task_ids or sorted(loader.TASK_DEFINITIONS)
    dataset_rows: list[dict[str, str | int]] = []
    task_rows: list[dict[str, str | int]] = []

    for task_id in task_ids:
        summary_rows, task_summary = build_task_outputs(
            task_id=task_id,
            task_dir=task_dir,
            matrix_dir=matrix_dir,
            expression_dir=expression_dir,
            metadata_path=metadata_path,
        )
        dataset_rows.extend(summary_rows)
        task_rows.append(task_summary)
        print(
            f"[wrote] {task_id} "
            f"(datasets={task_summary['n_datasets']}, samples={task_summary['n_samples']}, "
            f"shared_genes={task_summary['shared_genes']})"
        )

    task_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dataset_rows).to_csv(task_dir / "task_dataset_summary.tsv", sep="\t", index=False)
    pd.DataFrame(task_rows).to_csv(task_dir / "task_summary.tsv", sep="\t", index=False)
    print(f"[wrote] {task_dir / 'task_dataset_summary.tsv'}")
    print(f"[wrote] {task_dir / 'task_summary.tsv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
