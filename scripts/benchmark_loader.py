#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA = ROOT / "results/metadata/sample_inclusion.tsv"
DEFAULT_EXPRESSION_DIR = ROOT / "data/processed/expression_gene"
TASK_DEFINITIONS = {
    "task_a": {
        "task_name": "sepsis_or_shock_vs_healthy",
        "dataset_ids": ["GSE65682", "GSE95233", "GSE154918", "GSE28750"],
        "positive_labels": {"sepsis", "septic_shock"},
        "negative_labels": {"healthy"},
        "dataset_roles": {
            "GSE65682": "primary_discovery",
            "GSE95233": "external_validation",
            "GSE154918": "cross_platform_validation",
            "GSE28750": "stress_test_support",
        },
    },
    "task_b": {
        "task_name": "sepsis_vs_noninfectious_inflammation",
        "dataset_ids": ["GSE65682", "GSE28750"],
        "positive_labels": {"sepsis"},
        "negative_labels": {"noninfectious_inflammation"},
        "dataset_roles": {
            "GSE65682": "primary_discovery",
            "GSE28750": "confounding_stress_test",
        },
    },
    "task_c": {
        "task_name": "microarray_to_rnaseq_transfer",
        "dataset_ids": ["GSE65682", "GSE95233", "GSE154918", "GSE28750"],
        "positive_labels": {"sepsis", "septic_shock"},
        "negative_labels": {"healthy"},
        "dataset_roles": {
            "GSE65682": "microarray_discovery",
            "GSE95233": "microarray_external",
            "GSE154918": "rnaseq_validation",
            "GSE28750": "microarray_support",
        },
    },
}


def load_included_metadata(path: str | Path = DEFAULT_METADATA) -> pd.DataFrame:
    metadata = pd.read_csv(path, sep="\t")
    metadata["sample_id"] = metadata.apply(
        lambda row: row["sample_title"] if row["dataset_id"] == "GSE154918" else row["geo_accession"],
        axis=1,
    )
    return metadata


def load_gene_matrix(
    dataset_id: str,
    expression_dir: str | Path = DEFAULT_EXPRESSION_DIR,
    transpose: bool = False,
) -> pd.DataFrame:
    matrix_path = Path(expression_dir) / f"{dataset_id}_gene_expression.tsv.gz"
    matrix = pd.read_csv(matrix_path, sep="\t", compression="gzip")
    if "gene_symbol" not in matrix.columns:
        raise ValueError(f"{matrix_path} is missing gene_symbol column")
    matrix = matrix.set_index("gene_symbol")
    return matrix.T if transpose else matrix


def load_dataset(
    dataset_id: str,
    expression_dir: str | Path = DEFAULT_EXPRESSION_DIR,
    metadata_path: str | Path = DEFAULT_METADATA,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = load_included_metadata(metadata_path)
    metadata = metadata.loc[metadata["dataset_id"] == dataset_id].copy()
    matrix = load_gene_matrix(dataset_id=dataset_id, expression_dir=expression_dir, transpose=True)
    metadata = metadata.set_index("sample_id").loc[matrix.index].reset_index()
    return matrix, metadata


def shared_gene_space(
    dataset_ids: list[str],
    expression_dir: str | Path = DEFAULT_EXPRESSION_DIR,
) -> list[str]:
    shared: set[str] | None = None
    for dataset_id in dataset_ids:
        genes = set(load_gene_matrix(dataset_id, expression_dir=expression_dir).index)
        shared = genes if shared is None else shared & genes
    return sorted(shared or [])


def build_task_metadata(
    task_id: str,
    metadata_path: str | Path = DEFAULT_METADATA,
) -> pd.DataFrame:
    if task_id not in TASK_DEFINITIONS:
        raise ValueError(f"Unknown task_id: {task_id}")
    task = TASK_DEFINITIONS[task_id]
    metadata = load_included_metadata(metadata_path)
    metadata = metadata.loc[metadata["dataset_id"].isin(task["dataset_ids"])].copy()
    keep_labels = task["positive_labels"] | task["negative_labels"]
    metadata = metadata.loc[metadata["harmonized_label"].isin(keep_labels)].copy()
    metadata["task_id"] = task_id
    metadata["task_name"] = task["task_name"]
    metadata["dataset_role"] = metadata["dataset_id"].map(task["dataset_roles"])
    metadata["binary_target"] = metadata["harmonized_label"].isin(task["positive_labels"]).astype(int)
    return metadata


def task_shared_gene_space(
    task_id: str,
    expression_dir: str | Path = DEFAULT_EXPRESSION_DIR,
) -> list[str]:
    if task_id not in TASK_DEFINITIONS:
        raise ValueError(f"Unknown task_id: {task_id}")
    return shared_gene_space(TASK_DEFINITIONS[task_id]["dataset_ids"], expression_dir=expression_dir)
