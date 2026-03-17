#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "data/manifest.tsv"
PHENO_SUMMARY_PATH = ROOT / "results/metadata/phenotype_harmonization_summary.tsv"
TASK_SUMMARY_PATH = ROOT / "results/tasks/task_dataset_summary.tsv"
FEATURE_SUMMARY_PATH = ROOT / "results/metadata/feature_harmonization_summary.tsv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a manuscript-facing cohort master summary table."
    )
    parser.add_argument(
        "--out",
        default="results/tables/cohort_master_summary.tsv",
        help="Output TSV path.",
    )
    return parser.parse_args()


def summarise_labels(pheno_summary: pd.DataFrame, include_flag: str) -> pd.DataFrame:
    subset = pheno_summary.loc[pheno_summary["include_in_mainline"] == include_flag].copy()
    if subset.empty:
        return pd.DataFrame(
            columns=["dataset_id", "label_summary", "sample_summary", "n_total"]
        )

    subset["label_pair"] = subset.apply(
        lambda row: f'{row["harmonized_label"]}={int(row["n_samples"])}',
        axis=1,
    )
    return (
        subset.groupby("dataset_id", as_index=False)
        .agg(
            label_summary=("harmonized_label", lambda values: ";".join(sorted(set(values)))),
            sample_summary=("label_pair", lambda values: ";".join(sorted(values))),
            n_total=("n_samples", "sum"),
        )
    )


def summarise_tasks(task_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str | int]] = []
    for dataset_id, group in task_summary.groupby("dataset_id"):
        task_entries = []
        role_entries = []
        for _, row in group.sort_values(["task_id", "dataset_role"]).iterrows():
            task_entries.append(
                f'{row["task_id"]}:{int(row["n_samples"])}'
                f'({int(row["n_positive"])}+/{int(row["n_negative"])}-)'
            )
            role_entries.append(f'{row["task_id"]}:{row["dataset_role"]}')
        rows.append(
            {
                "dataset_id": dataset_id,
                "task_membership": ";".join(task_entries),
                "task_roles": ";".join(role_entries),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()

    manifest = pd.read_csv(MANIFEST_PATH, sep="\t")
    pheno_summary = pd.read_csv(PHENO_SUMMARY_PATH, sep="\t")
    task_summary = pd.read_csv(TASK_SUMMARY_PATH, sep="\t")
    feature_summary = pd.read_csv(FEATURE_SUMMARY_PATH, sep="\t")

    included = summarise_labels(pheno_summary, "yes").rename(
        columns={
            "label_summary": "included_labels",
            "sample_summary": "included_label_counts",
            "n_total": "mainline_included_n",
        }
    )
    excluded = summarise_labels(pheno_summary, "no").rename(
        columns={
            "label_summary": "excluded_labels",
            "sample_summary": "excluded_label_counts",
            "n_total": "mainline_excluded_n",
        }
    )
    task_rollup = summarise_tasks(task_summary)

    table = (
        manifest.merge(
            feature_summary[
                [
                    "dataset_id",
                    "platform_id",
                    "output_genes",
                    "mapping_source",
                ]
            ],
            on="dataset_id",
            how="left",
        )
        .merge(included, on="dataset_id", how="left")
        .merge(excluded, on="dataset_id", how="left")
        .merge(task_rollup, on="dataset_id", how="left")
    )
    table = table.loc[table["status"] == "downloaded"].copy()

    table["mainline_included_n"] = table["mainline_included_n"].fillna(0).astype(int)
    table["mainline_excluded_n"] = table["mainline_excluded_n"].fillna(0).astype(int)
    table["included_labels"] = table["included_labels"].fillna("")
    table["included_label_counts"] = table["included_label_counts"].fillna("")
    table["excluded_labels"] = table["excluded_labels"].fillna("")
    table["excluded_label_counts"] = table["excluded_label_counts"].fillna("")
    table["task_membership"] = table["task_membership"].fillna("")
    table["task_roles"] = table["task_roles"].fillna("")
    table["output_genes"] = table["output_genes"].fillna(0).astype(int)
    table["mapping_source"] = table["mapping_source"].fillna("")

    role_map = {
        "primary_discovery_training": "primary_discovery",
        "external_validation": "external_validation",
        "cross_platform_validation": "cross_platform_validation",
        "inflammation_confounding_stress_test": "stress_test",
        "optional_extension": "optional_extension",
    }
    table["analysis_role"] = table["role"].map(role_map).fillna(table["role"])

    output = table[
        [
            "dataset_id",
            "title",
            "platform_id",
            "experiment_type",
            "sample_count",
            "mainline_included_n",
            "mainline_excluded_n",
            "included_labels",
            "included_label_counts",
            "excluded_labels",
            "excluded_label_counts",
            "analysis_role",
            "task_membership",
            "task_roles",
            "output_genes",
            "mapping_source",
            "source_page_url",
            "notes",
        ]
    ].rename(
        columns={
            "sample_count": "geo_series_samples",
            "title": "dataset_title",
        }
    )

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(out_path, sep="\t", index=False)
    print(f"[wrote] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
