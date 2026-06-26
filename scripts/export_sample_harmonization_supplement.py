#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


TASK_IDS = ["task_a", "task_b", "task_c"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export sample-level harmonization supplement for peer review."
    )
    parser.add_argument(
        "--harmonization",
        default="results/metadata/phenotype_harmonization.tsv",
        help="Phenotype harmonization TSV.",
    )
    parser.add_argument(
        "--task-dir",
        default="results/tasks",
        help="Directory containing task sample tables.",
    )
    parser.add_argument(
        "--out",
        default="results/tables/sample_level_harmonization_supplement.tsv",
        help="Output TSV path.",
    )
    return parser.parse_args()


def infer_timepoint(row: pd.Series) -> str:
    text = " ".join(
        str(row.get(column, ""))
        for column in ["raw_group_signal", "mapping_rule", "sample_title"]
        if pd.notna(row.get(column, ""))
    ).lower()
    if "day-1" in text or "day 1" in text or "d1" in text:
        return "baseline_day_1"
    if "follow" in text or "day-3" in text or "day 3" in text or "d3" in text:
        return "follow_up_or_nonbaseline"
    return "not_available_or_not_applicable"


def inclusion_reason(row: pd.Series) -> str:
    if row["include_in_mainline"] == "yes":
        return "included_by_conservative_harmonization_rule"
    if row["harmonized_label"] == "unassigned":
        return "excluded_no_defensible_harmonized_label"
    return "excluded_not_in_mainline_task_scope"


def build_task_membership(task_dir: Path) -> pd.DataFrame:
    frames = []
    for task_id in TASK_IDS:
        path = task_dir / f"{task_id}_samples.tsv"
        task = pd.read_csv(path, sep="\t")
        frames.append(
            task[
                [
                    "dataset_id",
                    "geo_accession",
                    "sample_title",
                    "task_id",
                    "task_name",
                    "dataset_role",
                    "binary_target",
                ]
            ].copy()
        )
    tasks = pd.concat(frames, ignore_index=True)
    tasks["task_membership_entry"] = (
        tasks["task_id"]
        + ":"
        + tasks["task_name"]
        + ":"
        + tasks["dataset_role"]
        + ":target="
        + tasks["binary_target"].astype(str)
    )
    grouped = (
        tasks.groupby(["dataset_id", "geo_accession", "sample_title"], dropna=False)[
            "task_membership_entry"
        ]
        .apply(lambda values: ";".join(sorted(values)))
        .reset_index()
        .rename(columns={"task_membership_entry": "analytical_task_membership"})
    )
    return grouped


def main() -> int:
    args = parse_args()
    harmonization = pd.read_csv(args.harmonization, sep="\t")
    task_membership = build_task_membership(Path(args.task_dir))
    out = harmonization.merge(
        task_membership,
        on=["dataset_id", "geo_accession", "sample_title"],
        how="left",
    )
    out["analytical_task_membership"] = out["analytical_task_membership"].fillna("not_included_in_analysis_tasks")
    out["clinical_day_or_timepoint"] = out.apply(infer_timepoint, axis=1)
    out["inclusion_exclusion_reason"] = out.apply(inclusion_reason, axis=1)
    out = out[
        [
            "dataset_id",
            "geo_accession",
            "sample_title",
            "raw_group_signal",
            "clinical_day_or_timepoint",
            "harmonized_label",
            "include_in_mainline",
            "inclusion_exclusion_reason",
            "mapping_rule",
            "analytical_task_membership",
        ]
    ].sort_values(["dataset_id", "geo_accession", "sample_title"])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, sep="\t", index=False)
    print(f"[wrote] {out_path} ({out.shape[0]} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
