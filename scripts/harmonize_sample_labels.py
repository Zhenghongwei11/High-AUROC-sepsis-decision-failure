#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


VALID_LABELS = {
    "healthy",
    "infection_non_sepsis",
    "sepsis",
    "septic_shock",
    "noninfectious_inflammation",
    "unassigned",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a conservative cross-dataset phenotype harmonization table."
    )
    parser.add_argument(
        "--sample-dir",
        default="data/processed/sample_metadata",
        help="Directory containing per-dataset sample metadata TSVs",
    )
    parser.add_argument(
        "--out",
        default="results/metadata/phenotype_harmonization.tsv",
        help="Output TSV path",
    )
    parser.add_argument(
        "--summary-out",
        default="results/metadata/phenotype_harmonization_summary.tsv",
        help="Output summary TSV path",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def extract_after_prefix(value: str) -> str:
    if ": " in value:
        return value.split(": ", 1)[1]
    return value


def harmonize_gse154918(row: dict[str, str]) -> tuple[str, bool, str, str]:
    status = extract_after_prefix(row.get("characteristics_ch1", ""))
    mapping = {
        "Hlty": ("healthy", True, "status=Hlty"),
        "Inf1_P": ("infection_non_sepsis", True, "status=Inf1_P"),
        "Seps_P": ("sepsis", True, "status=Seps_P"),
        "Shock_P": ("septic_shock", True, "status=Shock_P"),
        "Seps_FU": ("unassigned", False, "follow-up sample excluded: status=Seps_FU"),
        "Shock_FU": ("unassigned", False, "follow-up sample excluded: status=Shock_FU"),
    }
    label, include, rule = mapping.get(
        status, ("unassigned", False, f"unmapped status={status or 'missing'}")
    )
    return label, include, rule, status


def harmonize_gse28750(row: dict[str, str]) -> tuple[str, bool, str, str]:
    status = extract_after_prefix(row.get("characteristics_ch1_2", ""))
    mapping = {
        "HEALTHY": ("healthy", True, "health status=HEALTHY"),
        "POST_SURGICAL": (
            "noninfectious_inflammation",
            True,
            "health status=POST_SURGICAL",
        ),
        "SEPSIS": ("sepsis", True, "health status=SEPSIS"),
    }
    label, include, rule = mapping.get(
        status, ("unassigned", False, f"unmapped health status={status or 'missing'}")
    )
    return label, include, rule, status


def harmonize_gse95233(row: dict[str, str]) -> tuple[str, bool, str, str]:
    source = row.get("source_name_ch1", "")
    time_point = extract_after_prefix(row.get("characteristics_ch1_3", ""))
    if source.startswith("Control "):
        return "healthy", True, "source_name startswith Control", time_point
    if source.startswith("Patient "):
        include = time_point == "D01"
        rule = (
            "patient day1 septic shock sample"
            if include
            else f"follow-up septic shock sample excluded: time point={time_point}"
        )
        return "septic_shock", include, rule, time_point
    return "unassigned", False, f"unmapped source_name={source or 'missing'}", time_point


def harmonize_gse65682(row: dict[str, str]) -> tuple[str, bool, str, str]:
    title = row.get("title", "").lower()
    pneumonia = extract_after_prefix(row.get("characteristics_ch1_3", ""))
    abdominal = extract_after_prefix(row.get("characteristics_ch1_12", ""))

    if "healthy subject" in title:
        return "healthy", True, "title contains healthy subject", pneumonia
    if abdominal == "abdo_s":
        return "sepsis", True, "abdominal_sepsis_and_controls=abdo_s", abdominal
    if abdominal == "ctrl_GI":
        return (
            "noninfectious_inflammation",
            True,
            "abdominal_sepsis_and_controls=ctrl_GI",
            abdominal,
        )
    if pneumonia in {"cap", "hap"}:
        return "infection_non_sepsis", True, f"pneumonia diagnoses={pneumonia}", pneumonia
    if pneumonia == "no-cap":
        return "noninfectious_inflammation", True, "pneumonia diagnoses=no-cap", pneumonia
    return "unassigned", False, "no conservative mapping rule matched", pneumonia or abdominal


def harmonize_row(dataset_id: str, row: dict[str, str]) -> tuple[str, bool, str, str]:
    if dataset_id == "GSE154918":
        return harmonize_gse154918(row)
    if dataset_id == "GSE28750":
        return harmonize_gse28750(row)
    if dataset_id == "GSE95233":
        return harmonize_gse95233(row)
    if dataset_id == "GSE65682":
        return harmonize_gse65682(row)
    return "unassigned", False, "dataset-specific rule missing", ""


def dataset_id_from_path(path: Path) -> str:
    name = path.name
    if not name.endswith("_samples.tsv"):
        raise ValueError(f"Unexpected sample metadata filename: {path}")
    return name[: -len("_samples.tsv")]


def main() -> int:
    args = parse_args()
    sample_dir = Path(args.sample_dir)
    out_path = Path(args.out)
    summary_path = Path(args.summary_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, str]] = []
    summary_counter: dict[tuple[str, str, str], int] = defaultdict(int)

    for sample_path in sorted(sample_dir.glob("*_samples.tsv")):
        dataset_id = dataset_id_from_path(sample_path)
        for row in read_rows(sample_path):
            label, include, rule, raw_group = harmonize_row(dataset_id, row)
            if label not in VALID_LABELS:
                raise ValueError(f"Invalid label {label!r} for {dataset_id}")
            record = {
                "dataset_id": dataset_id,
                "geo_accession": row.get("geo_accession", ""),
                "sample_title": row.get("title", ""),
                "raw_group_signal": raw_group,
                "harmonized_label": label,
                "include_in_mainline": "yes" if include else "no",
                "mapping_rule": rule,
            }
            records.append(record)
            summary_counter[(dataset_id, label, "yes" if include else "no")] += 1

    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset_id",
                "geo_accession",
                "sample_title",
                "raw_group_signal",
                "harmonized_label",
                "include_in_mainline",
                "mapping_rule",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(records)

    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dataset_id", "harmonized_label", "include_in_mainline", "n_samples"],
            delimiter="\t",
        )
        writer.writeheader()
        for (dataset_id, label, include), count in sorted(summary_counter.items()):
            writer.writerow(
                {
                    "dataset_id": dataset_id,
                    "harmonized_label": label,
                    "include_in_mainline": include,
                    "n_samples": count,
                }
            )

    label_counts = Counter(record["harmonized_label"] for record in records)
    print(f"[wrote] {out_path} ({len(records)} rows)")
    print(f"[wrote] {summary_path}")
    print("[labels]", ", ".join(f"{k}={v}" for k, v in sorted(label_counts.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
