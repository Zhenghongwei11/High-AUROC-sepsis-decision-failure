#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a dataset-level summary from phenotype harmonization output."
    )
    parser.add_argument(
        "--harmonized",
        default="results/metadata/phenotype_harmonization.tsv",
        help="Input phenotype harmonization TSV",
    )
    parser.add_argument(
        "--out",
        default="results/dataset_summary.tsv",
        help="Output dataset summary TSV",
    )
    return parser.parse_args()


def task_flag(counts: Counter[str], positive: list[str], negative: list[str]) -> str:
    return "yes" if all(counts.get(x, 0) > 0 for x in positive + negative) else "no"


def main() -> int:
    args = parse_args()
    in_path = Path(args.harmonized)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with in_path.open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    by_dataset: dict[str, Counter[str]] = defaultdict(Counter)
    included_totals: Counter[str] = Counter()
    excluded_totals: Counter[str] = Counter()

    for row in rows:
        dataset_id = row["dataset_id"]
        label = row["harmonized_label"]
        include = row["include_in_mainline"] == "yes"
        if include:
            by_dataset[dataset_id][label] += 1
            included_totals[dataset_id] += 1
        else:
            excluded_totals[dataset_id] += 1

    with out_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "dataset_id",
            "included_samples",
            "excluded_samples",
            "n_healthy",
            "n_infection_non_sepsis",
            "n_sepsis",
            "n_septic_shock",
            "n_noninfectious_inflammation",
            "supports_task_a_sepsis_or_shock_vs_healthy",
            "supports_task_b_sepsis_vs_noninfectious_inflammation",
            "supports_task_c_cross_platform_transfer_candidate",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for dataset_id in sorted(set(included_totals) | set(excluded_totals)):
            counts = by_dataset[dataset_id]
            writer.writerow(
                {
                    "dataset_id": dataset_id,
                    "included_samples": included_totals[dataset_id],
                    "excluded_samples": excluded_totals[dataset_id],
                    "n_healthy": counts.get("healthy", 0),
                    "n_infection_non_sepsis": counts.get("infection_non_sepsis", 0),
                    "n_sepsis": counts.get("sepsis", 0),
                    "n_septic_shock": counts.get("septic_shock", 0),
                    "n_noninfectious_inflammation": counts.get(
                        "noninfectious_inflammation", 0
                    ),
                    "supports_task_a_sepsis_or_shock_vs_healthy": task_flag(
                        counts,
                        positive=["healthy"],
                        negative=["sepsis"],
                    )
                    if counts.get("sepsis", 0) > 0
                    else task_flag(
                        counts,
                        positive=["healthy"],
                        negative=["septic_shock"],
                    ),
                    "supports_task_b_sepsis_vs_noninfectious_inflammation": task_flag(
                        counts,
                        positive=["sepsis"],
                        negative=["noninfectious_inflammation"],
                    ),
                    "supports_task_c_cross_platform_transfer_candidate": (
                        "yes" if dataset_id == "GSE154918" else "no"
                    ),
                }
            )

    print(f"[wrote] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
