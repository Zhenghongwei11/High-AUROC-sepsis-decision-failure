from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results" / "tables" / "cohort_master_summary.tsv"
OUTPUT = ROOT / "results" / "tables" / "cohort_flow_summary.tsv"


def parse_task_membership(raw: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not raw:
        return counts
    for item in raw.split(";"):
        item = item.strip()
        if not item or ":" not in item:
            continue
        task_id, payload = item.split(":", 1)
        total = payload.split("(", 1)[0].strip()
        try:
            counts[task_id.strip()] = int(total)
        except ValueError:
            continue
    return counts


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with INPUT.open("r", encoding="utf-8", newline="") as infile:
        reader = csv.DictReader(infile, delimiter="\t")
        rows = list(reader)

    fieldnames = [
        "dataset_id",
        "dataset_title",
        "geo_series_samples",
        "excluded_after_harmonization",
        "mainline_included",
        "task_a_included",
        "task_b_included",
        "task_c_included",
        "analysis_role",
        "notes",
    ]

    with OUTPUT.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            task_counts = parse_task_membership(row.get("task_membership", ""))
            writer.writerow(
                {
                    "dataset_id": row["dataset_id"],
                    "dataset_title": row["dataset_title"],
                    "geo_series_samples": row["geo_series_samples"],
                    "excluded_after_harmonization": row["mainline_excluded_n"],
                    "mainline_included": row["mainline_included_n"],
                    "task_a_included": task_counts.get("task_a", 0),
                    "task_b_included": task_counts.get("task_b", 0),
                    "task_c_included": task_counts.get("task_c", 0),
                    "analysis_role": row["analysis_role"],
                    "notes": row["notes"],
                }
            )


if __name__ == "__main__":
    main()
