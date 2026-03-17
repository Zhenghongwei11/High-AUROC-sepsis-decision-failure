#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export included samples from phenotype harmonization output."
    )
    parser.add_argument(
        "--harmonized",
        default="results/metadata/phenotype_harmonization.tsv",
        help="Input harmonized phenotype TSV",
    )
    parser.add_argument(
        "--out",
        default="results/metadata/sample_inclusion.tsv",
        help="Output TSV containing only included samples",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    in_path = Path(args.harmonized)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with in_path.open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    included = [row for row in rows if row["include_in_mainline"] == "yes"]

    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=included[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(included)

    print(f"[wrote] {out_path} ({len(included)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
