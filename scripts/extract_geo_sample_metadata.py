#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import gzip
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract !Sample_* metadata rows from GEO series_matrix.txt.gz files."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="One or more GEO series_matrix.txt.gz files",
    )
    parser.add_argument(
        "--outdir",
        default="data/processed/sample_metadata",
        help="Directory for output TSV files",
    )
    return parser.parse_args()


def clean_field(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        value = value[1:-1]
    return value


def parse_matrix(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    sample_rows: dict[str, list[str]] = {}
    key_counts: dict[str, int] = {}
    sample_count = 0

    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if line == "!series_matrix_table_begin":
                break
            if not line.startswith("!Sample_"):
                continue

            fields = line.split("\t")
            base_key = fields[0][len("!Sample_") :]
            count = key_counts.get(base_key, 0) + 1
            key_counts[base_key] = count
            key = base_key if count == 1 else f"{base_key}_{count}"
            values = [clean_field(field) for field in fields[1:]]
            sample_rows[key] = values
            sample_count = max(sample_count, len(values))

    if sample_count == 0:
        raise ValueError(f"No !Sample_* rows found in {path}")

    ordered_keys: list[str] = []
    for preferred in ("geo_accession", "title"):
        if preferred in sample_rows:
            ordered_keys.append(preferred)
    for key in sample_rows:
        if key not in ordered_keys:
            ordered_keys.append(key)

    records: list[dict[str, str]] = []
    for idx in range(sample_count):
        record = {}
        for key in ordered_keys:
            values = sample_rows.get(key, [])
            record[key] = values[idx] if idx < len(values) else ""
        records.append(record)

    return ordered_keys, records


def output_path(outdir: Path, input_path: Path) -> Path:
    name = input_path.name
    if name.endswith("_series_matrix.txt.gz"):
        stem = name[: -len("_series_matrix.txt.gz")]
    else:
        stem = input_path.stem
    return outdir / f"{stem}_samples.tsv"


def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for input_name in args.inputs:
        input_path = Path(input_name)
        fieldnames, records = parse_matrix(input_path)
        out_path = output_path(outdir, input_path)
        with out_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            writer.writerows(records)
        print(f"[wrote] {out_path} ({len(records)} samples)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
