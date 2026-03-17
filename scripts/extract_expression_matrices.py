#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import gzip
import shlex
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract included-sample expression matrices from GEO series matrix files."
    )
    parser.add_argument(
        "--manifest",
        default="data/manifest.tsv",
        help="Manifest TSV with dataset_id and local_relpath",
    )
    parser.add_argument(
        "--sample-inclusion",
        default="results/metadata/sample_inclusion.tsv",
        help="Included sample TSV produced by export_sample_inclusion.py",
    )
    parser.add_argument(
        "--outdir",
        default="data/processed/expression",
        help="Output directory for expression matrices",
    )
    parser.add_argument(
        "dataset_ids",
        nargs="*",
        help="Optional subset of dataset IDs to extract",
    )
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def build_manifest_index(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["dataset_id"]: row for row in rows}


def build_inclusion_index(rows: list[dict[str, str]]) -> dict[str, set[str]]:
    included: dict[str, set[str]] = {}
    for row in rows:
        dataset_id = row["dataset_id"]
        sample_id = row["sample_title"] if dataset_id == "GSE154918" else row["geo_accession"]
        included.setdefault(dataset_id, set()).add(sample_id)
    return included


def clean_fields(line: str) -> list[str]:
    return next(csv.reader([line], delimiter="\t", quotechar='"'))


def output_path(outdir: Path, dataset_id: str) -> Path:
    return outdir / f"{dataset_id}_expression_included.tsv.gz"


def extract_supplement_matrix(
    matrix_path: Path, included_samples: set[str], out_path: Path
) -> tuple[int, int]:
    selected_indices: list[int] | None = None
    sample_count = 0
    row_count = 0

    with gzip.open(matrix_path, "rt", encoding="utf-8", errors="replace") as src, gzip.open(
        out_path, "wt", encoding="utf-8", newline=""
    ) as dst:
        writer = csv.writer(dst, delimiter="\t", lineterminator="\n")

        for raw_line in src:
            line = raw_line.rstrip("\n")
            if not line:
                continue

            fields = shlex.split(line)
            if selected_indices is None:
                sample_start = None
                for idx, value in enumerate(fields):
                    if value in included_samples:
                        sample_start = idx
                        break
                if sample_start is None:
                    raise ValueError(f"No included sample columns found in {matrix_path}")
                metadata_indices = list(range(sample_start))
                sample_indices = [
                    idx for idx, value in enumerate(fields[sample_start:], start=sample_start)
                    if value in included_samples
                ]
                sample_count = len(sample_indices)
                selected_indices = metadata_indices + sample_indices
                writer.writerow([fields[idx] for idx in selected_indices])
                continue

            row_count += 1
            writer.writerow([fields[idx] if idx < len(fields) else "" for idx in selected_indices])

    if selected_indices is None:
        raise ValueError(f"Failed to parse supplement matrix header in {matrix_path}")

    return sample_count, row_count


def extract_dataset(
    dataset_id: str, matrix_path: Path, included_samples: set[str], outdir: Path
) -> tuple[Path, int, int]:
    out_path = output_path(outdir, dataset_id)
    outdir.mkdir(parents=True, exist_ok=True)

    if matrix_path.name == "GSE154918_Schughart_Sepsis_200320.txt.gz":
        sample_count, row_count = extract_supplement_matrix(
            matrix_path=matrix_path,
            included_samples=included_samples,
            out_path=out_path,
        )
        return out_path, sample_count, row_count

    selected_indices: list[int] | None = None
    selected_header: list[str] | None = None
    row_count = 0

    with gzip.open(matrix_path, "rt", encoding="utf-8", errors="replace") as src, gzip.open(
        out_path, "wt", encoding="utf-8", newline=""
    ) as dst:
        writer = csv.writer(dst, delimiter="\t", lineterminator="\n")

        in_table = False
        for raw_line in src:
            line = raw_line.rstrip("\n")

            if line == "!series_matrix_table_begin":
                in_table = True
                continue
            if line == "!series_matrix_table_end":
                break
            if not in_table:
                continue

            fields = clean_fields(line)
            if selected_indices is None:
                header = fields
                selected_indices = [0]
                for idx, sample_id in enumerate(header[1:], start=1):
                    if sample_id in included_samples:
                        selected_indices.append(idx)
                selected_header = [header[idx] for idx in selected_indices]
                writer.writerow(selected_header)
                continue

            row_count += 1
            writer.writerow([fields[idx] if idx < len(fields) else "" for idx in selected_indices])

    if selected_indices is None or selected_header is None:
        raise ValueError(f"Failed to locate matrix table in {matrix_path}")

    return out_path, len(selected_header) - 1, row_count


def main() -> int:
    args = parse_args()
    root = Path(".")
    manifest = build_manifest_index(read_tsv(root / args.manifest))
    inclusion = build_inclusion_index(read_tsv(root / args.sample_inclusion))
    outdir = root / args.outdir

    dataset_ids = args.dataset_ids or sorted(inclusion)

    for dataset_id in dataset_ids:
        if dataset_id not in manifest:
            raise ValueError(f"Dataset {dataset_id} not found in manifest")
        if dataset_id not in inclusion:
            raise ValueError(f"Dataset {dataset_id} has no included samples")
        matrix_path = root / manifest[dataset_id]["local_relpath"]
        out_path, sample_count, feature_count = extract_dataset(
            dataset_id=dataset_id,
            matrix_path=matrix_path,
            included_samples=inclusion[dataset_id],
            outdir=outdir,
        )
        print(
            f"[wrote] {out_path} "
            f"(samples={sample_count}, features={feature_count})"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
