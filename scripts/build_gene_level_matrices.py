#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import gzip
import io
import re
from pathlib import Path

import pandas as pd
import requests


PLATFORM_URLS = {
    "GPL570": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GPL570&targ=self&view=data&form=text",
    "GPL13667": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GPL13667&targ=self&view=data&form=text",
}

INVALID_GENE_TOKENS = {"", "---", "NA", "N/A", "NULL", "null", "nan"}
GENE_SPLIT_PATTERN = re.compile(r"\s*///\s*|\s*//\s*|\s*;\s*|\s*,\s*|\s*\|\s*")
ANNOTATION_COLUMNS = {
    "Row.names",
    "ENSEMBL_gene_ID",
    "ensembl_transcript_id",
    "gene_symbol",
    "entrezgene_id",
    "uniprotswissprot",
    "description",
    "refseq_mrna",
    "transcript_length",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create conservative gene-level matrices across array and RNA-seq cohorts."
    )
    parser.add_argument(
        "--expression-dir",
        default="data/processed/expression",
        help="Directory containing included-sample expression matrices",
    )
    parser.add_argument(
        "--sample-dir",
        default="data/processed/sample_metadata",
        help="Directory containing per-dataset sample metadata TSVs",
    )
    parser.add_argument(
        "--sample-inclusion",
        default="results/metadata/sample_inclusion.tsv",
        help="Included sample table used to recover expected sample identifiers",
    )
    parser.add_argument(
        "--platform-dir",
        default="data/raw/platforms",
        help="Directory for downloaded GEO platform annotation tables",
    )
    parser.add_argument(
        "--feature-map-dir",
        default="data/processed/feature_maps",
        help="Directory for processed probe-to-gene maps",
    )
    parser.add_argument(
        "--outdir",
        default="data/processed/expression_gene",
        help="Directory for gene-level expression matrices",
    )
    parser.add_argument(
        "--summary-out",
        default="results/metadata/feature_harmonization_summary.tsv",
        help="Output TSV summarizing mapping and collapse results",
    )
    parser.add_argument(
        "dataset_ids",
        nargs="*",
        help="Optional subset of dataset IDs to process",
    )
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def normalize_gene_symbol(raw_value: str) -> tuple[str | None, str]:
    raw_value = (raw_value or "").strip().strip('"')
    if raw_value in INVALID_GENE_TOKENS:
        return None, "blank_gene_symbol"

    tokens = []
    for token in GENE_SPLIT_PATTERN.split(raw_value):
        token = token.strip()
        if not token or token in INVALID_GENE_TOKENS:
            continue
        tokens.append(token)

    unique_tokens = sorted(set(tokens))
    if not unique_tokens:
        return None, "blank_gene_symbol"
    if len(unique_tokens) > 1:
        return None, "ambiguous_gene_symbol"
    return unique_tokens[0], "mapped_unique"


def sample_id_column(dataset_id: str) -> str:
    return "sample_title" if dataset_id == "GSE154918" else "geo_accession"


def build_expected_samples(rows: list[dict[str, str]]) -> dict[str, list[str]]:
    dataset_to_samples: dict[str, list[str]] = {}
    for row in rows:
        dataset_id = row["dataset_id"]
        dataset_to_samples.setdefault(dataset_id, []).append(row[sample_id_column(dataset_id)])
    return dataset_to_samples


def build_platform_index(sample_dir: Path) -> dict[str, str]:
    platform_index: dict[str, str] = {}
    for path in sorted(sample_dir.glob("*_samples.tsv")):
        rows = read_tsv(path)
        dataset_id = path.name.removesuffix("_samples.tsv")
        platform_ids = {row["platform_id"] for row in rows if row.get("platform_id")}
        if len(platform_ids) > 1:
            raise ValueError(f"{dataset_id} has multiple platform IDs: {sorted(platform_ids)}")
        platform_index[dataset_id] = next(iter(platform_ids), "")
    return platform_index


def download_platform_annotation(platform_id: str, platform_dir: Path) -> Path:
    if platform_id not in PLATFORM_URLS:
        raise ValueError(f"Unsupported platform {platform_id}")
    out_path = platform_dir / f"{platform_id}_annotation.txt.gz"
    if out_path.exists() and out_path.stat().st_size > 1024 and gzip_is_complete(out_path):
        return out_path
    out_path.unlink(missing_ok=True)

    platform_dir.mkdir(parents=True, exist_ok=True)
    response = requests.get(
        PLATFORM_URLS[platform_id],
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=(15, 120),
        stream=True,
    )
    response.raise_for_status()

    preview = b""
    with gzip.open(out_path, "wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 128):
            if not chunk:
                continue
            if len(preview) < 120:
                preview += chunk[: 120 - len(preview)]
            handle.write(chunk)

    if not preview.startswith(b"^PLATFORM"):
        out_path.unlink(missing_ok=True)
        preview_text = preview.decode("utf-8", errors="replace")
        raise RuntimeError(f"Unexpected response while downloading {platform_id}: {preview_text}")
    return out_path


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def gzip_is_complete(path: Path) -> bool:
    try:
        with gzip.open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                if not chunk:
                    break
        return True
    except (OSError, EOFError):
        return False


def parse_platform_annotation(path: Path, platform_id: str) -> pd.DataFrame:
    header: list[str] | None = None
    rows: list[dict[str, str]] = []
    in_table = False

    with open_text(path) as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if line == "!platform_table_begin":
                in_table = True
                continue
            if line == "!platform_table_end":
                break
            if not in_table:
                continue
            fields = next(csv.reader([line], delimiter="\t"))
            if header is None:
                header = fields
                continue
            row = dict(zip(header, fields))
            probe_id = row.get("ID", "").strip()
            raw_gene_symbol = row.get("Gene Symbol", "")
            gene_symbol, status = normalize_gene_symbol(raw_gene_symbol)
            rows.append(
                {
                    "platform_id": platform_id,
                    "probe_id": probe_id,
                    "raw_gene_symbol": raw_gene_symbol,
                    "gene_symbol": gene_symbol or "",
                    "mapping_status": status,
                }
            )

    if header is None:
        raise RuntimeError(f"Failed to parse platform table from {path}")
    return pd.DataFrame(rows)


def write_feature_map(platform_df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    platform_df.to_csv(out_path, sep="\t", index=False)


def collapse_array_dataset(
    dataset_id: str,
    matrix_path: Path,
    platform_id: str,
    feature_map: pd.DataFrame,
    expected_samples: list[str],
    outdir: Path,
) -> dict[str, str | int]:
    df = pd.read_csv(matrix_path, sep="\t", compression="gzip", dtype=str)
    if "ID_REF" not in df.columns:
        raise ValueError(f"{dataset_id} is missing ID_REF column")

    sample_cols = [sample for sample in expected_samples if sample in df.columns]
    if not sample_cols:
        raise ValueError(f"{dataset_id} has no expected sample columns in {matrix_path}")

    status_lookup = feature_map.set_index("probe_id")["mapping_status"].to_dict()
    gene_lookup = feature_map.set_index("probe_id")["gene_symbol"].to_dict()

    probe_status = df["ID_REF"].map(status_lookup).fillna("probe_not_in_platform")
    df["gene_symbol"] = df["ID_REF"].map(gene_lookup)
    valid = df["gene_symbol"].fillna("") != ""

    value_frame = df.loc[valid, sample_cols].apply(pd.to_numeric, errors="coerce")
    collapsed = value_frame.groupby(df.loc[valid, "gene_symbol"]).median().sort_index()
    collapsed.index.name = "gene_symbol"

    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"{dataset_id}_gene_expression.tsv.gz"
    with gzip.open(out_path, "wt", encoding="utf-8", newline="") as handle:
        collapsed.reset_index().to_csv(handle, sep="\t", index=False)

    status_counts = probe_status.value_counts().to_dict()
    return {
        "dataset_id": dataset_id,
        "platform_id": platform_id,
        "mapping_source": "geo_platform_gene_symbol",
        "input_features": int(df.shape[0]),
        "mapped_feature_rows": int(valid.sum()),
        "blank_feature_rows": int(status_counts.get("blank_gene_symbol", 0)),
        "ambiguous_feature_rows": int(status_counts.get("ambiguous_gene_symbol", 0)),
        "missing_platform_rows": int(status_counts.get("probe_not_in_platform", 0)),
        "output_genes": int(collapsed.shape[0]),
        "sample_count": len(sample_cols),
        "output_relpath": str(out_path),
    }


def collapse_rnaseq_dataset(
    dataset_id: str,
    matrix_path: Path,
    platform_id: str,
    expected_samples: list[str],
    outdir: Path,
) -> dict[str, str | int]:
    df = pd.read_csv(matrix_path, sep="\t", compression="gzip", dtype=str)
    if "gene_symbol" not in df.columns:
        raise ValueError(f"{dataset_id} is missing gene_symbol column")

    sample_cols = [sample for sample in expected_samples if sample in df.columns]
    if not sample_cols:
        sample_cols = [col for col in df.columns if col not in ANNOTATION_COLUMNS]
    if not sample_cols:
        raise ValueError(f"{dataset_id} has no sample columns in {matrix_path}")

    normalized = df["gene_symbol"].map(lambda value: normalize_gene_symbol(value)[0])
    status = df["gene_symbol"].map(lambda value: normalize_gene_symbol(value)[1])
    valid = normalized.notna()

    value_frame = df.loc[valid, sample_cols].apply(pd.to_numeric, errors="coerce")
    collapsed = value_frame.groupby(normalized.loc[valid]).median().sort_index()
    collapsed.index.name = "gene_symbol"

    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"{dataset_id}_gene_expression.tsv.gz"
    with gzip.open(out_path, "wt", encoding="utf-8", newline="") as handle:
        collapsed.reset_index().to_csv(handle, sep="\t", index=False)

    status_counts = status.value_counts().to_dict()
    return {
        "dataset_id": dataset_id,
        "platform_id": platform_id,
        "mapping_source": "matrix_gene_symbol",
        "input_features": int(df.shape[0]),
        "mapped_feature_rows": int(valid.sum()),
        "blank_feature_rows": int(status_counts.get("blank_gene_symbol", 0)),
        "ambiguous_feature_rows": int(status_counts.get("ambiguous_gene_symbol", 0)),
        "missing_platform_rows": 0,
        "output_genes": int(collapsed.shape[0]),
        "sample_count": len(sample_cols),
        "output_relpath": str(out_path),
    }


def main() -> int:
    args = parse_args()
    root = Path(".")
    expression_dir = root / args.expression_dir
    sample_dir = root / args.sample_dir
    feature_map_dir = root / args.feature_map_dir
    outdir = root / args.outdir
    summary_out = root / args.summary_out

    expected_samples = build_expected_samples(read_tsv(root / args.sample_inclusion))
    platform_index = build_platform_index(sample_dir)
    dataset_ids = args.dataset_ids or sorted(expected_samples)

    platform_cache: dict[str, pd.DataFrame] = {}
    summary_rows: list[dict[str, str | int]] = []

    for dataset_id in dataset_ids:
        matrix_path = expression_dir / f"{dataset_id}_expression_included.tsv.gz"
        if not matrix_path.exists():
            raise FileNotFoundError(f"Missing expression matrix for {dataset_id}: {matrix_path}")

        platform_id = platform_index.get(dataset_id, "")
        if dataset_id == "GSE154918":
            summary = collapse_rnaseq_dataset(
                dataset_id=dataset_id,
                matrix_path=matrix_path,
                platform_id=platform_id,
                expected_samples=expected_samples[dataset_id],
                outdir=outdir,
            )
        else:
            if platform_id not in platform_cache:
                annotation_path = download_platform_annotation(
                    platform_id=platform_id,
                    platform_dir=root / args.platform_dir,
                )
                platform_cache[platform_id] = parse_platform_annotation(
                    path=annotation_path,
                    platform_id=platform_id,
                )
                write_feature_map(
                    platform_df=platform_cache[platform_id],
                    out_path=feature_map_dir / f"{platform_id}_probe_to_gene.tsv",
                )
            summary = collapse_array_dataset(
                dataset_id=dataset_id,
                matrix_path=matrix_path,
                platform_id=platform_id,
                feature_map=platform_cache[platform_id],
                expected_samples=expected_samples[dataset_id],
                outdir=outdir,
            )

        summary_rows.append(summary)
        print(
            f"[wrote] {summary['output_relpath']} "
            f"(samples={summary['sample_count']}, genes={summary['output_genes']})"
        )

    summary_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).sort_values("dataset_id").to_csv(
        summary_out,
        sep="\t",
        index=False,
    )
    print(f"[wrote] {summary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
