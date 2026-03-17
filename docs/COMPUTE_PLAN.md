# Compute plan

## Intended use

This public package is designed to rerun the full benchmark from public GEO files on a standard workstation. It is not a raw-data redistribution archive.

## Full-run expectations

- Hardware: standard CPU workstation or laptop
- Runtime: usually 15 to 45 minutes after network download, depending on connection speed
- Memory: approximately 8 to 16 GB
- Disk: approximately 1 to 3 GB during the full run because GEO downloads and intermediate matrices are generated locally

## Downsampled smoke test

If public GEO files were already downloaded in a previous run, a quick smoke test is:

```bash
bash scripts/reproduce_one_click.sh --skip-download
```

This reuses the local downloads and reruns the benchmark and figure-generation steps.

## What the full run performs

- GEO download
- sample-level metadata extraction
- conservative phenotype harmonization
- expression extraction and gene-level harmonization
- task-matrix assembly
- preprocessing and calibration benchmarks
- confidence-interval and coefficient export
- figure regeneration
- audit log capture

## What remains excluded from the public package

- manuscript drafting sources
- journal submission materials
- internal planning directories

## Public inputs

All public inputs are listed in `docs/DATA_MANIFEST.tsv` and `data/manifest.tsv`.
