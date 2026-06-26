# High-AUROC sepsis decision failure

This repository is the public reproducibility package for the manuscript:

High AUROC can mask decision failure in sepsis transcriptomic classifiers: preprocessing stability outweighs post hoc calibration across cohorts

## What this repository reproduces

This package supports end-to-end reproduction of the public-data benchmark from GEO download to final benchmark tables and figure exports. It is designed for reviewers and readers who want to:

- download the public source matrices used in the study,
- rebuild sample-level metadata harmonization,
- reconstruct gene-level task matrices,
- rerun the preprocessing, calibration, threshold, bootstrap, and sensitivity analyses,
- regenerate the manuscript figures from analysis outputs,
- inspect figure, table, and data provenance.

This repository intentionally excludes manuscript drafting files, journal submission materials, and internal planning scaffolding.

## Repository structure

- `scripts/`: pipeline entrypoints and analysis utilities
- `data/manifest.tsv`: machine-readable dataset manifest used by the pipeline
- `docs/`: manifests, harmonization notes, decision rules, compute expectations, and release documentation
- `results/benchmarks/`: benchmark summary tables, prediction-level outputs, and regularization sensitivity results
- `results/tables/`: cohort summaries, confidence intervals, classification diagnostics, threshold sweeps, paired bootstrap differences, calibration diagnostics, adaptive-reference-size sensitivity, and sample-level harmonization supplements
- `results/models/`: exported logistic coefficient tables and fold-level coefficient-stability summaries
- `plots/`: publication-ready figure exports mirrored from the regenerated outputs

## Environment

Python 3.9 or newer is recommended.

Install dependencies:

```bash
pip install -r requirements.txt
```

## One-command reproduction

Run the full public pipeline:

```bash
bash scripts/reproduce_one_click.sh
```

This command performs:

1. GEO download for the four benchmark cohorts
2. Sample-metadata extraction and conservative phenotype harmonization
3. Included-sample expression extraction
4. Probe-to-gene or gene-level harmonization
5. Task matrix construction
6. Benchmark reruns for the primary, stress-test, and cross-platform analyses
7. Classification diagnostics, threshold sweeps, paired bootstrap differences, calibration summaries, adaptive-reference-size sensitivity, and regularization sensitivity
8. Confidence-interval, coefficient, and harmonization-supplement export
9. Regeneration of main and supplementary figures
10. Audit log and environment capture under `logs/` and `docs/audit_runs/`

If the raw public files are already present locally, use:

```bash
bash scripts/reproduce_one_click.sh --skip-download
```

## Review bundle

To build the reviewer-facing archive used for release packaging:

```bash
bash scripts/build_review_bundle.sh
```

The archive, checksum manifest, and file list are written under `docs/review_bundle/`.

## Public data sources

Public source files and download URLs are listed in [docs/DATA_MANIFEST.tsv](docs/DATA_MANIFEST.tsv). The analysis-specific dataset manifest consumed by the pipeline lives at `data/manifest.tsv`.

## Expected runtime

See [docs/COMPUTE_PLAN.md](docs/COMPUTE_PLAN.md) for runtime, memory, disk, and smoke-test expectations.

## Provenance

- Figure-to-source mapping: [docs/FIGURE_PROVENANCE.tsv](docs/FIGURE_PROVENANCE.tsv)
- Harmonization rationale: [docs/HARMONIZATION_NOTES.md](docs/HARMONIZATION_NOTES.md)
- Statistical rules: [docs/STATISTICAL_DECISION_RULES.md](docs/STATISTICAL_DECISION_RULES.md)

## Current release tag

The revised public package is tagged as `v1.1.0-pone-revision`.

## Release note

The manuscript should cite the Zenodo version DOI associated with the tagged release used for submission.
