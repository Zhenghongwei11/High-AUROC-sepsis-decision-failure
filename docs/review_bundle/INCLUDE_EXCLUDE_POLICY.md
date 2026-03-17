# Include and exclude policy

## Included

The review bundle includes the reviewer-facing computational package:

- `README.md`
- `LICENSE`
- `requirements.txt`
- `scripts/`
- `docs/` analysis documents and manifests
- `data/manifest.tsv`
- `results/benchmarks/`
- `results/tables/`
- `results/models/`
- `plots/`

## Excluded

The review bundle intentionally excludes generated or local-only artifacts:

- raw GEO downloads under `data/raw/`
- generated intermediate matrices under `data/processed/`
- runtime logs under `logs/`
- audit-run outputs under `docs/audit_runs/`
- generated review-bundle artifacts under `docs/review_bundle/`
- git metadata under `.git/`

## Rationale

The bundle is designed to stay small, reviewer-facing, and free of local runtime clutter while preserving everything needed to inspect the analysis outputs and rerun the full public workflow from GEO.
