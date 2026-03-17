# Reviewer guide

## Goal

This package allows reviewers to reproduce the sepsis transcriptomic benchmark from public GEO files and regenerate the manuscript figures.

## Recommended workflow

1. Create a Python 3.11+ environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the public pipeline:

```bash
bash scripts/reproduce_one_click.sh
```

4. Inspect the regenerated figure exports in `plots/` and the benchmark tables under `results/`.

## Key outputs

- `plots/figure1_design_overview.png`
- `plots/figure2_external_threshold_stability.png`
- `plots/figure3_harder_negative_stress_test.png`
- `plots/figureS1_cohort_flow.png`
- `plots/figureS2_task_c_rnaseq_transfer.png`
- `plots/figureS3_task_a_coefficient_stability.png`

## Provenance

- Dataset downloads: `docs/DATA_MANIFEST.tsv`
- Figure-to-table mapping: `docs/FIGURE_PROVENANCE.tsv`
- Harmonization logic: `docs/HARMONIZATION_NOTES.md`
- Statistical rules: `docs/STATISTICAL_DECISION_RULES.md`
