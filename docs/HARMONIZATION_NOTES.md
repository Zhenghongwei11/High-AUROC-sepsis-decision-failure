# Harmonization notes

## Scope

This benchmark uses four public whole-blood transcriptomic cohorts from GEO:

- `GSE65682`: primary development cohort
- `GSE95233`: external microarray validation cohort
- `GSE154918`: RNA-seq transfer cohort
- `GSE28750`: inflammatory confounding stress-test cohort

## Phenotype harmonization

Phenotype labels were harmonized conservatively and only when the processed GEO metadata supported a defensible mapping. The harmonized labels are:

- `healthy`
- `infection_non_sepsis`
- `sepsis`
- `septic_shock`
- `noninfectious_inflammation`
- `unassigned`

Samples were excluded from the main benchmark when the GEO metadata did not support a stable mapping or when they represented follow-up rather than baseline sampling.

## Dataset-specific rules

### GSE65682

- Healthy volunteers were identified from sample titles.
- `abdo_s` was mapped to `sepsis`.
- `ctrl_GI` was mapped to `noninfectious_inflammation`.
- Pneumonia descriptors were retained only when they supported conservative mapping.

### GSE95233

- Healthy controls were identified from `source_name_ch1`.
- Septic-shock samples were restricted to day-1 baseline measurements.
- Follow-up measurements were excluded from the mainline benchmark.

### GSE154918

- The expression matrix comes from the GEO supplementary file.
- Sample metadata comes from the GEO series-matrix file.
- Follow-up samples were excluded from the mainline benchmark.

### GSE28750

- `HEALTHY` was mapped to `healthy`.
- `POST_SURGICAL` was mapped to `noninfectious_inflammation`.
- `SEPSIS` was mapped to `sepsis`.

## Feature harmonization

- Microarray cohorts were mapped from probe to gene using GEO platform annotations.
- Probes with ambiguous or multi-symbol gene mappings were excluded.
- When multiple probes mapped uniquely to the same gene, gene-level expression was collapsed using the median within cohort.
- The smaller feature space of `GSE65682` reflects conservative single-symbol probe mapping rather than downstream feature filtering.

## Task definitions

- Primary analysis: sepsis or septic shock versus healthy controls
- Secondary analysis: sepsis versus noninfectious inflammation
- Cross-platform readout: microarray-to-RNA-seq transfer viewed as a platform-specific readout of the primary analysis
