# Statistical Decision Rules

## Benchmark framing

- Primary analysis: sepsis or septic shock versus healthy controls
- Secondary analysis: sepsis versus non-infectious inflammation
- Cross-platform readout: RNA-seq transfer evaluated as a platform-specific view of the primary analysis

## Model family

- Baseline learner: logistic regression
- Fixed settings: `class_weight="balanced"`, `solver="liblinear"`, `max_iter=5000`, random seed `20260313`

## Primary evaluation rules

- Internal development performance: five-fold stratified cross-validation in the development cohort
- External validation: train once on the development cohort, then score held-out cohorts without retuning
- Primary decision threshold: fixed `0.5`
- Secondary threshold readout: training-derived balanced-accuracy threshold

## Performance metrics

- AUROC
- Average precision
- Balanced accuracy
- Brier score
- Predicted positive rate

Balanced accuracy is treated as the key decision-level metric because the central question is whether externally transferred scores remain usable at a fixed threshold.

## Uncertainty

- External confidence intervals: 2,000 stratified bootstrap replicates
- Positive and negative classes are resampled separately within each external cohort

## Rounding conventions

- Main text: usually two decimal places for balanced accuracy and AUROC
- Supplementary tables: retain full exported precision

## Interpretation constraints

- High AUROC alone is not treated as evidence of usable external classification
- Calibration is interpreted only after preprocessing stability is considered
- The inflammatory comparator analysis is treated as a harder external stress test, not as a standalone discovery claim
