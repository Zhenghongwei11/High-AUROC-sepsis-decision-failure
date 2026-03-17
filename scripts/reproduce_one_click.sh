#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

SKIP_DOWNLOAD=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/reproduce_one_click.sh [--skip-download]

Options:
  --skip-download   Reuse already-downloaded GEO files under data/raw/geo/
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-download)
      SKIP_DOWNLOAD=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

mkdir -p \
  logs \
  docs/audit_runs \
  results/figures \
  results/metadata \
  results/tasks \
  results/benchmarks \
  results/tables \
  results/models \
  plots

RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="logs/reproduce_${RUN_ID}.log"
AUDIT_DIR="docs/audit_runs/${RUN_ID}"
mkdir -p "${AUDIT_DIR}"

{
  echo "run_id=${RUN_ID}"
  echo "start_time=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "python=$(python3 --version 2>&1)"
  echo "pip=$(python3 -m pip --version 2>&1)"
  echo "root=${ROOT}"
} > "${AUDIT_DIR}/run_metadata.txt"

exec > >(tee "${LOG_PATH}") 2>&1

echo "[run] ${RUN_ID}"
echo "[audit] ${AUDIT_DIR}"

if [[ "${SKIP_DOWNLOAD}" -eq 0 ]]; then
  bash scripts/download_geo_series_matrices.sh
fi

python3 scripts/extract_geo_sample_metadata.py \
  data/raw/geo/GSE65682/GSE65682_series_matrix.txt.gz \
  data/raw/geo/GSE95233/GSE95233_series_matrix.txt.gz \
  data/raw/geo/GSE154918/GSE154918_series_matrix.txt.gz \
  data/raw/geo/GSE28750/GSE28750_series_matrix.txt.gz \
  --outdir data/processed/sample_metadata

python3 scripts/harmonize_sample_labels.py
python3 scripts/export_sample_inclusion.py
python3 scripts/extract_expression_matrices.py GSE65682 GSE95233 GSE154918 GSE28750
python3 scripts/build_gene_level_matrices.py GSE65682 GSE95233 GSE154918 GSE28750
python3 scripts/build_dataset_summary.py
python3 scripts/build_task_ready_datasets.py task_a task_b task_c
python3 scripts/run_task_a_preprocessing_benchmark.py
python3 scripts/run_task_preprocessing_benchmark.py --task-id task_b
python3 scripts/run_task_preprocessing_benchmark.py --task-id task_c
python3 scripts/run_task_a_calibration_benchmark.py
python3 scripts/compute_external_metric_ci.py
python3 scripts/export_logistic_model_coefficients.py --task-id task_a
python3 scripts/build_cohort_master_summary.py
python3 scripts/build_cohort_flow_summary.py
python3 scripts/build_metadata_completeness_table.py
python3 scripts/plot_benchmark_design_figure.py
python3 scripts/plot_task_a_figure2_prototype.py
python3 scripts/plot_task_b_figure_prototype.py
python3 scripts/plot_supplementary_figures.py

find plots -type f \( -name '*.png' -o -name '*.pdf' -o -name '*.tsv' \) -delete
find results/figures -maxdepth 1 -type f \( -name '*.png' -o -name '*.pdf' -o -name '*.tsv' \) -exec cp {} plots/ \;

{
  echo "completed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "log=${LOG_PATH}"
} >> "${AUDIT_DIR}/run_metadata.txt"

python3 -m pip freeze > "${AUDIT_DIR}/pip_freeze.txt"

echo "[done] Full public pipeline reproduction complete."
echo "[figures] results/figures/"
echo "[plots] plots/"
