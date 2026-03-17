#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${ROOT_DIR}/data/raw/geo"
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/download_geo_series_matrices.sh [--dry-run] [GSE...]

Examples:
  bash scripts/download_geo_series_matrices.sh
  bash scripts/download_geo_series_matrices.sh --dry-run
  bash scripts/download_geo_series_matrices.sh GSE65682 GSE154918
EOF
}

download_file() {
  local url="$1"
  local out_file="$2"

  mkdir -p "$(dirname "${out_file}")"

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf '[dry-run] %s -> %s\n' "${url}" "${out_file}"
    return 0
  fi

  if [[ -f "${out_file}" ]]; then
    printf '[skip] %s already exists\n' "${out_file}"
    return 0
  fi

  curl -fL --retry 3 --retry-delay 2 --output "${out_file}" "${url}"
  printf '[done] %s\n' "${out_file}"
}

download_one() {
  local dataset="$1"
  local out_dir="${OUT_ROOT}/${dataset}"

  printf '[dataset] %s\n' "${dataset}"

  case "${dataset}" in
    GSE65682)
      download_file \
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE65nnn/GSE65682/matrix/GSE65682_series_matrix.txt.gz" \
        "${out_dir}/GSE65682_series_matrix.txt.gz"
      ;;
    GSE95233)
      download_file \
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE95nnn/GSE95233/matrix/GSE95233_series_matrix.txt.gz" \
        "${out_dir}/GSE95233_series_matrix.txt.gz"
      ;;
    GSE154918)
      download_file \
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE154nnn/GSE154918/suppl/GSE154918_Schughart_Sepsis_200320.txt.gz" \
        "${out_dir}/GSE154918_Schughart_Sepsis_200320.txt.gz"
      download_file \
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE154nnn/GSE154918/matrix/GSE154918_series_matrix.txt.gz" \
        "${out_dir}/GSE154918_series_matrix.txt.gz"
      ;;
    GSE28750)
      download_file \
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE28nnn/GSE28750/matrix/GSE28750_series_matrix.txt.gz" \
        "${out_dir}/GSE28750_series_matrix.txt.gz"
      ;;
    GSE69528)
      download_file \
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE69nnn/GSE69528/matrix/GSE69528_series_matrix.txt.gz" \
        "${out_dir}/GSE69528_series_matrix.txt.gz"
      ;;
    *)
      printf 'Unknown dataset: %s\n' "${dataset}" >&2
      return 1
      ;;
  esac
}

declare -a datasets=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    GSE*)
      datasets+=("$1")
      shift
      ;;
    *)
      printf 'Unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ ${#datasets[@]} -eq 0 ]]; then
  datasets=(GSE65682 GSE95233 GSE154918 GSE28750)
fi

for dataset in "${datasets[@]}"; do
  download_one "${dataset}"
done
