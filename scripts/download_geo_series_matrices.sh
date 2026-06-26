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

url_for() {
  case "$1" in
    GSE65682) printf '%s\n' "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE65nnn/GSE65682/matrix/GSE65682_series_matrix.txt.gz" ;;
    GSE95233) printf '%s\n' "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE95nnn/GSE95233/matrix/GSE95233_series_matrix.txt.gz" ;;
    GSE154918) printf '%s\n' "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE154nnn/GSE154918/suppl/GSE154918_Schughart_Sepsis_200320.txt.gz" ;;
    GSE28750) printf '%s\n' "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE28nnn/GSE28750/matrix/GSE28750_series_matrix.txt.gz" ;;
    GSE69528) printf '%s\n' "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE69nnn/GSE69528/matrix/GSE69528_series_matrix.txt.gz" ;;
    *)
      printf 'Unknown dataset: %s\n' "$1" >&2
      return 1
      ;;
  esac
}

download_one() {
  local dataset="$1"
  local url
  local out_dir
  local out_file

  url="$(url_for "${dataset}")"
  out_dir="${OUT_ROOT}/${dataset}"
  if [[ "${dataset}" == "GSE154918" ]]; then
    out_file="${out_dir}/GSE154918_Schughart_Sepsis_200320.txt.gz"
  else
    out_file="${out_dir}/${dataset}_series_matrix.txt.gz"
  fi

  mkdir -p "${out_dir}"

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf '[dry-run] %s -> %s\n' "${url}" "${out_file}"
    return 0
  fi

  if [[ -f "${out_file}" ]]; then
    printf '[skip] %s already exists\n' "${out_file}"
    return 0
  fi

  printf '[download] %s\n' "${dataset}"
  curl -fL --retry 3 --retry-delay 2 --output "${out_file}" "${url}"
  printf '[done] %s\n' "${out_file}"
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
