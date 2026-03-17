#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

OUTDIR="docs/review_bundle"
TMP_LIST="${OUTDIR}/CONTENTS.txt"
ZIP_PATH="${OUTDIR}/high-auroc-sepsis-decision-failure_review_bundle.zip"
CHECKSUM_PATH="${OUTDIR}/CHECKSUMS.sha256"

mkdir -p "${OUTDIR}"

find . \
  -type f \
  ! -path './.git/*' \
  ! -path './docs/review_bundle/*' \
  ! -path './data/raw/*' \
  ! -path './data/processed/*' \
  ! -path './results/metadata/*' \
  ! -path './results/tasks/*' \
  ! -path './results/figures/*' \
  ! -path './logs/*' \
  ! -path './docs/audit_runs/*' \
  ! -name '.DS_Store' \
  | sed 's#^\./##' \
  | LC_ALL=C sort > "${TMP_LIST}"

rm -f "${ZIP_PATH}"
zip -X -q "${ZIP_PATH}" -@ < "${TMP_LIST}"
shasum -a 256 "${ZIP_PATH}" > "${CHECKSUM_PATH}"

echo "[wrote] ${ZIP_PATH}"
echo "[wrote] ${CHECKSUM_PATH}"
echo "[wrote] ${TMP_LIST}"
