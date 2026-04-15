#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

DATASET_ROOT="${DATASET_ROOT:-${ROOT_DIR}/PDBscreen}"
OUT_PARQUET="${OUT_PARQUET:-${ROOT_DIR}/dataset/pdbscreen_manifest.parquet}"
FEATURE_WORKERS="${FEATURE_WORKERS:-8}"
FEATURE_CHUNKSIZE="${FEATURE_CHUNKSIZE:-200}"

"${PYTHON_BIN}" "${ROOT_DIR}/utils/generate_manifest.py" \
  --dataset-root "${DATASET_ROOT}" \
  --out-parquet "${OUT_PARQUET}"

"${PYTHON_BIN}" "${ROOT_DIR}/utils/features.py" \
  --manifest "${OUT_PARQUET}" \
  --workers "${FEATURE_WORKERS}" \
  --chunksize "${FEATURE_CHUNKSIZE}"
