#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

DATASET_ROOT="${DATASET_ROOT:-${ROOT_DIR}/PDBscreen}"
OUT_PARQUET="${OUT_PARQUET:-${DATASET_ROOT}/pdbscreen_manifest.parquet}"

"${PYTHON_BIN}" "${ROOT_DIR}/utils/generate_manifest.py" \
  --dataset-root "${DATASET_ROOT}" \
  --out-parquet "${OUT_PARQUET}"
