#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${1:-data/raw}"
mkdir -p "${DATA_DIR}"

if command -v kaggle >/dev/null 2>&1; then
  KAGGLE_BIN="kaggle"
elif [ -x ".venv/bin/kaggle" ]; then
  KAGGLE_BIN=".venv/bin/kaggle"
else
  echo "Error: kaggle CLI not found. Install via .venv/bin/pip install kaggle" >&2
  exit 1
fi

echo "[1/2] Downloading Kaggle competition files into ${DATA_DIR}..."
"${KAGGLE_BIN}" competitions download -c name-that-llm -p "${DATA_DIR}"

echo "[2/2] Extracting archive..."
unzip -o "${DATA_DIR}/name-that-llm.zip" -d "${DATA_DIR}"
ls -lh "${DATA_DIR}"
