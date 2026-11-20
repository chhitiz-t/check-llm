#!/usr/bin/env bash
set -euo pipefail

# Run this on a Linux x86_64 machine (Python 3.12 matching Kaggle's 2026 image)
# to package offline vLLM wheels + QAT W4A16 base weights + your 5 LoRA adapters
# into Kaggle Datasets for internet-disabled submission notebooks.

BUNDLE_DIR="${1:-kaggle_upload_bundle}"
mkdir -p "${BUNDLE_DIR}/wheelhouse" "${BUNDLE_DIR}/adapters" "${BUNDLE_DIR}/src_pkg"

echo "[1/4] Downloading vLLM nightly wheelhouse (gemma4-unified support) for Linux x86_64..."
pip download \
  --dest "${BUNDLE_DIR}/wheelhouse" \
  --pre vllm \
  --extra-index-url https://wheels.vllm.ai/nightly/cu128 \
  --extra-index-url https://download.pytorch.org/whl/cu128

echo "[2/4] Building local name-that-llm wheel..."
pip wheel . --no-deps --wheel-dir "${BUNDLE_DIR}/src_pkg"

echo "[3/4] Copying trained 5-fold LoRA adapters & postprocess_params.json..."
for FOLD in 0 1 2 3 4; do
  if [ -d "artifacts/lora_fold_${FOLD}/best_adapter" ]; then
    cp -r "artifacts/lora_fold_${FOLD}/best_adapter" "${BUNDLE_DIR}/adapters/fold_${FOLD}"
  fi
done
if [ -f "artifacts/postprocess_params.json" ]; then
  cp "artifacts/postprocess_params.json" "${BUNDLE_DIR}/adapters/postprocess_params.json"
fi

echo "[4/4] (Optional) Pre-download google/gemma-4-12B-it-qat-w4a16-ct if not attaching from Kaggle Models:"
echo "  huggingface-cli download google/gemma-4-12B-it-qat-w4a16-ct --local-dir ${BUNDLE_DIR}/gemma-4-12B-it-qat-w4a16-ct"
echo "Done! Upload ${BUNDLE_DIR} as a private Kaggle Dataset and attach it to your submission notebook."
