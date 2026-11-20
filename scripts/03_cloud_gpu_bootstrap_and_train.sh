#!/usr/bin/env bash
# ==============================================================================
# Phase 3 One-Command Cloud GPU Bootstrap, 5-Fold QLoRA Training & Packaging
# ==============================================================================
# Usage on a Linux NVIDIA GPU machine (L4 24GB / RTX 4090 / A100 / H100):
#
#   # Full 5-fold run (with crash/spot-instance auto-resume):
#   bash scripts/03_cloud_gpu_bootstrap_and_train.sh
#
#   # Quick 1-fold pilot check (~7-20 min depending on GPU):
#   FOLDS="0" bash scripts/03_cloud_gpu_bootstrap_and_train.sh
#
# Optional environment variables:
#   CONFIG="configs/gemma4_12b_vllm.yaml"   Config path
#   FOLDS="0 1 2 3 4"                       Space-separated folds to train
#   FORCE_RETRAIN="0"                       Set "1" to retrain completed folds
#   HF_TOKEN="hf_..."                       Optional Hugging Face token
#   KAGGLE_API_TOKEN="..."                  Auto-downloads train.csv if missing
#   BUILD_VLLM_BUNDLE="0"                   Set "1" to also build Kaggle vLLM wheels
# ==============================================================================

set -euo pipefail

CONFIG="${CONFIG:-configs/gemma4_12b_vllm.yaml}"
FOLDS="${FOLDS:-0 1 2 3 4}"
FORCE_RETRAIN="${FORCE_RETRAIN:-0}"
BUILD_VLLM_BUNDLE="${BUILD_VLLM_BUNDLE:-0}"

echo "============================================================"
echo "[Phase 3] Step 1/6 — Checking NVIDIA GPU & Auto-Tuning Batch/Dtype"
echo "============================================================"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found. Phase 3 requires an NVIDIA CUDA GPU." >&2
  exit 1
fi

nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv
GPU_MEM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
GPU_CC_MAJOR=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | cut -d. -f1 | tr -d ' ')

# Auto-select dtype & batch size from GPU compute capability and VRAM
if [ "${GPU_CC_MAJOR}" -lt 8 ]; then
  # Turing (e.g., Tesla T4 SM75) — no native bfloat16
  COMPUTE_DTYPE="float16"
else
  # Ampere / Ada / Hopper / Blackwell (L4, 3090, 4090, A100, H100)
  COMPUTE_DTYPE="bfloat16"
fi

if [ "${GPU_MEM_MIB}" -ge 38000 ]; then
  # 40GB - 80GB+ GPUs (A100 / H100 / L40S)
  BATCH_SIZE="${BATCH_SIZE:-8}"
  EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
  GRAD_ACCUM="${GRAD_ACCUM:-2}"
elif [ "${GPU_MEM_MIB}" -ge 20000 ]; then
  # 24GB GPUs (L4 / RTX 3090 / RTX 4090 / A10G)
  BATCH_SIZE="${BATCH_SIZE:-2}"
  EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
  GRAD_ACCUM="${GRAD_ACCUM:-8}"
else
  # 16GB GPUs (T4x2)
  BATCH_SIZE="${BATCH_SIZE:-1}"
  EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
  GRAD_ACCUM="${GRAD_ACCUM:-16}"
fi

echo "-> Selected compute_dtype=${COMPUTE_DTYPE}, batch_size=${BATCH_SIZE}, grad_accum=${GRAD_ACCUM} (effective batch=$((BATCH_SIZE * GRAD_ACCUM)))"

echo "============================================================"
echo "[Phase 3] Step 2/6 — Setting Up Python Virtualenv & Dependencies"
echo "============================================================"
if [ ! -d ".venv" ]; then
  # Reuse pre-installed container PyTorch+CUDA packages when available
  python3 -m venv --system-site-packages .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

pip install --upgrade pip -q
pip install -e ".[train]" "huggingface_hub[cli]" kaggle six tabulate -q

if [ -n "${HF_TOKEN:-}" ]; then
  export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
fi

echo "============================================================"
echo "[Phase 3] Step 3/6 — Verifying Competition Dataset & Building Folds"
echo "============================================================"
if [ ! -f "data/raw/train.csv" ]; then
  echo "data/raw/train.csv not found locally; attempting Kaggle CLI download..."
  bash scripts/01_download_data.sh data/raw
fi

python scripts/02_prepare_folds.py --config "${CONFIG}"
python scripts/02c_phase2_prompt_audit.py --config "${CONFIG}"

echo "============================================================"
echo "[Phase 3] Step 4/6 — Running Gemma 4 12B QLoRA Training (Folds: ${FOLDS})"
echo "============================================================"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

for FOLD in ${FOLDS}; do
  OOF_FILE="artifacts/oof_fold_${FOLD}.npz"
  ADAPTER_DIR="artifacts/lora_fold_${FOLD}/best_adapter"

  if [ "${FORCE_RETRAIN}" != "1" ] && [ -f "${OOF_FILE}" ] && [ -d "${ADAPTER_DIR}" ]; then
    echo "[Fold ${FOLD}] Found existing ${ADAPTER_DIR} and ${OOF_FILE} — skipping (set FORCE_RETRAIN=1 to override)."
    continue
  fi

  echo "------------------------------------------------------------"
  echo "[Fold ${FOLD}] Starting Gemma 4 12B QLoRA fine-tuning..."
  echo "------------------------------------------------------------"
  python -m name_that_llm.train_qlora \
    --config "${CONFIG}" \
    --fold "${FOLD}" \
    --folds-parquet data/processed/train_folds.parquet \
    --batch-size "${BATCH_SIZE}" \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --grad-accum "${GRAD_ACCUM}" \
    --compute-dtype "${COMPUTE_DTYPE}"
done

echo "============================================================"
echo "[Phase 3] Step 5/6 — OOF Evaluation & Balanced Accuracy Calibration"
echo "============================================================"
python scripts/03b_fit_oof_postprocess.py --config "${CONFIG}"

echo "============================================================"
echo "[Phase 3] Step 6/6 — Packaging Trained Adapters Archive"
echo "============================================================"
mkdir -p artifacts/export_adapters
for FOLD in ${FOLDS}; do
  if [ -d "artifacts/lora_fold_${FOLD}/best_adapter" ]; then
    rm -rf "artifacts/export_adapters/fold_${FOLD}"
    cp -r "artifacts/lora_fold_${FOLD}/best_adapter" "artifacts/export_adapters/fold_${FOLD}"
  fi
done
if [ -f "artifacts/postprocess_params.json" ]; then
  cp "artifacts/postprocess_params.json" "artifacts/export_adapters/postprocess_params.json"
fi

tar -czf artifacts/gemma4_12b_lora_adapters.tar.gz -C artifacts export_adapters
ls -lh artifacts/gemma4_12b_lora_adapters.tar.gz

if [ "${BUILD_VLLM_BUNDLE}" = "1" ]; then
  echo "Building full offline Kaggle vLLM upload bundle..."
  bash scripts/04_build_offline_wheels.sh kaggle_upload_bundle
fi

echo "============================================================"
echo "Phase 3 Complete! Artifacts ready in artifacts/gemma4_12b_lora_adapters.tar.gz"
echo "============================================================"
