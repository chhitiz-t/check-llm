#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/gemma4_12b_vllm.yaml}"

python scripts/02_prepare_folds.py --config "${CONFIG}"

for FOLD in 0 1 2 3 4; do
  echo "============================================================"
  echo "Training Gemma 4 12B QLoRA — Fold ${FOLD}"
  echo "============================================================"
  python -m name_that_llm.train_qlora \
    --config "${CONFIG}" \
    --fold "${FOLD}" \
    --folds-parquet data/processed/train_folds.parquet
done

echo "Fitting OOF Balanced Accuracy post-processing multipliers..."
python scripts/03b_fit_oof_postprocess.py --config "${CONFIG}"
