# Phase 3 Runbook — Gemma 4 12B 5-Fold QLoRA Training

This guide walks through running **Phase 3** on a Linux NVIDIA GPU machine (RunPod, Vast.ai, Lambda, GCP, AWS, or Colab Pro+) from a fresh clone to trained 5-fold LoRA adapters + calibrated Out-Of-Fold parameters.

---

## 1. Provision Your GPU Instance

- **Recommended Instance:** `1× NVIDIA L4 (24 GB)`, `RTX 4090 (24 GB)`, or `1× A100 (40 GB / 80 GB)`
- **Base Image:** Any PyTorch `2.6+` + CUDA `12.4–12.9` Ubuntu 22.04/24.04 `x86_64` image
- **Disk Space:** `80 GB+` NVMe SSD (`~27 GB` Hugging Face cache for `google/gemma-4-12B-it` + checkpoints)

---

## 2. Clone Repository & Supply `train.csv`

```bash
git clone https://github.com/chhitiz-t/check-llm.git
cd check-llm
mkdir -p data/raw
```

Copy `train.csv` and `test.csv` (`2.0 MB` total) to the GPU box (from your Mac):

```bash
# Run on your Mac terminal:
scp /Users/chhitiz.t/Downloads/kaggle/name-that-llm/data/raw/{train.csv,test.csv,sample_submission.parquet} <USER>@<GPU_HOST>:~/check-llm/data/raw/
```

*(Alternatively, set `export KAGGLE_API_TOKEN="..."` on the GPU box and the Phase 3 script will auto-download `train.csv`.)*

If your Hugging Face account uses a token for higher download speeds / Gemma access:
```bash
export HF_TOKEN="hf_your_token_here"
```

---

## 3. Recommended Step A: Run Fold 0 Pilot Check First (~8–20 min)

Validate GPU VRAM, exact Gemma 4 HF tokenization, training loss convergence, and Fold 0 Balanced Accuracy before running all 5 folds:

```bash
FOLDS="0" bash scripts/03_cloud_gpu_bootstrap_and_train.sh
```

What to expect on **Fold 0**:
- `scripts/02c_phase2_prompt_audit.py` switches automatically to the **exact `google/gemma-4-12B-it` HF tokenizer** when `HF_TOKEN` / HF access is active.
- Baseline linear TF-IDF + stylometry scored **`0.3426`** Balanced Accuracy; Fold 0 Gemma 4 12B QLoRA should surpass this substantially by Epoch 2–3.
- Fold 0 is saved automatically (`artifacts/lora_fold_0/best_adapter/` + `artifacts/oof_fold_0.npz`), so **Step B will automatically skip Fold 0 and resume from Fold 1**.

---

## 4. Step B: Train Remaining Folds (`0..4`) + Calibrate OOF + Bundle

Run the full 5-fold pipeline (automatically skips Fold 0 since it already finished):

```bash
# Set BUILD_VLLM_BUNDLE=1 if you also want the offline Kaggle vLLM wheelhouse built right away!
BUILD_VLLM_BUNDLE=1 bash scripts/03_cloud_gpu_bootstrap_and_train.sh
```

### Auto-Tuning Behavior Inside `03_cloud_gpu_bootstrap_and_train.sh`
| Detected GPU VRAM | Compute Dtype | `batch_size` | `grad_accum` | Effective Batch | Est. Time (All 5 Folds) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **$\ge 38$ GB (A100 / H100 / L40S)** | `bfloat16` | `8` | `2` | `16` | **~35–45 min** |
| **20–37 GB (L4 / RTX 3090 / 4090)** | `bfloat16` | `2` | `8` | `16` | **~1.5–2.0 hrs** |
| **$< 20$ GB (`2× T4` fallback)** | `float16` | `1` | `16` | `16` | ~5.5–6.5 hrs |

---

## 5. Verify Outputs After Phase 3 Completes

Check that these artifacts exist:
1. `artifacts/lora_fold_{0..4}/best_adapter/` — 5 trained PEFT LoRA adapters (~65 MB each)
2. `artifacts/oof_fold_{0..4}.npz` — Out-Of-Fold validation probability matrices
3. `artifacts/postprocess_params.json` — Calibrated 18-class Balanced Accuracy log-multipliers
4. `artifacts/gemma4_12b_lora_adapters.tar.gz` — Single compressed archive (~250–325 MB) ready to download or upload to Kaggle Datasets
5. `kaggle_upload_bundle/` *(when `BUILD_VLLM_BUNDLE=1`)* — Complete offline Kaggle Dataset folder containing Linux `x86_64` `vllm` wheels + `name-that-llm` wheel + all 5 fold adapters + `postprocess_params.json`.
