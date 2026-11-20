# Name That LLM — Gemma 4 12B QLoRA + Dual-T4 vLLM Pipeline

End-to-end training, quantization, calibration, and offline **Dual-T4 vLLM** inference pipeline for the [Name That LLM](https://www.kaggle.com/competitions/name-that-llm) Kaggle competition (classify ~4,000 short passages across 18 frontier LLMs, evaluated on **Balanced Accuracy**).

---

## Architecture Highlights

1. **Single-Token Restricted Label SFT (`A`..`R`):**
   - Maps all 18 valid model names deterministically to 18 single-token uppercase codes (`A`..`R`).
   - Disables Gemma 4 thinking mode (`enable_thinking=False`) so assistant output token `#0` is directly the class token.
   - `RestrictedLabelTrainer` computes inverse-frequency class-weighted cross-entropy over **only the 18 class-code logits**, matching an 18-way classification head while remaining 100% compatible with `vLLM.generate(max_tokens=1, logprobs=20)`.
2. **Leak-Free Cross-Validation (`StratifiedGroupKFold` on `prompt_id`):**
   - `train.csv` has 2,699 passages across 150 prompts, while `test.csv` uses **unseen prompts**. Grouping by `prompt_id` prevents topic leakage.
3. **Official QAT W4A16 + Dual-T4 vLLM Parallelism:**
   - Serves Google's official QAT checkpoint `google/gemma-4-12B-it-qat-w4a16-ct` (**~8.3 GB** static weights in `compressed-tensors` W4A16 format) with `--dtype float16`, `--limit-mm-per-prompt '{"image": 0, "audio": 0}'`, `max_model_len=2048`, and **Automatic Prefix Caching** (`enable_prefix_caching=True`).
   - Spawns **two isolated worker processes** (`CUDA_VISIBLE_DEVICES=0` and `CUDA_VISIBLE_DEVICES=1`), one per 16 GB T4 GPU, scoring all 5 fold LoRA adapters without reloading base weights (~25–35 minutes total for all 5 folds × 4,000 test passages, well inside the 9-hour limit).
4. **Balanced Accuracy Post-Processing:**
   - Tunes class prior adjustment ($\tau$) + 18 log-multipliers on Out-Of-Fold probabilities via coordinate ascent to counter unequal test priors and public/private distribution shifts.

---

## Project Layout

```text
name-that-llm/
├── configs/
│   └── gemma4_12b_vllm.yaml           # Model, CV, QLoRA, vLLM T4, and post-processing config
├── src/name_that_llm/
│   ├── constants.py                   # 18 official model labels <-> single-token codes ('A'..'R')
│   ├── prompts.py                     # Static prefix-cacheable Gemma 4 prompt builder
│   ├── data.py                        # Validation, StratifiedGroupKFold(prompt_id), SFT tokenization
│   ├── metrics.py                     # Balanced Accuracy, per-class recall & top confusions
│   ├── train_qlora.py                 # 4-bit QLoRA trainer with 18-way class-weighted CE loss
│   ├── merge_and_quantize.py          # Optional LoRA merge + llmcompressor W4A16 export
│   ├── vllm_worker.py                 # Isolated single-GPU vLLM logprob scorer
│   ├── infer_vllm_dual_t4.py          # Dual-T4 shard orchestrator -> submission.parquet
│   ├── postprocess.py                 # OOF prior & coordinate-ascent Balanced Accuracy optimizer
│   └── submission.py                  # Strict competition schema & label validator
├── scripts/
│   ├── 01_download_data.sh            # Download competition CSVs via Kaggle CLI
│   ├── 02_prepare_folds.py            # Build data/processed/train_folds.parquet
│   ├── 03_train_all_folds.sh          # Train 5 QLoRA folds + fit OOF multipliers
│   ├── 03b_fit_oof_postprocess.py     # Aggregate OOFs & export postprocess_params.json
│   ├── 04_build_offline_wheels.sh     # Bundle vLLM nightly wheels + LoRA adapters for offline Kaggle
│   └── 05_smoke_test_pipeline.py      # Local CPU end-to-end smoke test
└── kaggle_notebook/
    └── kaggle_submit_vllm.py          # Self-contained offline Kaggle T4x2 submission script
```

---

## End-to-End Quickstart

### Step 1: Install & Run Smoke Test Locally
```bash
pip install -e .
python scripts/05_smoke_test_pipeline.py
```

### Step 2: Download Competition Data & Prepare Folds
*(Accept competition rules on Kaggle first.)*
```bash
bash scripts/01_download_data.sh
python scripts/02_prepare_folds.py --config configs/gemma4_12b_vllm.yaml
```

### Step 3: Train 5-Fold Gemma 4 12B QLoRA & Calibrate OOF
*(Recommended training GPU: 1× L4 24GB, RTX 3090/4090, or A100 40GB with `[train]` extras installed.)*
```bash
pip install -e ".[train]"
bash scripts/03_train_all_folds.sh configs/gemma4_12b_vllm.yaml
```
Produces:
- `artifacts/lora_fold_{0..4}/best_adapter/` (~65 MB per fold)
- `artifacts/oof_fold_{0..4}.npz`
- `artifacts/postprocess_params.json`

### Step 4: Package Offline Assets for Kaggle
```bash
bash scripts/04_build_offline_wheels.sh kaggle_upload_bundle
```
Upload `kaggle_upload_bundle` as a private Kaggle Dataset (and attach `google/gemma-4-12B-it-qat-w4a16-ct` from Kaggle Models or Hugging Face).

### Step 5: Submit on Kaggle (`GPU T4 x2`, Internet Disabled)
Copy `kaggle_notebook/kaggle_submit_vllm.py` into a Kaggle Notebook, verify the 1–3 row interactive run produces `/kaggle/working/submission.parquet`, save & commit, and click **Submit to Competition**.
