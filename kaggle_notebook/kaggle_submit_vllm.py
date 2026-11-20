"""Self-contained Kaggle Code Competition submission script (Gemma 4 12B + Dual-T4 vLLM).

Paste into a Kaggle Notebook (Accelerator: GPU T4 x2, Internet: OFF) with attached datasets:
1. Competition data: `/kaggle/input/competitions/name-that-llm`
2. Base QAT model:   `/kaggle/input/gemma-4-12b-qat-w4a16` (contains `google/gemma-4-12B-it-qat-w4a16-ct`)
3. Trained bundle:   `/kaggle/input/ntl-gemma4-12b-bundle` (wheelhouse + 5 LoRA adapters + postprocess_params.json)
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Paths (update dataset mount names to match your attached Kaggle Datasets)
# ---------------------------------------------------------------------------
COMP_TEST_CSV = Path("/kaggle/input/competitions/name-that-llm/test.csv")
BASE_MODEL_DIR = Path("/kaggle/input/gemma-4-12b-qat-w4a16")
BUNDLE_DIR = Path("/kaggle/input/ntl-gemma4-12b-bundle")
OUTPUT_PARQUET = Path("/kaggle/working/submission.parquet")


def ensure_offline_packages() -> None:
    """Install offline vLLM + project wheel if not already installed."""
    try:
        import name_that_llm  # noqa: F401
        import vllm  # noqa: F401

        return
    except ImportError:
        pass

    wheelhouse = BUNDLE_DIR / "wheelhouse"
    src_pkg = BUNDLE_DIR / "src_pkg"

    if wheelhouse.exists():
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-index",
                f"--find-links={wheelhouse}",
                "vllm",
            ]
        )
    if src_pkg.exists():
        wheels = sorted(glob.glob(str(src_pkg / "*.whl")))
        if wheels:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-deps", wheels[-1]])


def main() -> None:
    t0 = time.time()
    ensure_offline_packages()

    from name_that_llm.data import load_and_validate_test
    from name_that_llm.infer_vllm_dual_t4 import run_dual_gpu_vllm_inference
    from name_that_llm.postprocess import apply_log_multipliers, load_postprocess_params
    from name_that_llm.submission import write_submission_parquet

    test_df = load_and_validate_test(COMP_TEST_CSV)
    print(f"Loaded test.csv with {len(test_df)} passages.")

    adapter_dirs = sorted(glob.glob(str(BUNDLE_DIR / "adapters" / "fold_*")))
    postprocess_json = BUNDLE_DIR / "adapters" / "postprocess_params.json"

    # Interactive commit smoke-run only has 1-3 rows -> use 1 GPU to skip spawning 2 engines
    num_gpus = 1 if len(test_df) <= 8 else 2

    vllm_cfg = {
        "dtype": "float16",
        "max_model_len": 1024,
        "gpu_memory_utilization": 0.90,
        "enforce_eager": True,
        "enable_prefix_caching": True,
        "max_lora_rank": 32,
        "lora_dtype": "float16",
        "top_logprobs": 20,
        "num_gpus": num_gpus,
    }

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    probs = run_dual_gpu_vllm_inference(
        test_df=test_df,
        model_path=str(BASE_MODEL_DIR),
        adapter_dirs=adapter_dirs,
        vllm_cfg=vllm_cfg,
        max_passage_chars=2400,
        include_stylometry_header=True,
    )

    if postprocess_json.exists():
        log_mult = load_postprocess_params(postprocess_json)
        probs = apply_log_multipliers(probs, log_mult)

    pred_indices = probs.argmax(axis=1)
    out = write_submission_parquet(
        test_ids=test_df["id"],
        pred_indices=pred_indices,
        output_path=OUTPUT_PARQUET,
    )
    elapsed = (time.time() - t0) / 60.0
    print(f"Done in {elapsed:.2f} min! Saved valid submission -> {out}")


if __name__ == "__main__":
    main()
