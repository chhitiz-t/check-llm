"""Dual-T4 parallel vLLM inference orchestrator -> post-processing -> `submission.parquet`."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from name_that_llm.config import load_config
from name_that_llm.constants import NUM_CLASSES
from name_that_llm.data import load_and_validate_test
from name_that_llm.postprocess import apply_log_multipliers, load_postprocess_params
from name_that_llm.submission import write_submission_parquet


def detect_gpu_count(requested_gpus: int) -> int:
    try:
        import torch

        avail = torch.cuda.device_count()
        return max(1, min(requested_gpus, avail)) if avail > 0 else 1
    except Exception:
        return max(1, requested_gpus)


def run_dual_gpu_vllm_inference(
    test_df: pd.DataFrame,
    model_path: str,
    adapter_dirs: list[str],
    vllm_cfg: dict[str, object],
    max_passage_chars: int = 2400,
) -> np.ndarray:
    n_rows = len(test_df)
    requested_gpus = int(vllm_cfg.get("num_gpus", 2))
    n_workers = min(detect_gpu_count(requested_gpus), n_rows)

    shards = np.array_split(np.arange(n_rows), n_workers)

    with tempfile.TemporaryDirectory(prefix="vllm_gemma4_shards_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        procs: list[tuple[int, subprocess.Popen[bytes], Path]] = []

        for gpu_id, idx_slice in enumerate(shards):
            shard_df = test_df.iloc[idx_slice].reset_index(drop=True)
            shard_parquet = tmp_path / f"shard_{gpu_id}.parquet"
            shard_out_npy = tmp_path / f"probs_{gpu_id}.npy"
            shard_df.to_parquet(shard_parquet, index=False)

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            env.setdefault("TOKENIZERS_PARALLELISM", "false")
            env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

            cmd = [
                sys.executable,
                "-m",
                "name_that_llm.vllm_worker",
                "--shard-parquet",
                str(shard_parquet),
                "--output-npy",
                str(shard_out_npy),
                "--model-path",
                str(model_path),
                "--adapter-dirs-json",
                json.dumps([str(p) for p in adapter_dirs]),
                "--dtype",
                str(vllm_cfg.get("dtype", "float16")),
                "--max-model-len",
                str(vllm_cfg.get("max_model_len", 2048)),
                "--gpu-memory-utilization",
                str(vllm_cfg.get("gpu_memory_utilization", 0.90)),
                "--max-lora-rank",
                str(vllm_cfg.get("max_lora_rank", 32)),
                "--lora-dtype",
                str(vllm_cfg.get("lora_dtype", "float16")),
                "--top-logprobs",
                str(vllm_cfg.get("top_logprobs", 20)),
                "--max-passage-chars",
                str(max_passage_chars),
            ]
            print(f"[Orchestrator] Launching vLLM worker on GPU {gpu_id} ({len(shard_df)} passages)...")
            proc = subprocess.Popen(cmd, env=env)
            procs.append((gpu_id, proc, shard_out_npy))

        shard_arrays: list[np.ndarray] = []
        for gpu_id, proc, shard_out_npy in procs:
            ret = proc.wait()
            if ret != 0:
                raise RuntimeError(f"vLLM worker on GPU {gpu_id} exited with non-zero code {ret}")
            arr = np.load(shard_out_npy)
            if arr.ndim != 2 or arr.shape[1] != NUM_CLASSES:
                raise RuntimeError(f"Unexpected shard output shape from GPU {gpu_id}: {arr.shape}")
            shard_arrays.append(arr)

        full_probs = np.vstack(shard_arrays)
        if full_probs.shape != (n_rows, NUM_CLASSES):
            raise RuntimeError(f"Final probability shape mismatch: {full_probs.shape}")
        return full_probs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Dual-T4 vLLM inference & write submission.parquet")
    parser.add_argument("--config", type=str, default="configs/gemma4_12b_vllm.yaml")
    parser.add_argument("--test-csv", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--adapter-dirs", type=str, nargs="*", default=[])
    parser.add_argument("--postprocess-json", type=str, default="")
    parser.add_argument("--save-probs-npy", type=str, default="")
    parser.add_argument("--output", type=str, default="submission.parquet")
    args = parser.parse_args()

    cfg = load_config(args.config)
    test_df = load_and_validate_test(args.test_csv)

    probs = run_dual_gpu_vllm_inference(
        test_df=test_df,
        model_path=args.model_path,
        adapter_dirs=list(args.adapter_dirs),
        vllm_cfg=cfg["vllm"],
        max_passage_chars=int(cfg["model"]["max_passage_chars"]),
    )

    if args.save_probs_npy:
        Path(args.save_probs_npy).parent.mkdir(parents=True, exist_ok=True)
        np.save(args.save_probs_npy, probs)

    if args.postprocess_json and Path(args.postprocess_json).exists():
        log_mult = load_postprocess_params(args.postprocess_json)
        probs = apply_log_multipliers(probs, log_mult)

    pred_indices = probs.argmax(axis=1)
    out_path = write_submission_parquet(
        test_ids=test_df["id"],
        pred_indices=pred_indices,
        output_path=args.output,
    )
    print(f"Wrote validated submission ({len(test_df)} rows) -> {out_path}")


if __name__ == "__main__":
    main()
