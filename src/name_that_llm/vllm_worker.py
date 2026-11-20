"""Single-GPU isolated vLLM worker process for Gemma 4 12B 18-way logprob scoring.

Why isolated worker processes?
On Kaggle's `2x T4` (2 x 16 GB VRAM), instantiating two `vllm.LLM` engines inside one Python
process causes CUDA context / distributed group collisions. Spawning one process per GPU with
`CUDA_VISIBLE_DEVICES=0` and `CUDA_VISIBLE_DEVICES=1` gives each 8.3 GB `w4a16-ct` Gemma 4 12B
engine a dedicated 16 GB T4 card and doubles throughput cleanly.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from name_that_llm.constants import CLASS_CODES, NUM_CLASSES
from name_that_llm.prompts import render_inference_prompt


def build_candidate_token_map(tokenizer: Any) -> dict[int, list[int]]:
    """Map each class index (0..17) to candidate token IDs for both 'A' and ' A'."""
    mapping: dict[int, list[int]] = {}
    for idx, code in enumerate(CLASS_CODES):
        cands: list[int] = []
        for variant in (code, f" {code}"):
            ids = tokenizer.encode(variant, add_special_tokens=False)
            if len(ids) == 1 and int(ids[0]) not in cands:
                cands.append(int(ids[0]))
        if not cands:
            raise RuntimeError(f"Could not find single-token ID for class code {code!r}")
        mapping[idx] = cands
    return mapping


def extract_18_probs_from_output(
    request_output: Any,
    cand_token_map: dict[int, list[int]],
    floor_logprob: float = -25.0,
) -> np.ndarray:
    """Convert vLLM step-0 top logprobs into a normalized 18-class probability vector."""
    raw_logprobs = np.full(NUM_CLASSES, floor_logprob, dtype=np.float64)
    comp = request_output.outputs[0]

    if comp.logprobs and len(comp.logprobs) > 0:
        step0: dict[int, Any] = comp.logprobs[0]
        for cls_idx, token_ids in cand_token_map.items():
            matched_vals = []
            for tid in token_ids:
                if tid in step0:
                    entry = step0[tid]
                    lp = float(entry.logprob) if hasattr(entry, "logprob") else float(entry)
                    matched_vals.append(lp)
            if matched_vals:
                # Combine 'A' and ' A' probability mass safely via logsumexp
                m = max(matched_vals)
                raw_logprobs[cls_idx] = m + math.log(sum(math.exp(v - m) for v in matched_vals))
    else:
        # Fallback: parse generated text character if logprobs were unavailable
        txt = (comp.text or "").strip()[:1].upper()
        if txt in CLASS_CODES:
            raw_logprobs[CLASS_CODES.index(txt)] = 0.0

    shifted = raw_logprobs - raw_logprobs.max()
    exp_p = np.exp(shifted)
    return exp_p / exp_p.sum()


def run_worker(
    shard_parquet: str | Path,
    output_npy: str | Path,
    model_path: str,
    adapter_dirs: list[str],
    dtype: str = "float16",
    max_model_len: int = 2048,
    gpu_memory_utilization: float = 0.90,
    enforce_eager: bool = True,
    enable_prefix_caching: bool = True,
    max_lora_rank: int = 32,
    lora_dtype: str = "float16",
    top_logprobs: int = 20,
    max_passage_chars: int = 2400,
    enable_thinking: bool = False,
    tensor_parallel_size: int = 1,
) -> None:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    shard_df = pd.read_parquet(shard_parquet)
    use_lora = len(adapter_dirs) > 0

    llm_kwargs: dict[str, Any] = {
        "model": model_path,
        "dtype": dtype,
        "tensor_parallel_size": tensor_parallel_size,
        "max_model_len": max_model_len,
        "gpu_memory_utilization": gpu_memory_utilization,
        "enforce_eager": enforce_eager,
        "enable_prefix_caching": enable_prefix_caching,
        "limit_mm_per_prompt": {"image": 0, "audio": 0},
        "max_logprobs": max(top_logprobs, 20),
        "trust_remote_code": True,
    }
    if use_lora:
        llm_kwargs.update(
            {
                "enable_lora": True,
                "max_loras": 1,
                "max_lora_rank": max_lora_rank,
                "lora_dtype": lora_dtype,
            }
        )

    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    cand_token_map = build_candidate_token_map(tokenizer)

    prompts = [
        render_inference_prompt(
            tokenizer=tokenizer,
            passage=text,
            max_chars=max_passage_chars,
            enable_thinking=enable_thinking,
        )
        for text in shard_df["text"].tolist()
    ]

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        logprobs=top_logprobs,
    )

    accum_probs = np.zeros((len(prompts), NUM_CLASSES), dtype=np.float64)

    if use_lora:
        for idx, adapter_path in enumerate(adapter_dirs, start=1):
            lora_req = LoRARequest(
                lora_name=f"fold_{idx}",
                lora_int_id=idx,
                lora_path=str(adapter_path),
            )
            outputs = llm.generate(prompts, sampling_params, lora_request=lora_req)
            fold_probs = np.vstack(
                [extract_18_probs_from_output(out, cand_token_map) for out in outputs]
            )
            accum_probs += fold_probs
        accum_probs /= float(len(adapter_dirs))
    else:
        outputs = llm.generate(prompts, sampling_params)
        accum_probs = np.vstack(
            [extract_18_probs_from_output(out, cand_token_map) for out in outputs]
        )

    out_file = Path(output_npy)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_file, accum_probs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-GPU vLLM Gemma 4 12B worker")
    parser.add_argument("--shard-parquet", type=str, required=True)
    parser.add_argument("--output-npy", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--adapter-dirs-json", type=str, default="[]")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--enforce-eager", action="store_true", default=True)
    parser.add_argument("--enable-prefix-caching", action="store_true", default=True)
    parser.add_argument("--max-lora-rank", type=int, default=32)
    parser.add_argument("--lora-dtype", type=str, default="float16")
    parser.add_argument("--top-logprobs", type=int, default=20)
    parser.add_argument("--max-passage-chars", type=int, default=2400)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    args = parser.parse_args()

    adapter_dirs = json.loads(args.adapter_dirs_json)
    run_worker(
        shard_parquet=args.shard_parquet,
        output_npy=args.output_npy,
        model_path=args.model_path,
        adapter_dirs=adapter_dirs,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        enable_prefix_caching=args.enable_prefix_caching,
        max_lora_rank=args.max_lora_rank,
        lora_dtype=args.lora_dtype,
        top_logprobs=args.top_logprobs,
        max_passage_chars=args.max_passage_chars,
        tensor_parallel_size=args.tensor_parallel_size,
    )


if __name__ == "__main__":
    main()
