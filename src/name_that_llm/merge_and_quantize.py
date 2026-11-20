"""Merge LoRA adapter into Gemma 4 12B and optionally export W4A16 compressed-tensors for vLLM.

When to use which mode on Kaggle T4x2:
1. Default (`adapter_mode: lora`):
   Keep `google/gemma-4-12B-it-qat-w4a16-ct` (~8.3 GB) untouched and pass your 5 fold LoRA
   adapters (~65 MB each!) via `LoRARequest` in vLLM. Uses <9 GB total disk space for all 5 folds!
2. Alternative (`adapter_mode: merged`):
   Merge your best fold (or LoRA-souped average of 5 adapters) into FP16/BF16 and compress to
   `W4A16` (`compressed-tensors`) using `llmcompressor` so vLLM runs pure W4A16 with `enable_lora=False`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge_lora_to_base(
    base_model_id: str,
    adapter_dir: str | Path,
    merged_out_dir: str | Path,
    dtype_str: str = "float16",
) -> Path:
    dtype = torch.float16 if dtype_str == "float16" else torch.bfloat16
    print(f"Loading base model {base_model_id} in {dtype_str}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        device_map="cpu",
        trust_remote_code=True,
    )
    print(f"Merging LoRA adapter from {adapter_dir}...")
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    merged = peft_model.merge_and_unload()

    out_path = Path(merged_out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out_path), safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir), trust_remote_code=True)
    tokenizer.save_pretrained(str(out_path))
    print(f"Saved merged checkpoint -> {out_path}")
    return out_path


def quantize_merged_w4a16(merged_dir: str | Path, w4a16_out_dir: str | Path) -> Path:
    """Export weight-only W4A16 (`group_size=32`) compressed-tensors checkpoint for vLLM T4."""
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier

    recipe = QuantizationModifier(
        targets="Linear",
        scheme="W4A16",
        ignore=["lm_head"],
    )
    out_path = Path(w4a16_out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    print(f"Running W4A16 compressed-tensors quantization -> {out_path}...")
    oneshot(
        model=str(merged_dir),
        recipe=recipe,
        output_dir=str(out_path),
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge LoRA & optionally quantize W4A16 for vLLM")
    parser.add_argument("--base-model", type=str, default="google/gemma-4-12B-it")
    parser.add_argument("--adapter-dir", type=str, required=True)
    parser.add_argument("--merged-dir", type=str, required=True)
    parser.add_argument("--w4a16-dir", type=str, default="")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16"])
    args = parser.parse_args()

    merged_path = merge_lora_to_base(args.base_model, args.adapter_dir, args.merged_dir, args.dtype)
    if args.w4a16_dir:
        quantize_merged_w4a16(merged_path, args.w4a16_dir)


if __name__ == "__main__":
    main()
