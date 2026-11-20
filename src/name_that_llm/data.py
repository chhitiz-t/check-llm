"""Dataset loading, schema validation, StratifiedGroupKFold splitting, and SFT tokenization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from name_that_llm.constants import MODEL_TO_CODE, MODEL_TO_IDX, NUM_CLASSES, VALID_MODELS
from name_that_llm.prompts import render_inference_prompt


def load_and_validate_train(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"id", "prompt_id", "text", "model"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"train.csv missing required columns: {sorted(missing)}")
    if df[["id", "prompt_id", "text", "model"]].isnull().any().any():
        raise ValueError("train.csv contains null values in required columns")
    if df["id"].duplicated().any():
        raise ValueError("train.csv contains duplicate passage IDs")

    unknown_models = set(df["model"].unique()) - set(VALID_MODELS)
    if unknown_models:
        raise ValueError(f"train.csv contains unknown model labels: {sorted(unknown_models)}")

    df = df.copy()
    df["label_idx"] = df["model"].map(MODEL_TO_IDX).astype(int)
    df["label_code"] = df["model"].map(MODEL_TO_CODE)
    return df


def load_and_validate_test(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"id", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"test.csv missing required columns: {sorted(missing)}")
    if df["id"].isnull().any() or df["id"].duplicated().any():
        raise ValueError("test.csv has null or duplicate IDs")
    df = df.copy()
    df["text"] = df["text"].fillna("").astype(str)
    return df


def create_folds(
    df: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
    group_col: str = "prompt_id",
    target_col: str = "model",
) -> pd.DataFrame:
    """Create leak-free StratifiedGroupKFold splits grouped by `prompt_id`."""
    out = df.copy()
    out["fold"] = -1
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (_, val_idx) in enumerate(sgkf.split(out, out[target_col], out[group_col])):
        out.iloc[val_idx, out.columns.get_loc("fold")] = fold

    if (out["fold"] < 0).any():
        raise RuntimeError("Failed to assign all rows to a fold")

    # Verify zero prompt leakage across folds
    prompt_fold_counts = out.groupby(group_col)["fold"].nunique()
    if (prompt_fold_counts > 1).any():
        raise RuntimeError("Detected prompt_id leakage across folds!")
    return out


def compute_balanced_class_weights(labels: np.ndarray) -> np.ndarray:
    """Compute inverse-frequency weights normalized to mean=1.0 across all 18 classes."""
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    inv = 1.0 / counts
    weights = inv * (NUM_CLASSES / inv.sum())
    return weights.astype(np.float32)


def compute_class_priors(labels: np.ndarray, smoothing: float = 1.0) -> np.ndarray:
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float64) + smoothing
    return counts / counts.sum()


def tokenize_sft_example(
    tokenizer: Any,
    passage: str,
    label_code: str,
    max_seq_length: int = 768,
    max_passage_chars: int = 2400,
    enable_thinking: bool = False,
    include_stylometry_header: bool = True,
) -> dict[str, list[int]]:
    """Build input_ids & labels where loss is computed ONLY on the single class code token."""
    prompt_text = render_inference_prompt(
        tokenizer=tokenizer,
        passage=passage,
        max_chars=max_passage_chars,
        enable_thinking=enable_thinking,
        include_stylometry_header=include_stylometry_header,
    )
    prompt_ids: list[int] = tokenizer.encode(prompt_text, add_special_tokens=False)
    code_ids: list[int] = tokenizer.encode(label_code, add_special_tokens=False)
    if len(code_ids) != 1:
        raise ValueError(
            f"Expected class code {label_code!r} to tokenize to 1 token, got {code_ids}"
        )

    # Preserve the generation turn suffix (`<turn|>\n<|turn>model\n...`) if passage ever exceeds max_seq_length
    max_prompt_tokens = max_seq_length - 1
    if len(prompt_ids) > max_prompt_tokens:
        tail_keep = min(32, max_prompt_tokens // 4)
        head_keep = max_prompt_tokens - tail_keep
        prompt_ids = prompt_ids[:head_keep] + prompt_ids[-tail_keep:]

    input_ids = prompt_ids + code_ids
    labels = ([-100] * len(prompt_ids)) + code_ids
    attention_mask = [1] * len(input_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
