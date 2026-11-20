"""Strict validation and writing of `submission.parquet` per Kaggle rules."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from name_that_llm.constants import IDX_TO_MODEL, VALID_MODELS


def build_submission_df(test_ids: pd.Series | list[str], pred_indices: np.ndarray) -> pd.DataFrame:
    models = [IDX_TO_MODEL[int(idx)] for idx in pred_indices]
    return pd.DataFrame({"id": list(test_ids), "model": models})


def validate_submission_df(sub_df: pd.DataFrame, expected_ids: pd.Series | list[str] | None = None) -> None:
    """Enforce every rejection check listed on the Name That LLM Evaluation page."""
    if list(sub_df.columns) != ["id", "model"]:
        raise ValueError(f"Submission columns must be exactly ['id', 'model'], got {list(sub_df.columns)}")
    if len(sub_df) == 0:
        raise ValueError("Submission DataFrame is empty")
    if sub_df["id"].isnull().any() or sub_df["model"].isnull().any():
        raise ValueError("Submission contains null values")
    if sub_df["id"].duplicated().any():
        raise ValueError("Submission contains duplicate IDs")

    invalid_labels = set(sub_df["model"].unique()) - set(VALID_MODELS)
    if invalid_labels:
        raise ValueError(f"Submission contains invalid model labels: {sorted(invalid_labels)}")

    if expected_ids is not None:
        exp_list = list(expected_ids)
        if len(sub_df) != len(exp_list) or list(sub_df["id"]) != exp_list:
            if set(sub_df["id"]) != set(exp_list):
                raise ValueError("Submission IDs do not match test.csv IDs")


def write_submission_parquet(
    test_ids: pd.Series | list[str],
    pred_indices: np.ndarray,
    output_path: str | Path = "submission.parquet",
) -> Path:
    sub_df = build_submission_df(test_ids=test_ids, pred_indices=pred_indices)
    validate_submission_df(sub_df, expected_ids=test_ids)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub_df.to_parquet(out_path, index=False)

    # Re-read from disk to guarantee Parquet integrity
    reloaded = pd.read_parquet(out_path)
    validate_submission_df(reloaded, expected_ids=test_ids)
    return out_path
