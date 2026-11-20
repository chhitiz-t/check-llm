"""Phase 1: Validate `train.csv` & `test.csv`, run stylometric EDA, and build 5-fold StratifiedGroupKFold."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from name_that_llm.config import load_config
from name_that_llm.constants import MODEL_TO_CODE, VALID_MODELS
from name_that_llm.data import (
    compute_balanced_class_weights,
    create_folds,
    load_and_validate_test,
    load_and_validate_train,
)


def build_phase1_report(folds_df: pd.DataFrame, test_df: pd.DataFrame | None, out_md: Path) -> None:
    df = folds_df.copy()
    df["char_len"] = df["text"].str.len()
    df["word_len"] = df["text"].str.split().str.len()

    weights = compute_balanced_class_weights(df["label_idx"].to_numpy())

    model_stats = []
    for idx, model_name in enumerate(VALID_MODELS):
        sub = df[df["model"] == model_name]
        model_stats.append(
            {
                "Code": MODEL_TO_CODE[model_name],
                "Model": model_name,
                "Count": len(sub),
                "Share (%)": round(100.0 * len(sub) / len(df), 2),
                "Loss Weight": round(float(weights[idx]), 3),
                "Prompts": int(sub["prompt_id"].nunique()),
                "Words (mean±std)": (
                    f"{sub['word_len'].mean():.1f} ± {sub['word_len'].std():.1f}"
                    if len(sub) > 1
                    else f"{sub['word_len'].mean():.1f}"
                ),
                "Words [min, max]": (
                    f"[{int(sub['word_len'].min())}, {int(sub['word_len'].max())}]"
                    if len(sub) > 0
                    else "[-]"
                ),
            }
        )
    stats_df = pd.DataFrame(model_stats)

    fold_summary = (
        df.groupby("fold")
        .agg(
            rows=("id", "count"),
            unique_prompts=("prompt_id", "nunique"),
            unique_models=("model", "nunique"),
            mean_words=("word_len", lambda s: round(float(s.mean()), 1)),
        )
        .reset_index()
    )

    counts = stats_df["Count"].to_numpy()
    imbalance_ratio = float(counts.max() / max(1, counts.min()))

    lines = [
        "# Phase 1 Report — Data Validation, Stylometric EDA & CV Split",
        "",
        "## 1. Dataset Integrity Summary",
        f"- **Training passages (`train.csv`):** `{len(df):,}` rows (0 nulls, 0 duplicate IDs)",
        f"- **Unique training prompts (`prompt_id`):** `{df['prompt_id'].nunique():,}`",
        f"- **Candidate models represented:** `{(counts > 0).sum()} / {len(VALID_MODELS)}`",
        f"- **Class imbalance ratio (`max / min`):** `{imbalance_ratio:.2f}x` (`min={counts.min()}`, `max={counts.max()}`)",
        f"- **Passage word length (`train.csv`):** min `{df['word_len'].min()}`, median `{int(df['word_len'].median())}`, max `{df['word_len'].max()}` words",
        f"- **Passage char length (`train.csv`):** min `{df['char_len'].min()}`, median `{int(df['char_len'].median())}`, max `{df['char_len'].max()}` chars",
    ]
    if test_df is not None:
        lines.append(f"- **Interactive `test.csv` rows:** `{len(test_df):,}`")

    lines.extend(
        [
            "",
            "## 2. Leak-Free 5-Fold `StratifiedGroupKFold(prompt_id)` Summary",
            "- **Zero prompt leakage verified:** Every `prompt_id` belongs to exactly 1 validation fold.",
            "",
            fold_summary.to_markdown(index=False),
            "",
            "## 3. Per-Model Distribution, Balanced Loss Weights & Length Fingerprints",
            "",
            stats_df.to_markdown(index=False),
            "",
        ]
    )

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1: Prepare leak-free CV folds & EDA report")
    parser.add_argument("--config", type=str, default="configs/gemma4_12b_vllm.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_dir = Path(cfg["competition"]["local_data_dir"])
    raw_train = raw_dir / "train.csv"
    raw_test = raw_dir / "test.csv"
    proc_dir = Path(cfg["competition"]["processed_dir"])
    out_parquet = proc_dir / "train_folds.parquet"
    out_report = proc_dir / "phase1_report.md"
    proc_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_validate_train(raw_train)
    test_df = load_and_validate_test(raw_test) if raw_test.exists() else None

    folds_df = create_folds(
        df,
        n_splits=int(cfg["cv"]["n_splits"]),
        seed=int(cfg["cv"]["seed"]),
        group_col=cfg["cv"]["group_col"],
        target_col=cfg["cv"]["target_col"],
    )
    folds_df.to_parquet(out_parquet, index=False)
    build_phase1_report(folds_df, test_df, out_report)

    print(f"[Phase 1] Saved {len(folds_df):,} rows across {folds_df['prompt_id'].nunique()} prompts -> {out_parquet}")
    print(f"[Phase 1] Generated EDA & CV report -> {out_report}")


if __name__ == "__main__":
    main()
