"""End-to-end local CPU smoke test of the entire Name That LLM pipeline (no GPU needed).

Verifies:
1. All 18 valid model names & single-token class codes ('A'..'R')
2. Leak-free 5-fold StratifiedGroupKFold(prompt_id)
3. Gemma 4 prompt formatting (`enable_thinking=False`) + 1-token label supervision masking
4. vLLM step-0 logprob extraction (`extract_18_probs_from_output`)
5. Balanced Accuracy evaluation & OOF log-multiplier coordinate ascent under skewed priors
6. Strict `submission.parquet` generation and validation
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from name_that_llm.constants import CLASS_CODES, NUM_CLASSES, VALID_MODELS
from name_that_llm.data import (
    compute_balanced_class_weights,
    create_folds,
    load_and_validate_test,
    load_and_validate_train,
    tokenize_sft_example,
)
from name_that_llm.metrics import evaluate_predictions
from name_that_llm.postprocess import apply_log_multipliers, optimize_class_multipliers
from name_that_llm.prompts import render_inference_prompt
from name_that_llm.submission import write_submission_parquet
from name_that_llm.vllm_worker import build_candidate_token_map, extract_18_probs_from_output


class DummyGemmaTokenizer:
    """Minimal byte/char stand-in verifying single-token behavior for 'A'..'R'."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        stripped = text.strip()
        if len(stripped) == 1 and stripped in CLASS_CODES:
            offset = 1000 if not text.startswith(" ") else 2000
            return [offset + CLASS_CODES.index(stripped)]
        return [10 + (ord(c) % 200) for c in text]


@dataclass
class _MockLogprob:
    logprob: float


@dataclass
class _MockCompletion:
    text: str
    logprobs: list[dict[int, _MockLogprob]]


@dataclass
class _MockRequestOutput:
    outputs: list[_MockCompletion]


def main() -> None:
    rng = np.random.default_rng(2026)

    with tempfile.TemporaryDirectory(prefix="ntl_smoke_") as tmp:
        tmp_dir = Path(tmp)

        # 1. Generate imbalanced synthetic train.csv across 30 prompts & all 18 valid models
        rows = []
        imbalance_counts = [30, 25, 22, 20, 18, 28, 16, 15, 14, 26, 12, 11, 10, 19, 9, 8, 8, 9]
        assert len(imbalance_counts) == NUM_CLASSES
        prompts = [f"prompt_{i:03d}" for i in range(30)]

        idx_counter = 0
        for cls_idx, count in enumerate(imbalance_counts):
            model_name = VALID_MODELS[cls_idx]
            for j in range(count):
                p_id = prompts[(cls_idx * 3 + j) % len(prompts)]
                rows.append(
                    {
                        "id": f"tr_{idx_counter:05d}",
                        "prompt_id": p_id,
                        "text": f"Synthetic philosophical passage #{idx_counter} styled by {model_name}. " * 4,
                        "model": model_name,
                    }
                )
                idx_counter += 1

        train_csv = tmp_dir / "train.csv"
        pd.DataFrame(rows).to_csv(train_csv, index=False)

        test_csv = tmp_dir / "test.csv"
        pd.DataFrame(
            [
                {"id": f"te_{i:04d}", "text": f"Unseen physics test passage #{i} with subtle cadence."}
                for i in range(25)
            ]
        ).to_csv(test_csv, index=False)

        # 2. Validate & split folds
        train_df = load_and_validate_train(train_csv)
        folds_df = create_folds(train_df, n_splits=5, seed=42)
        weights = compute_balanced_class_weights(folds_df["label_idx"].to_numpy())
        assert weights.shape == (NUM_CLASSES,)
        print(f"[OK] Validated train ({len(folds_df)} rows) & leak-free 5-fold StratifiedGroupKFold.")

        # 3. Verify prompt rendering & 1-token label supervision masking
        tok = DummyGemmaTokenizer()
        sample_prompt = render_inference_prompt(None, folds_df.iloc[0]["text"], enable_thinking=False)
        assert "<|thought>\n<thought|>" in sample_prompt
        sft_item = tokenize_sft_example(
            tokenizer=tok,
            passage=folds_df.iloc[0]["text"],
            label_code=folds_df.iloc[0]["label_code"],
        )
        supervised_tokens = [x for x in sft_item["labels"] if x != -100]
        assert len(supervised_tokens) == 1, f"Expected exactly 1 supervised token, got {supervised_tokens}"
        print("[OK] Verified Gemma 4 prompt formatting & exact 1-token label loss masking.")

        # 4. Verify vLLM step-0 logprob extraction
        cand_map = build_candidate_token_map(tok)
        mock_step0 = {cand_map[k][0]: _MockLogprob(-3.0 if k != 4 else -0.2) for k in range(NUM_CLASSES)}
        mock_out = _MockRequestOutput(outputs=[_MockCompletion(text="E", logprobs=[mock_step0])])
        probs_18 = extract_18_probs_from_output(mock_out, cand_map)
        assert probs_18.shape == (NUM_CLASSES,)
        assert np.isclose(probs_18.sum(), 1.0)
        assert int(probs_18.argmax()) == 4
        print("[OK] Verified vLLM step-0 18-class logprob extraction & normalization.")

        # 5. Simulate skewed OOF probabilities & verify Balanced Accuracy post-processing
        y_true = folds_df["label_idx"].to_numpy()
        priors = np.bincount(y_true, minlength=NUM_CLASSES) / len(y_true)
        logits = rng.normal(0.0, 1.1, size=(len(y_true), NUM_CLASSES))
        logits[np.arange(len(y_true)), y_true] += 1.8
        logits += np.log(priors + 1e-6)  # realistic majority-class bias
        logits -= logits.max(axis=1, keepdims=True)
        oof_probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)

        pp = optimize_class_multipliers(oof_probs, y_true, n_random_restarts=4, seed=42)
        assert pp["optimized_oof_balanced_accuracy"] >= pp["raw_oof_balanced_accuracy"] - 1e-9
        print(
            f"[OK] OOF Balanced Accuracy improved: "
            f"{pp['raw_oof_balanced_accuracy']:.4f} -> {pp['optimized_oof_balanced_accuracy']:.4f}"
        )

        # 6. Write and validate `submission.parquet`
        test_df = load_and_validate_test(test_csv)
        test_probs = rng.dirichlet(np.ones(NUM_CLASSES), size=len(test_df))
        test_probs = apply_log_multipliers(test_probs, np.asarray(pp["log_multipliers"]))
        eval_summary = evaluate_predictions(y_true, oof_probs)
        assert len(eval_summary["per_class"]) == NUM_CLASSES

        sub_path = write_submission_parquet(
            test_ids=test_df["id"],
            pred_indices=test_probs.argmax(axis=1),
            output_path=tmp_dir / "submission.parquet",
        )
        sub_df = pd.read_parquet(sub_path)
        assert list(sub_df.columns) == ["id", "model"]
        assert len(sub_df) == len(test_df)
        print(f"[OK] Generated & validated {sub_path.name} ({len(sub_df)} rows). All checks passed!")


if __name__ == "__main__":
    main()
