"""QLoRA fine-tuning of Gemma 4 12B with 18-token restricted Balanced-Accuracy loss.

Key trick:
Instead of computing standard 262,144-vocab causal LM loss, `RestrictedLabelTrainer` slices
the logits at the final target position onto ONLY the 18 class-code token IDs ('A'..'R') and
applies inverse-class-frequency weighted CrossEntropyLoss + label smoothing. This matches
18-way sequence classification mathematically while keeping full compatibility with vLLM's
`LLM.generate(max_tokens=1, logprobs=20)` engine!
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from name_that_llm.config import load_config
from name_that_llm.constants import CLASS_CODES, NUM_CLASSES
from name_that_llm.data import (
    compute_balanced_class_weights,
    create_folds,
    load_and_validate_train,
    tokenize_sft_example,
)
from name_that_llm.metrics import evaluate_predictions


def resolve_code_token_ids(tokenizer: Any) -> list[int]:
    token_ids: list[int] = []
    for code in CLASS_CODES:
        ids = tokenizer.encode(code, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Class code {code!r} did not map to 1 token: {ids}")
        token_ids.append(int(ids[0]))
    if len(set(token_ids)) != NUM_CLASSES:
        raise ValueError("Duplicate token IDs encountered across A..R!")
    return token_ids


@dataclass
class LeftOrRightPaddedCollator:
    pad_token_id: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        batch_input_ids = []
        batch_mask = []
        batch_labels = []
        batch_class_idx = []

        for f in features:
            pad_n = max_len - len(f["input_ids"])
            batch_input_ids.append(f["input_ids"] + [self.pad_token_id] * pad_n)
            batch_mask.append(f["attention_mask"] + [0] * pad_n)
            batch_labels.append(f["labels"] + [-100] * pad_n)
            batch_class_idx.append(int(f["label_idx"]))

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
            "class_idx": torch.tensor(batch_class_idx, dtype=torch.long),
        }


class RestrictedLabelTrainer(Trainer):
    """Computes class-weighted 18-way cross-entropy at the exact label token position."""

    def __init__(
        self,
        *args: Any,
        code_token_ids: list[int],
        class_weights: np.ndarray | None = None,
        label_smoothing: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.code_token_ids = code_token_ids
        self.class_weights_np = class_weights
        self.label_smoothing = label_smoothing

    def _extract_18_logits(
        self, model: nn.Module, inputs: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        class_idx = inputs["class_idx"]
        labels = inputs["labels"]
        model_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }
        outputs = model(**model_inputs)
        logits = outputs.logits  # [B, T, V]

        # In causal LM, logits[:, t - 1, :] predicts token at position t where labels[:, t] != -100
        target_mask = labels.ne(-100)
        target_pos = target_mask.int().argmax(dim=1)  # [B]
        pred_pos = torch.clamp(target_pos - 1, min=0)
        batch_indices = torch.arange(logits.size(0), device=logits.device)

        step_logits = logits[batch_indices, pred_pos, :]  # [B, V]
        token_idx_tensor = torch.tensor(self.code_token_ids, device=logits.device, dtype=torch.long)
        logits_18 = step_logits.index_select(dim=-1, index=token_idx_tensor).float()  # [B, 18]
        return logits_18, class_idx

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        logits_18, class_idx = self._extract_18_logits(model, inputs)
        weight = None
        if self.class_weights_np is not None:
            weight = torch.tensor(self.class_weights_np, device=logits_18.device, dtype=torch.float32)

        loss_fn = nn.CrossEntropyLoss(weight=weight, label_smoothing=self.label_smoothing)
        loss = loss_fn(logits_18, class_idx)
        return (loss, logits_18) if return_outputs else loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        with torch.no_grad():
            loss, logits_18 = self.compute_loss(model, inputs, return_outputs=True)
        if prediction_loss_only:
            return (loss.detach(), None, None)
        return (loss.detach(), logits_18.detach(), inputs["class_idx"].detach())


def run_fold_training(config_path: str | Path, fold: int, folds_parquet: str | Path | None = None) -> None:
    cfg = load_config(config_path)
    comp_cfg = cfg["competition"]
    model_cfg = cfg["model"]
    qlora_cfg = cfg["qlora"]
    cv_cfg = cfg["cv"]

    if folds_parquet and Path(folds_parquet).exists():
        df = pd.read_parquet(folds_parquet)
    else:
        raw_train = Path(comp_cfg["local_data_dir"]) / "train.csv"
        df = load_and_validate_train(raw_train)
        df = create_folds(
            df,
            n_splits=int(cv_cfg["n_splits"]),
            seed=int(cv_cfg["seed"]),
            group_col=cv_cfg["group_col"],
            target_col=cv_cfg["target_col"],
        )

    train_df = df[df["fold"] != fold].reset_index(drop=True)
    val_df = df[df["fold"] == fold].reset_index(drop=True)
    print(f"[Fold {fold}] Train rows={len(train_df)}, Val rows={len(val_df)}")

    model_id = model_cfg["train_model_id"]
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    code_token_ids = resolve_code_token_ids(tokenizer)

    def _encode_df(split_df: pd.DataFrame) -> Dataset:
        records = []
        for row in split_df.itertuples(index=False):
            tok = tokenize_sft_example(
                tokenizer=tokenizer,
                passage=row.text,
                label_code=row.label_code,
                max_seq_length=int(qlora_cfg["max_seq_length"]),
                max_passage_chars=int(model_cfg["max_passage_chars"]),
                enable_thinking=bool(model_cfg["enable_thinking"]),
            )
            tok["label_idx"] = int(row.label_idx)
            records.append(tok)
        return Dataset.from_list(records)

    train_ds = _encode_df(train_df)
    val_ds = _encode_df(val_df)

    compute_dtype = (
        torch.bfloat16 if qlora_cfg.get("compute_dtype", "bfloat16") == "bfloat16" else torch.float16
    )
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=bool(qlora_cfg.get("load_in_4bit", True)),
        bnb_4bit_quant_type=qlora_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=bool(qlora_cfg.get("bnb_4bit_use_double_quant", True)),
        bnb_4bit_compute_dtype=compute_dtype,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    base_model.config.use_cache = False
    base_model = prepare_model_for_kbit_training(base_model)

    lora_config = LoraConfig(
        r=int(qlora_cfg["lora_r"]),
        lora_alpha=int(qlora_cfg["lora_alpha"]),
        lora_dropout=float(qlora_cfg["lora_dropout"]),
        target_modules=list(qlora_cfg["target_modules"]),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    class_weights = (
        compute_balanced_class_weights(train_df["label_idx"].to_numpy())
        if qlora_cfg.get("use_balanced_class_weights", True)
        else None
    )

    out_dir = Path(comp_cfg["artifacts_dir"]) / f"lora_fold_{fold}"
    out_dir.mkdir(parents=True, exist_ok=True)

    def _compute_metrics(eval_pred: Any) -> dict[str, float]:
        logits_18, y_true = eval_pred
        shifted = logits_18 - logits_18.max(axis=1, keepdims=True)
        probs = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
        res = evaluate_predictions(y_true, probs)
        return {
            "balanced_accuracy": res["balanced_accuracy"],
            "accuracy": res["accuracy"],
            "log_loss": res["log_loss"],
        }

    use_bf16 = compute_dtype == torch.bfloat16
    training_args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=float(qlora_cfg["num_train_epochs"]),
        per_device_train_batch_size=int(qlora_cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(qlora_cfg["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(qlora_cfg["gradient_accumulation_steps"]),
        learning_rate=float(qlora_cfg["learning_rate"]),
        weight_decay=float(qlora_cfg["weight_decay"]),
        warmup_ratio=float(qlora_cfg["warmup_ratio"]),
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="balanced_accuracy",
        greater_is_better=True,
        bf16=use_bf16,
        fp16=not use_bf16,
        gradient_checkpointing=True,
        logging_steps=20,
        remove_unused_columns=False,
        save_total_limit=2,
        seed=int(cv_cfg["seed"]) + fold,
        report_to="none",
    )

    trainer = RestrictedLabelTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=LeftOrRightPaddedCollator(pad_token_id=tokenizer.pad_token_id),
        compute_metrics=_compute_metrics,
        code_token_ids=code_token_ids,
        class_weights=class_weights,
        label_smoothing=float(qlora_cfg.get("label_smoothing", 0.0)),
    )

    trainer.train()

    # Save lightweight LoRA adapter for vLLM multi-LoRA serving
    adapter_dir = out_dir / "best_adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    # Save fold OOF predictions for post-processing & ensembling
    pred_output = trainer.predict(val_ds)
    logits_18 = pred_output.predictions
    shifted = logits_18 - logits_18.max(axis=1, keepdims=True)
    val_probs = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    oof_path = Path(comp_cfg["artifacts_dir"]) / f"oof_fold_{fold}.npz"
    np.savez_compressed(
        oof_path,
        ids=val_df["id"].to_numpy(),
        y_true=val_df["label_idx"].to_numpy(),
        probs=val_probs,
    )
    val_metrics = evaluate_predictions(val_df["label_idx"].to_numpy(), val_probs)
    print(
        f"[Fold {fold}] Best Val Balanced Accuracy = {val_metrics['balanced_accuracy']:.5f} "
        f"| Saved adapter -> {adapter_dir}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Gemma 4 12B QLoRA fold")
    parser.add_argument("--config", type=str, default="configs/gemma4_12b_vllm.yaml")
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--folds-parquet", type=str, default="data/processed/train_folds.parquet")
    args = parser.parse_args()
    run_fold_training(args.config, args.fold, args.folds_parquet)


if __name__ == "__main__":
    main()
