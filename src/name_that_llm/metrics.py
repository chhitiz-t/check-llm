"""Competition evaluation metric (Balanced Accuracy) and per-model diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, log_loss

from name_that_llm.constants import IDX_TO_MODEL, NUM_CLASSES, VALID_MODELS


def compute_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Official Kaggle metric: unweighted mean of per-class recall across K=18 models."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    row_sums = cm.sum(axis=1)
    present = row_sums > 0
    if not np.any(present):
        return 0.0
    recalls = np.zeros(NUM_CLASSES, dtype=np.float64)
    recalls[present] = np.diag(cm)[present] / row_sums[present]
    return float(recalls[present].mean())


def evaluate_predictions(y_true: np.ndarray, probs: np.ndarray, preds: np.ndarray | None = None) -> dict[str, Any]:
    if preds is None:
        preds = probs.argmax(axis=1)

    bal_acc = compute_balanced_accuracy(y_true, preds)
    raw_acc = float((y_true == preds).mean())
    clipped = np.clip(probs, 1e-7, 1.0 - 1e-7)
    clipped = clipped / clipped.sum(axis=1, keepdims=True)
    ce_loss = float(log_loss(y_true, clipped, labels=list(range(NUM_CLASSES))))

    cm = confusion_matrix(y_true, preds, labels=list(range(NUM_CLASSES)))
    per_class_rows = []
    for idx in range(NUM_CLASSES):
        support = int(cm[idx].sum())
        correct = int(cm[idx, idx])
        recall = float(correct / support) if support > 0 else float("nan")
        pred_count = int(cm[:, idx].sum())
        precision = float(correct / pred_count) if pred_count > 0 else 0.0
        per_class_rows.append(
            {
                "model": IDX_TO_MODEL[idx],
                "support": support,
                "predicted": pred_count,
                "recall": recall,
                "precision": precision,
            }
        )

    # Top confused pairs (off-diagonal counts)
    cm_off = cm.copy()
    np.fill_diagonal(cm_off, 0)
    flat_order = np.argsort(cm_off.ravel())[::-1]
    top_confusions = []
    for flat_idx in flat_order[:10]:
        count = int(cm_off.ravel()[flat_idx])
        if count == 0:
            break
        true_i, pred_j = divmod(int(flat_idx), NUM_CLASSES)
        top_confusions.append(
            {
                "true_model": VALID_MODELS[true_i],
                "pred_model": VALID_MODELS[pred_j],
                "count": count,
            }
        )

    return {
        "balanced_accuracy": bal_acc,
        "sklearn_balanced_accuracy": float(balanced_accuracy_score(y_true, preds)),
        "accuracy": raw_acc,
        "log_loss": ce_loss,
        "per_class": pd.DataFrame(per_class_rows),
        "top_confusions": top_confusions,
    }
