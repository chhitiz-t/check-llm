"""Assemble 5-fold OOF predictions, print diagnostics, and fit Balanced Accuracy multipliers."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from name_that_llm.config import load_config
from name_that_llm.metrics import evaluate_predictions
from name_that_llm.postprocess import (
    apply_log_multipliers,
    optimize_class_multipliers,
    save_postprocess_params,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit OOF Balanced Accuracy post-processing")
    parser.add_argument("--config", type=str, default="configs/gemma4_12b_vllm.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    artifacts_dir = Path(cfg["competition"]["artifacts_dir"])
    n_splits = int(cfg["cv"]["n_splits"])

    all_ids = []
    all_y = []
    all_probs = []

    found_folds = []
    for fold in range(n_splits):
        oof_file = artifacts_dir / f"oof_fold_{fold}.npz"
        if not oof_file.exists():
            continue
        found_folds.append(fold)
        data = np.load(oof_file, allow_pickle=True)
        all_ids.append(data["ids"])
        all_y.append(data["y_true"])
        all_probs.append(data["probs"])

    if not found_folds:
        raise FileNotFoundError(f"No oof_fold_*.npz files found in {artifacts_dir}")
    print(f"Loaded OOF predictions from fold(s): {found_folds} ({sum(len(y) for y in all_y)} rows)")

    y_true = np.concatenate(all_y, axis=0)
    oof_probs = np.vstack(all_probs)

    raw_eval = evaluate_predictions(y_true, oof_probs)
    print(f"\n[OOF Raw] Balanced Accuracy : {raw_eval['balanced_accuracy']:.5f}")
    print(f"[OOF Raw] Standard Accuracy : {raw_eval['accuracy']:.5f}")
    print(f"[OOF Raw] Multi-class LogLoss: {raw_eval['log_loss']:.5f}")

    pp_cfg = cfg["postprocess"]
    params = optimize_class_multipliers(
        oof_probs=oof_probs,
        y_true=y_true,
        use_prior_init=bool(pp_cfg.get("use_prior_adjustment", True)),
        n_random_restarts=int(pp_cfg.get("n_random_restarts", 8)),
        seed=int(pp_cfg.get("seed", 42)),
    )

    adj_probs = apply_log_multipliers(oof_probs, np.asarray(params["log_multipliers"]))
    adj_eval = evaluate_predictions(y_true, adj_probs)

    print(f"\n[OOF Prior-Adjusted] Balanced Accuracy: {params['prior_adjusted_balanced_accuracy']:.5f} (tau={params['prior_tau']:.3f})")
    print(f"[OOF Optimized]      Balanced Accuracy: {adj_eval['balanced_accuracy']:.5f}")
    print("\nPer-model recall after calibration:")
    print(adj_eval["per_class"].to_string(index=False))

    out_json = artifacts_dir / "postprocess_params.json"
    save_postprocess_params(params, out_json)
    print(f"\nSaved post-processing parameters -> {out_json}")


if __name__ == "__main__":
    main()
