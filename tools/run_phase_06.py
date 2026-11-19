"""Phase 06 training runner: Fit classifiers, regressors and save OOF predictions."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure root workspace is on python path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import numpy as np
import pandas as pd

from src.cutoff import apply_cutoff_and_targets, apply_eval_filter
from src.evaluate import compute_classification_metrics, compute_regression_metrics
from src.features import build_all_features
from src.load import load_config, load_parquet, load_raw_cdms, save_parquet
from src.models import (
    fit_predict_classifiers,
    fit_predict_classifiers_augmented,
    fit_predict_regressors,
)


def run_phase_06() -> None:
    cfg = load_config()
    high_thresh = float(cfg["thresholds"]["high_risk_log10"])

    features = load_parquet("data/processed/features_train.parquet")
    targets = load_parquet("data/processed/targets_train.parquet")
    folds = load_parquet("data/processed/folds.parquet")

    # Align on event_id for train split
    train_folds = folds[folds["split"] == "train"].copy()
    df_merged = (
        train_folds[["event_id", "split", "fold"]]
        .merge(targets, on="event_id")
        .merge(features, on="event_id")
    )

    X = df_merged.drop(
        columns=[
            "y_risk",
            "y_high_risk",
            "y_operational_risk",
            "final_time_to_tca",
            "total_cdm_count",
            "pre_cutoff_cdm_count",
        ]
    )
    y_high = df_merged["y_high_risk"]
    y_risk = df_merged["y_risk"]
    latest_risk = df_merged["latest_log_risk"]

    cutoff_days = float(cfg["data"].get("cutoff_days", 2.0))
    train_cutoffs = [float(c) for c in cfg["data"].get("train_cutoffs", [cutoff_days])]
    extra_cutoffs = [c for c in train_cutoffs if abs(c - cutoff_days) > 1e-6]

    if extra_cutoffs:
        print(f"Building multi-cutoff augmentation features for cutoffs={extra_cutoffs}...")
        raw_train = load_raw_cdms("data/raw", "train")
        train_eids = set(df_merged["event_id"].values)
        raw_train = raw_train[raw_train["event_id"].isin(train_eids)]
        op_thresh = float(cfg["thresholds"].get("operational_risk_log10", -4.0))
        aug_X_list, aug_y_list = [], []
        for c in extra_cutoffs:
            vis_c, tgt_c, _ = apply_cutoff_and_targets(raw_train, c, high_thresh, op_thresh)
            feat_c = build_all_features(vis_c, include_physics=True)
            m_c = feat_c.merge(tgt_c[["event_id", "y_high_risk"]], on="event_id")
            aug_X_list.append(m_c.drop(columns=["y_high_risk"]))
            aug_y_list.append(m_c["y_high_risk"])
        X_aug = pd.concat(aug_X_list, ignore_index=True)
        y_aug = pd.concat(aug_y_list, ignore_index=True)
        print(f"Augmented training pool: +{len(X_aug)} rows across {extra_cutoffs}")

        print("Fitting Unweighted Classifiers (Multi-Cutoff Augmented)...")
        clfs_unweighted = fit_predict_classifiers_augmented(
            X, y_high, df_merged, X_aug, y_aug, cfg, imbalance_strategy="unweighted"
        )
        print("Fitting Weighted Classifiers (Multi-Cutoff Augmented)...")
        clfs_weighted = fit_predict_classifiers_augmented(
            X, y_high, df_merged, X_aug, y_aug, cfg, imbalance_strategy="weighted"
        )
    else:
        print("Fitting Unweighted Classifiers...")
        clfs_unweighted = fit_predict_classifiers(
            X, y_high, df_merged, cfg, imbalance_strategy="unweighted"
        )

        print("Fitting Weighted Classifiers...")
        clfs_weighted = fit_predict_classifiers(
            X, y_high, df_merged, cfg, imbalance_strategy="weighted"
        )

    print("Fitting Regressors (Direct & Residual)...")
    regs = fit_predict_regressors(X, y_risk, latest_risk, df_merged, cfg)

    # Build OOF DataFrame
    oof_df = pd.DataFrame({
        "event_id": df_merged["event_id"],
        "fold": df_merged["fold"],
        "y_high_risk": y_high,
        "y_risk": y_risk,
        "latest_log_risk": latest_risk,
    })
    for name, data in clfs_unweighted.items():
        oof_df[f"{name}_unweighted_prob"] = data["oof_scores"].values
    for name, data in clfs_weighted.items():
        oof_df[f"{name}_weighted_prob"] = data["oof_scores"].values
    for name, data in regs.items():
        oof_df[f"{name}_pred_risk"] = data["oof_pred_risk"].values

    save_parquet(oof_df, "data/processed/oof_predictions.parquet")
    print(f"Saved oof_predictions.parquet successfully with {len(oof_df)} rows.")

    # Evaluation on filtered events
    eval_mask = apply_eval_filter(df_merged, cfg)
    filtered_targets = df_merged[eval_mask]
    filtered_oof = oof_df[eval_mask]

    n_pos = int(filtered_targets["y_high_risk"].sum())
    n_tot = len(filtered_targets)
    print(f"\n--- MODEL COMPARISON (Filtered Events, N = {n_tot}) ---")
    print(f"Positive events: {n_pos} / {n_tot} ({n_pos / n_tot:.2%})\n")

    for name in ["logreg", "rf", "lgbm"]:
        for strat in ["unweighted", "weighted"]:
            col = f"{name}_{strat}_prob"
            prob = filtered_oof[col].values
            # Default decision prob from config (0.35)
            pred_bin = (prob >= 0.35).astype(int)
            m = compute_classification_metrics(
                filtered_targets["y_high_risk"], pred_bin, prob
            )
            print(
                f"{name.upper():6s} ({strat:10s}): Recall={m['recall']:6.2%}, "
                f"Precision={m['precision']:6.2%}, F2={m['f2']:6.4f}, "
                f"PR-AUC={m['pr_auc']:6.4f}, FN={m['fn']:2d}, FP={m['fp']:3d}"
            )

    print("\n--- REGRESSOR COMPARISON (High-Risk MSE) ---")
    for reg_name in ["lgbm_direct", "lgbm_residual", "lgbm_conditional"]:
        col = f"{reg_name}_pred_risk"
        reg_m = compute_regression_metrics(
            filtered_targets["y_risk"], filtered_oof[col].values, high_thresh
        )
        print(
            f"{reg_name:14s}: High-Risk MSE = {reg_m['mse_high_risk']:6.4f}, "
            f"Overall MSE = {reg_m['mse_overall']:6.4f}"
        )


if __name__ == "__main__":
    run_phase_06()
