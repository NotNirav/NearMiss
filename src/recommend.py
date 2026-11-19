"""Export dashboard-ready parquet files (predictions.parquet and cdm_history.parquet).

Provides both generate_mock_dashboard_files() (Phase 0) and the production
export_production_dashboard_files() (Phase 8).

Rules:
- Section 6: exact output schema for predictions.parquet and cdm_history.parquet.
- Section 12 Rule 4 (No magic numbers): read paths and thresholds from config.yaml.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.calibrate import apply_platt_scaler, assign_confidence, fit_platt_scaler, tune_decision_threshold
from src.cutoff import apply_cutoff_and_targets, apply_eval_filter
from src.features import build_all_features
from src.load import load_config, load_parquet, load_raw_cdms, save_parquet

PREDICTIONS_COLUMNS = [
    "event_id",
    "mission_id",
    "object_type",
    "cdm_count_pre_cutoff",
    "latest_log_risk",
    "pred_final_log_risk",
    "high_risk_prob",
    "recommendation",
    "confidence",
    "baseline_recommendation",
    "actual_final_log_risk",
    "actual_high_risk",
    "split",
]

CDM_HISTORY_COLUMNS = [
    "event_id",
    "time_to_tca",
    "log_risk",
    "miss_distance",
    "visible",
]


def build_predictions_table(
    events_df: pd.DataFrame,
    cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Format event-level predictions into the Section 6 predictions.parquet schema.

    Required columns:
        event_id, mission_id, object_type, cdm_count_pre_cutoff,
        latest_log_risk, pred_final_log_risk, high_risk_prob,
        recommendation, confidence, baseline_recommendation,
        actual_final_log_risk, actual_high_risk, split.
    """
    for col in PREDICTIONS_COLUMNS:
        if col not in events_df.columns:
            raise ValueError(f"Missing required predictions column: '{col}'")
    return events_df[PREDICTIONS_COLUMNS].copy()


def build_cdm_history_table(
    cdms_df: pd.DataFrame,
    cutoff_days: float = 2.0,
) -> pd.DataFrame:
    """Format CDM history into the Section 6 cdm_history.parquet schema.

    Required columns:
        event_id, time_to_tca, log_risk, miss_distance, visible.
    """
    df = cdms_df.copy()
    risk_col = "risk" if "risk" in df.columns else ("true_risk" if "true_risk" in df.columns else "log_risk")
    if "log_risk" not in df.columns and risk_col in df.columns:
        df["log_risk"] = df[risk_col]

    if "visible" not in df.columns:
        df["visible"] = df["time_to_tca"] >= cutoff_days

    for col in CDM_HISTORY_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing required cdm_history column: '{col}'")
    return df[CDM_HISTORY_COLUMNS].copy()


def export_dashboard_files(
    predictions_df: pd.DataFrame,
    cdm_history_df: pd.DataFrame,
    processed_dir: str | Path,
) -> tuple[Path, Path]:
    """Validate schemas and write predictions.parquet and cdm_history.parquet."""
    pred_clean = build_predictions_table(predictions_df)
    hist_clean = build_cdm_history_table(cdm_history_df)

    out_dir = Path(processed_dir)
    pred_path = out_dir / "predictions.parquet"
    hist_path = out_dir / "cdm_history.parquet"

    save_parquet(pred_clean, pred_path)
    save_parquet(hist_clean, hist_path)

    return pred_path.resolve(), hist_path.resolve()


def generate_mock_dashboard_files(
    processed_dir: str | Path = "data/processed",
    seed: int | None = None,
    n_events: int = 120,
) -> tuple[Path, Path]:
    """Generate realistic mock predictions.parquet and cdm_history.parquet for Phase 0."""
    cfg = load_config()
    if seed is None:
        seed = int(cfg.get("project", {}).get("random_seed", 42))

    rng = np.random.default_rng(seed)
    cutoff_days = float(cfg.get("data", {}).get("cutoff_days", 2.0))
    thresholds = cfg.get("thresholds", {})
    high_risk_thresh = float(thresholds.get("high_risk_log10", -6.0))
    decision_prob = float(thresholds.get("decision_prob", 0.35))
    margin_high = float(thresholds.get("confidence_margin_high", 0.25))
    margin_med = float(thresholds.get("confidence_margin_medium", 0.10))

    object_types = ["DEBRIS", "PAYLOAD", "ROCKET BODY", "UNKNOWN"]
    splits = ["val", "test"]

    events_list = []
    cdms_list = []

    for i in range(n_events):
        event_id = 10000 + i
        mission_id = int(rng.choice([1, 2, 3, 4]))
        obj_type = str(rng.choice(object_types))
        split = splits[i % len(splits)]

        archetype = rng.choice([0, 1, 2, 3], p=[0.70, 0.10, 0.10, 0.10])
        n_cdms = int(rng.integers(3, 14))
        times = np.sort(rng.uniform(0.15, 6.8, size=n_cdms))[::-1]
        if times[0] < cutoff_days:
            times[0] = float(rng.uniform(cutoff_days + 0.1, 6.5))
            times = np.sort(times)[::-1]

        risks = []
        miss_dists = []
        base_dist = float(rng.uniform(100.0, 25000.0))
        for t in times:
            if archetype == 0:
                r = float(rng.uniform(-30.0, -12.0))
                d = base_dist + rng.normal(0, 500)
            elif archetype == 1:
                r = float(rng.uniform(-5.5, -2.5))
                d = float(rng.uniform(30.0, 400.0)) + rng.normal(0, 30)
            elif archetype == 2:
                if t >= cutoff_days:
                    r = float(rng.uniform(-25.0, -8.0))
                    d = float(rng.uniform(800.0, 5000.0))
                else:
                    r = float(rng.uniform(-5.8, -2.8))
                    d = float(rng.uniform(25.0, 250.0))
            else:
                if t >= cutoff_days:
                    r = float(rng.uniform(-5.5, -3.5))
                    d = float(rng.uniform(50.0, 350.0))
                else:
                    r = float(rng.uniform(-18.0, -9.0))
                    d = float(rng.uniform(1500.0, 8000.0))

            risks.append(r)
            miss_dists.append(max(5.0, d))

        for t, r, d in zip(times, risks, miss_dists):
            cdms_list.append({
                "event_id": event_id,
                "time_to_tca": round(float(t), 4),
                "log_risk": round(float(r), 4),
                "miss_distance": round(float(d), 2),
                "visible": bool(t >= cutoff_days),
            })

        visible_indices = [idx for idx, t in enumerate(times) if t >= cutoff_days]
        cdm_count_pre = len(visible_indices)
        latest_vis_idx = visible_indices[-1]
        latest_risk = risks[latest_vis_idx]

        final_idx = len(times) - 1
        actual_final_risk = risks[final_idx]
        actual_high_risk = int(actual_final_risk >= high_risk_thresh)

        baseline_rec = "MANOEUVRE" if latest_risk >= high_risk_thresh else "NO_MANOEUVRE"

        if archetype in (1, 2):
            high_risk_prob = float(np.clip(rng.uniform(0.60, 0.96), 0.0, 1.0))
            pred_final_risk = float(rng.uniform(-5.5, -2.5))
        elif archetype == 3:
            high_risk_prob = float(np.clip(rng.uniform(0.10, 0.45), 0.0, 1.0))
            pred_final_risk = float(rng.uniform(-12.0, -7.0))
        else:
            high_risk_prob = float(np.clip(rng.uniform(0.01, 0.15), 0.0, 1.0))
            pred_final_risk = float(rng.uniform(-28.0, -10.0))

        rec = "MANOEUVRE" if high_risk_prob >= decision_prob else "NO_MANOEUVRE"
        margin = abs(high_risk_prob - decision_prob)
        conf = "HIGH" if margin >= margin_high else ("MEDIUM" if margin >= margin_med else "LOW")

        events_list.append({
            "event_id": event_id,
            "mission_id": mission_id,
            "object_type": obj_type,
            "cdm_count_pre_cutoff": cdm_count_pre,
            "latest_log_risk": round(latest_risk, 4),
            "pred_final_log_risk": round(pred_final_risk, 4),
            "high_risk_prob": round(high_risk_prob, 4),
            "recommendation": rec,
            "confidence": conf,
            "baseline_recommendation": baseline_rec,
            "actual_final_log_risk": round(actual_final_risk, 4),
            "actual_high_risk": actual_high_risk,
            "split": split,
        })

    predictions_df = pd.DataFrame(events_list)
    cdm_history_df = pd.DataFrame(cdms_list)

    return export_dashboard_files(predictions_df, cdm_history_df, processed_dir)


def export_production_dashboard_files(
    processed_dir: str | Path = "data/processed",
) -> tuple[Path, Path]:
    """Build and export production predictions.parquet and cdm_history.parquet.

    Evaluates across train, validation (fold 0), and the locked test set.
    """
    cfg = load_config()
    high_thresh = float(cfg["thresholds"]["high_risk_log10"])

    calibrated_oof = load_parquet("data/processed/calibrated_oof.parquet")
    features_train = load_parquet("data/processed/features_train.parquet")
    targets_train = load_parquet("data/processed/targets_train.parquet")
    raw_train = load_raw_cdms("data/raw", "train")

    # Tune decision threshold dynamically on filtered OOF calibrated probabilities
    cal_eval = calibrated_oof.merge(
        targets_train[["event_id", "final_time_to_tca", "pre_cutoff_cdm_count"]],
        on="event_id",
    )
    eval_mask = apply_eval_filter(cal_eval, cfg)
    decision_th, _ = tune_decision_threshold(
        cal_eval.loc[eval_mask, "y_high_risk"].values,
        cal_eval.loc[eval_mask, "calibrated_prob"].values,
        beta=2.0,
    )

    # Map object_type and mission_id from latest CDMs in raw_train
    raw_sorted = raw_train.sort_values(["event_id", "time_to_tca"], ascending=[True, False])
    vis_raw = raw_sorted[raw_sorted["time_to_tca"] >= float(cfg["data"]["cutoff_days"])]
    latest_cdms_train = vis_raw.groupby("event_id").last().reset_index()

    train_meta = latest_cdms_train[["event_id", "mission_id", "c_object_type"]].rename(
        columns={"c_object_type": "object_type"}
    )

    # 1. Prepare Train & Validation events
    train_val_df = calibrated_oof.merge(train_meta, on="event_id", how="left")
    train_val_df["object_type"] = train_val_df["object_type"].fillna("UNKNOWN")
    train_val_df["mission_id"] = train_val_df["mission_id"].fillna(1).astype(int)

    # CDM count pre cutoff
    pre_counts = vis_raw.groupby("event_id").size().rename("cdm_count_pre_cutoff")
    train_val_df = train_val_df.merge(pre_counts, on="event_id", how="left")
    train_val_df["cdm_count_pre_cutoff"] = train_val_df["cdm_count_pre_cutoff"].fillna(1).astype(int)

    oof_preds = load_parquet("data/processed/oof_predictions.parquet")
    reg_col = "lgbm_conditional_pred_risk" if "lgbm_conditional_pred_risk" in oof_preds.columns else "lgbm_direct_pred_risk"
    train_val_df["pred_final_log_risk"] = oof_preds[reg_col].round(4)
    train_val_df["high_risk_prob"] = train_val_df["calibrated_prob"].round(4)
    train_val_df["recommendation"] = np.where(train_val_df["high_risk_prob"] >= decision_th, "MANOEUVRE", "NO_MANOEUVRE")
    train_val_df["confidence"] = assign_confidence(train_val_df["high_risk_prob"], decision_th, cfg)
    train_val_df["baseline_recommendation"] = np.where(
        train_val_df["latest_log_risk"] >= high_thresh, "MANOEUVRE", "NO_MANOEUVRE"
    )
    train_val_df["actual_final_log_risk"] = train_val_df["y_risk"].round(4)
    train_val_df["actual_high_risk"] = train_val_df["y_high_risk"].astype(int)
    # Fold 0 is validation; Folds 1..4 are train
    train_val_df["split"] = np.where(train_val_df["fold"] == 0, "val", "train")

    # 2. Score the locked Test Set
    test_cdms = load_raw_cdms("data/raw", "test")
    test_priv = load_raw_cdms("data/raw", "test_private")

    test_features = build_all_features(test_cdms, include_physics=True)

    # Fit champion Random Forest on full training set (with multi-cutoff augmentation if configured)
    from src.models import _get_feature_cols
    feat_cols = _get_feature_cols(features_train)
    X_train_full = features_train[feat_cols].copy()
    y_train_full = targets_train["y_high_risk"].values

    cutoff_days = float(cfg["data"].get("cutoff_days", 2.0))
    train_cutoffs = [float(c) for c in cfg["data"].get("train_cutoffs", [cutoff_days])]
    extra_cutoffs = [c for c in train_cutoffs if abs(c - cutoff_days) > 1e-6]
    if extra_cutoffs:
        op_thresh = float(cfg["thresholds"].get("operational_risk_log10", -4.0))
        train_eids = set(features_train["event_id"].values)
        raw_train_sub = raw_train[raw_train["event_id"].isin(train_eids)]
        aug_X_parts, aug_y_parts = [X_train_full], [y_train_full]
        for c in extra_cutoffs:
            vis_c, tgt_c, _ = apply_cutoff_and_targets(raw_train_sub, c, high_thresh, op_thresh)
            feat_c = build_all_features(vis_c, include_physics=True)
            m_c = feat_c.merge(tgt_c[["event_id", "y_high_risk"]], on="event_id")
            aug_X_parts.append(m_c[feat_cols])
            aug_y_parts.append(m_c["y_high_risk"].values)
        X_train_clf = pd.concat(aug_X_parts, axis=0, ignore_index=True)
        y_train_clf = np.concatenate(aug_y_parts)
    else:
        X_train_clf = X_train_full
        y_train_clf = y_train_full

    rf_prod = RandomForestClassifier(
        n_estimators=120,
        max_depth=8,
        class_weight="balanced_subsample",
        random_state=int(cfg["project"]["random_seed"]),
        n_jobs=-1,
    )
    rf_prod.fit(X_train_clf, y_train_clf)

    # Platt calibrator
    scaler_prod = fit_platt_scaler(calibrated_oof["raw_prob"].values, calibrated_oof["y_high_risk"].values)

    X_test = test_features[feat_cols].copy()
    raw_test_scores = rf_prod.predict_proba(X_test)[:, 1]
    cal_test_probs = apply_platt_scaler(scaler_prod, raw_test_scores)

    # Two-stage conditional regressor on test (separates -30.0 floor from active risk)
    from lightgbm import LGBMClassifier, LGBMRegressor
    y_risk_train = targets_train["y_risk"].values
    floor_val = -30.0
    nonfloor_mask = y_risk_train > (floor_val + 1e-6)
    stage1_clf = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        random_state=int(cfg["project"]["random_seed"]),
        n_jobs=-1,
        verbose=-1,
    )
    stage1_clf.fit(X_train_full, nonfloor_mask.astype(int))
    p_nonfloor_test = stage1_clf.predict_proba(X_test)[:, 1]

    X_nf = X_train_full[nonfloor_mask]
    y_nf = y_risk_train[nonfloor_mask]
    sw_nf = np.where(y_nf >= high_thresh, 5.0, 1.0)
    stage2_reg = LGBMRegressor(
        n_estimators=120,
        learning_rate=0.05,
        max_depth=5,
        random_state=int(cfg["project"]["random_seed"]),
        n_jobs=-1,
        verbose=-1,
    )
    stage2_reg.fit(X_nf, y_nf, sample_weight=sw_nf)
    y_nf_test = np.clip(stage2_reg.predict(X_test), floor_val, 0.0)
    pred_test_risk = p_nonfloor_test * y_nf_test + (1.0 - p_nonfloor_test) * floor_val

    # Test metadata
    test_last_cdm = test_cdms.sort_values(["event_id", "time_to_tca"], ascending=[True, False]).groupby("event_id").last().reset_index()
    test_pre_counts = test_cdms.groupby("event_id").size().rename("cdm_count_pre_cutoff")

    test_events_df = pd.DataFrame({"event_id": test_features["event_id"]})
    test_events_df = test_events_df.merge(
        test_last_cdm[["event_id", "mission_id", "c_object_type", "risk"]].rename(
            columns={"c_object_type": "object_type", "risk": "latest_log_risk"}
        ),
        on="event_id",
        how="left",
    )
    test_events_df = test_events_df.merge(test_pre_counts, on="event_id", how="left")
    test_events_df = test_events_df.merge(
        test_priv[["event_id", "risk"]].rename(columns={"risk": "actual_final_log_risk"}),
        on="event_id",
        how="left",
    )

    test_events_df["object_type"] = test_events_df["object_type"].fillna("UNKNOWN")
    test_events_df["mission_id"] = test_events_df["mission_id"].fillna(1).astype(int)
    test_events_df["cdm_count_pre_cutoff"] = test_events_df["cdm_count_pre_cutoff"].fillna(1).astype(int)
    test_events_df["pred_final_log_risk"] = np.round(pred_test_risk, 4)
    test_events_df["high_risk_prob"] = np.round(cal_test_probs, 4)
    test_events_df["recommendation"] = np.where(test_events_df["high_risk_prob"] >= decision_th, "MANOEUVRE", "NO_MANOEUVRE")
    test_events_df["confidence"] = assign_confidence(test_events_df["high_risk_prob"], decision_th, cfg)
    test_events_df["baseline_recommendation"] = np.where(
        test_events_df["latest_log_risk"] >= high_thresh, "MANOEUVRE", "NO_MANOEUVRE"
    )
    test_events_df["actual_final_log_risk"] = test_events_df["actual_final_log_risk"].round(4)
    test_events_df["actual_high_risk"] = (test_events_df["actual_final_log_risk"] >= high_thresh).astype(int)
    test_events_df["split"] = "test"

    # Combine all splits
    all_predictions = pd.concat([train_val_df[PREDICTIONS_COLUMNS], test_events_df[PREDICTIONS_COLUMNS]], ignore_index=True)

    # 3. CDM History
    # We export CDM history for validation (fold 0) and test events to keep history file compact (~25MB)
    focus_event_ids = set(all_predictions[all_predictions["split"].isin(["val", "test"])]["event_id"])

    train_focus_cdms = raw_train[raw_train["event_id"].isin(focus_event_ids)][
        ["event_id", "time_to_tca", "risk", "miss_distance"]
    ].rename(columns={"risk": "log_risk"})
    train_focus_cdms["visible"] = train_focus_cdms["time_to_tca"] >= float(cfg["data"]["cutoff_days"])

    test_merged = load_raw_cdms("data/raw", "test_merged")
    test_focus_cdms = test_merged[
        ["event_id", "time_to_tca", "risk", "miss_distance"]
    ].rename(columns={"risk": "log_risk"})
    test_focus_cdms["visible"] = test_focus_cdms["time_to_tca"] >= float(cfg["data"]["cutoff_days"])

    all_history = pd.concat([train_focus_cdms, test_focus_cdms], ignore_index=True)

    return export_dashboard_files(all_predictions, all_history, processed_dir)
