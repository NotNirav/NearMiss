"""Baseline and ML models (classifiers and regressors) with grouped CV.

Rules:
- Section 12 Rule 2 (Group isolation): use fold assignments from split.py.
- Section 12 Rule 3 (Test set quarantine): never tune on test set; return OOF
  predictions for training/validation folds.
- Section 12 Rule 4 (No magic numbers): read random_seed and thresholds from config.
- Section 12 Rule 5 (Fit on training data only): fit scalers/imputers inside each fold.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def predict_baseline_a(
    features_df: pd.DataFrame,
    high_risk_threshold: float,
) -> pd.DataFrame:
    """Baseline A (persistence): final risk = latest_log_risk.

    Inputs:
        features_df: Event-level features containing 'latest_log_risk', or
            visible CDMs containing 'event_id', 'time_to_tca', and 'risk'.
        high_risk_threshold: Threshold for MANOEUVRE recommendation (e.g. -6.0).
    Outputs:
        DataFrame with columns:
            pred_final_log_risk, baseline_recommendation ('MANOEUVRE' / 'NO_MANOEUVRE').
    Rules:
        Section 12 Rule 4.
    """
    df = features_df.copy()
    if "latest_log_risk" in df.columns:
        pred_risk = df["latest_log_risk"].values
        if "event_id" in df.columns:
            res = pd.DataFrame({"event_id": df["event_id"]})
        else:
            res = pd.DataFrame(index=df.index)
    elif "time_to_tca" in df.columns and ("risk" in df.columns or "true_risk" in df.columns):
        risk_col = "risk" if "risk" in df.columns else "true_risk"
        sorted_df = df.sort_values(["event_id", "time_to_tca"], ascending=[True, False])
        last_vis = sorted_df.groupby("event_id", as_index=False).last()
        res = pd.DataFrame({"event_id": last_vis["event_id"]})
        pred_risk = last_vis[risk_col].values
    else:
        raise ValueError(
            "features_df must contain 'latest_log_risk' or visible CDMs with 'time_to_tca' and 'risk'"
        )

    res["pred_final_log_risk"] = pred_risk
    res["baseline_recommendation"] = np.where(
        pred_risk >= high_risk_threshold, "MANOEUVRE", "NO_MANOEUVRE"
    )
    return res


def compute_residual_predictions(
    latest_log_risk: pd.Series | np.ndarray,
    pred_delta: pd.Series | np.ndarray,
) -> np.ndarray:
    """Reconstruct final log10 risk as latest_log_risk + pred_delta."""
    return np.asarray(latest_log_risk, dtype=float) + np.asarray(pred_delta, dtype=float)


def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    exclude = {
        "event_id",
        "split",
        "fold",
        "y_high_risk",
        "y_risk",
        "y_operational_risk",
        "final_time_to_tca",
        "total_cdm_count",
        "pre_cutoff_cdm_count",
    }
    return [c for c in df.columns if c not in exclude and np.issubdtype(df[c].dtype, np.number)]


def fit_predict_classifiers(
    X: pd.DataFrame,
    y: pd.Series,
    folds_df: pd.DataFrame,
    cfg: dict[str, Any],
    imbalance_strategy: str = "unweighted",
) -> dict[str, dict[str, Any]]:
    """Train LogisticRegression, RandomForest, and LightGBM classifiers via grouped CV.

    Inputs:
        X: Event-level feature matrix aligned with y and folds_df (train split only).
        y: Binary target y_high_risk.
        folds_df: Fold assignments from split.py.
        cfg: Loaded config dictionary.
        imbalance_strategy: 'unweighted' or 'weighted' (class_weight / scale_pos_weight).
    Outputs:
        Dict keyed by model name ('logreg', 'rf', 'lgbm'), each containing:
            'oof_scores': pd.Series of OOF positive-class probabilities/scores,
            'models': list of fitted fold estimators/pipelines.
    Rules:
        Section 12 Rule 2, Rule 3, Rule 4, Rule 5.
    """
    random_seed = int(cfg.get("project", {}).get("random_seed", 42))
    val_cfg = cfg.get("validation", {})
    n_splits = int(val_cfg.get("n_splits", 5))

    feat_cols = _get_feature_cols(X)
    X_mat = X[feat_cols].copy()
    y_vec = np.asarray(y, dtype=int)

    train_mask = folds_df["split"] == "train" if "split" in folds_df.columns else np.ones(len(folds_df), dtype=bool)
    train_folds = folds_df[train_mask].copy()

    models_dict = {
        "logreg": {"oof_scores": np.zeros(len(train_folds)), "models": []},
        "rf": {"oof_scores": np.zeros(len(train_folds)), "models": []},
        "lgbm": {"oof_scores": np.zeros(len(train_folds)), "models": []},
    }

    # Extract fold indices
    for fold_i in range(n_splits):
        val_idx = np.where(train_folds["fold"].values == fold_i)[0]
        trn_idx = np.where(train_folds["fold"].values != fold_i)[0]

        X_train, y_train = X_mat.iloc[trn_idx], y_vec[trn_idx]
        X_val, y_val = X_mat.iloc[val_idx], y_vec[val_idx]

        pos_count = int(np.sum(y_train == 1))
        neg_count = int(np.sum(y_train == 0))
        pos_weight = float(neg_count / max(1, pos_count))

        # 1. Logistic Regression
        lr_class_weight = "balanced" if imbalance_strategy == "weighted" else None
        lr_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight=lr_class_weight,
                    random_state=random_seed + fold_i,
                    max_iter=1000,
                    C=0.1,
                ),
            ),
        ])
        lr_pipeline.fit(X_train, y_train)
        models_dict["logreg"]["models"].append(lr_pipeline)
        lr_prob = lr_pipeline.predict_proba(X_val)[:, 1]
        models_dict["logreg"]["oof_scores"][val_idx] = lr_prob

        # 2. Random Forest
        rf_class_weight = (
            "balanced_subsample" if imbalance_strategy == "weighted" else None
        )
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            class_weight=rf_class_weight,
            random_state=random_seed + fold_i,
            n_jobs=-1,
        )
        rf_model.fit(X_train, y_train)
        models_dict["rf"]["models"].append(rf_model)
        rf_prob = rf_model.predict_proba(X_val)[:, 1]
        models_dict["rf"]["oof_scores"][val_idx] = rf_prob

        # 3. LightGBM Classifier
        lgb_spw = pos_weight if imbalance_strategy == "weighted" else 1.0
        lgb_model = LGBMClassifier(
            n_estimators=120,
            learning_rate=0.05,
            max_depth=5,
            scale_pos_weight=lgb_spw,
            random_state=random_seed + fold_i,
            n_jobs=-1,
            verbose=-1,
        )
        lgb_model.fit(X_train, y_train)
        models_dict["lgbm"]["models"].append(lgb_model)
        lgb_prob = lgb_model.predict_proba(X_val)[:, 1]
        models_dict["lgbm"]["oof_scores"][val_idx] = lgb_prob

    result = {}
    for name, data in models_dict.items():
        result[name] = {
            "oof_scores": pd.Series(
                data["oof_scores"], index=train_folds.index, name=f"{name}_oof"
            ),
            "models": data["models"],
        }
    return result


def fit_predict_regressors(
    X: pd.DataFrame,
    y_risk: pd.Series,
    latest_log_risk: pd.Series,
    folds_df: pd.DataFrame,
    cfg: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Train direct LightGBM regressor and residual LightGBM regressor via grouped CV.

    Inputs:
        X: Event-level feature matrix (train split only).
        y_risk: Continuous target final log10 risk.
        latest_log_risk: Latest visible log10 risk per event.
        folds_df: Fold assignments from split.py.
        cfg: Loaded config dictionary.
    Outputs:
        Dict keyed by 'lgbm_direct' and 'lgbm_residual', each containing:
            'oof_pred_risk': pd.Series of OOF predicted final log10 risk,
            'models': list of fitted fold regressors.
    Rules:
        Section 12 Rule 2, Rule 3, Rule 4, Rule 5.
    """
    random_seed = int(cfg.get("project", {}).get("random_seed", 42))
    val_cfg = cfg.get("validation", {})
    n_splits = int(val_cfg.get("n_splits", 5))

    feat_cols = _get_feature_cols(X)
    X_mat = X[feat_cols].copy()
    y_val_vec = np.asarray(y_risk, dtype=float)
    latest_vec = np.asarray(latest_log_risk, dtype=float)
    delta_vec = y_val_vec - latest_vec

    train_mask = folds_df["split"] == "train" if "split" in folds_df.columns else np.ones(len(folds_df), dtype=bool)
    train_folds = folds_df[train_mask].copy()

    models_dict = {
        "lgbm_direct": {"oof_pred_risk": np.zeros(len(train_folds)), "models": []},
        "lgbm_residual": {"oof_pred_risk": np.zeros(len(train_folds)), "models": []},
        "lgbm_conditional": {"oof_pred_risk": np.zeros(len(train_folds)), "models": []},
    }

    high_thresh = float(cfg.get("thresholds", {}).get("high_risk_log10", -6.0))
    floor_val = -30.0

    for fold_i in range(n_splits):
        val_idx = np.where(train_folds["fold"].values == fold_i)[0]
        trn_idx = np.where(train_folds["fold"].values != fold_i)[0]

        X_train = X_mat.iloc[trn_idx]
        X_val = X_mat.iloc[val_idx]
        y_trn = y_val_vec[trn_idx]

        # 1. Direct Regressor
        direct_model = LGBMRegressor(
            n_estimators=120,
            learning_rate=0.05,
            max_depth=5,
            random_state=random_seed + fold_i,
            n_jobs=-1,
            verbose=-1,
        )
        direct_model.fit(X_train, y_trn)
        models_dict["lgbm_direct"]["models"].append(direct_model)
        models_dict["lgbm_direct"]["oof_pred_risk"][val_idx] = direct_model.predict(
            X_val
        )

        # 2. Residual Regressor (predicts delta = y_risk - latest_log_risk)
        residual_model = LGBMRegressor(
            n_estimators=120,
            learning_rate=0.05,
            max_depth=5,
            random_state=random_seed + fold_i,
            n_jobs=-1,
            verbose=-1,
        )
        residual_model.fit(X_train, delta_vec[trn_idx])
        models_dict["lgbm_residual"]["models"].append(residual_model)
        pred_delta = residual_model.predict(X_val)
        models_dict["lgbm_residual"]["oof_pred_risk"][val_idx] = (
            compute_residual_predictions(latest_vec[val_idx], pred_delta)
        )

        # 3. Two-Stage Conditional (Hurdle) Regressor: separates -30.0 floor from active risk
        nonfloor_mask = y_trn > (floor_val + 1e-6)
        nonfloor_y = nonfloor_mask.astype(int)
        if len(np.unique(nonfloor_y)) >= 2:
            stage1_clf = LGBMClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=5,
                random_state=random_seed + fold_i,
                n_jobs=-1,
                verbose=-1,
            )
            stage1_clf.fit(X_train, nonfloor_y)
            p_nonfloor = stage1_clf.predict_proba(X_val)[:, 1]
        else:
            stage1_clf = None
            p_nonfloor = np.full(len(X_val), float(nonfloor_y[0]) if len(nonfloor_y) > 0 else 1.0)

        X_nf = X_train[nonfloor_mask]
        y_nf = y_trn[nonfloor_mask]
        if len(X_nf) == 0:
            X_nf = X_train
            y_nf = y_trn
        # Emphasize high-risk tail inside non-floor regression
        sw_nf = np.where(y_nf >= high_thresh, 5.0, 1.0)
        stage2_reg = LGBMRegressor(
            n_estimators=120,
            learning_rate=0.05,
            max_depth=5,
            random_state=random_seed + fold_i,
            n_jobs=-1,
            verbose=-1,
        )
        stage2_reg.fit(X_nf, y_nf, sample_weight=sw_nf)
        y_nf_pred = np.clip(stage2_reg.predict(X_val), floor_val, 0.0)
        models_dict["lgbm_conditional"]["models"].append((stage1_clf, stage2_reg))
        models_dict["lgbm_conditional"]["oof_pred_risk"][val_idx] = (
            p_nonfloor * y_nf_pred + (1.0 - p_nonfloor) * floor_val
        )

    result = {}
    for name, data in models_dict.items():
        result[name] = {
            "oof_pred_risk": pd.Series(
                data["oof_pred_risk"], index=train_folds.index, name=f"{name}_oof"
            ),
            "models": data["models"],
        }
    return result


def fit_predict_classifiers_augmented(
    X_base: pd.DataFrame,
    y_base: pd.Series,
    folds_df: pd.DataFrame,
    X_aug: pd.DataFrame,
    y_aug: pd.Series,
    cfg: dict[str, Any],
    imbalance_strategy: str = "weighted",
) -> dict[str, dict[str, Any]]:
    """Train classifiers with multi-cutoff augmentation strictly inside each training fold.

    Validation predictions are evaluated exclusively on X_base (the primary 2.0-day cutoff).
    Augmented rows from X_aug (e.g. 3.0 and 4.0 day cutoffs) are added to fold training
    data ONLY for event_ids belonging to that fold's training split (Section 12 Rule 2).
    """
    random_seed = int(cfg.get("project", {}).get("random_seed", 42))
    n_splits = int(cfg.get("validation", {}).get("n_splits", 5))

    feat_cols = _get_feature_cols(X_base)
    X_base_mat = X_base[feat_cols].copy()
    y_base_vec = np.asarray(y_base, dtype=int)

    X_aug_mat = X_aug[feat_cols].copy()
    y_aug_vec = np.asarray(y_aug, dtype=int)
    aug_eids = X_aug["event_id"].values

    train_mask = folds_df["split"] == "train" if "split" in folds_df.columns else np.ones(len(folds_df), dtype=bool)
    train_folds = folds_df[train_mask].copy()
    base_eids = train_folds["event_id"].values

    models_dict = {
        "logreg": {"oof_scores": np.zeros(len(train_folds)), "models": []},
        "rf": {"oof_scores": np.zeros(len(train_folds)), "models": []},
        "lgbm": {"oof_scores": np.zeros(len(train_folds)), "models": []},
    }

    for fold_i in range(n_splits):
        val_idx = np.where(train_folds["fold"].values == fold_i)[0]
        trn_idx = np.where(train_folds["fold"].values != fold_i)[0]
        trn_eid_set = set(base_eids[trn_idx])

        aug_trn_mask = np.isin(aug_eids, list(trn_eid_set))
        X_train = pd.concat([X_base_mat.iloc[trn_idx], X_aug_mat[aug_trn_mask]], axis=0, ignore_index=True)
        y_train = np.concatenate([y_base_vec[trn_idx], y_aug_vec[aug_trn_mask]])
        X_val = X_base_mat.iloc[val_idx]

        pos_count = int(np.sum(y_train == 1))
        neg_count = int(np.sum(y_train == 0))
        pos_weight = float(neg_count / max(1, pos_count))

        lr_class_weight = "balanced" if imbalance_strategy == "weighted" else None
        lr_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight=lr_class_weight, random_state=random_seed + fold_i, max_iter=1000, C=0.1)),
        ])
        lr_pipeline.fit(X_train, y_train)
        models_dict["logreg"]["models"].append(lr_pipeline)
        models_dict["logreg"]["oof_scores"][val_idx] = lr_pipeline.predict_proba(X_val)[:, 1]

        rf_class_weight = "balanced_subsample" if imbalance_strategy == "weighted" else None
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            class_weight=rf_class_weight,
            random_state=random_seed + fold_i,
            n_jobs=-1,
        )
        rf_model.fit(X_train, y_train)
        models_dict["rf"]["models"].append(rf_model)
        models_dict["rf"]["oof_scores"][val_idx] = rf_model.predict_proba(X_val)[:, 1]

        lgb_spw = pos_weight if imbalance_strategy == "weighted" else 1.0
        lgb_model = LGBMClassifier(
            n_estimators=120,
            learning_rate=0.05,
            max_depth=5,
            scale_pos_weight=lgb_spw,
            random_state=random_seed + fold_i,
            n_jobs=-1,
            verbose=-1,
        )
        lgb_model.fit(X_train, y_train)
        models_dict["lgbm"]["models"].append(lgb_model)
        models_dict["lgbm"]["oof_scores"][val_idx] = lgb_model.predict_proba(X_val)[:, 1]

    result = {}
    for name, data in models_dict.items():
        result[name] = {
            "oof_scores": pd.Series(
                data["oof_scores"], index=train_folds.index, name=f"{name}_aug_oof"
            ),
            "models": data["models"],
        }
    return result
