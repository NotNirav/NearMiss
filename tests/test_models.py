"""Tests for baseline, classification, and regression models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.load import load_config
from src.models import (
    compute_residual_predictions,
    fit_predict_classifiers,
    fit_predict_classifiers_augmented,
    fit_predict_regressors,
    predict_baseline_a,
)


@pytest.fixture
def dummy_train_setup() -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    rng = np.random.default_rng(42)
    n = 60
    eids = np.arange(100, 100 + n)
    X = pd.DataFrame({
        "event_id": eids,
        "feat_1": rng.normal(0, 1, size=n),
        "feat_2": rng.uniform(0, 10, size=n),
        "latest_log_risk": rng.uniform(-25.0, -5.0, size=n),
    })
    y_high = pd.Series((rng.uniform(0, 1, size=n) < 0.2).astype(int), name="y_high_risk")
    y_high.iloc[0] = 1
    y_high.iloc[1] = 1

    y_risk = pd.Series(X["latest_log_risk"] + rng.normal(0, 1, size=n), name="y_risk")

    folds_df = pd.DataFrame({
        "event_id": eids,
        "split": ["train"] * n,
        "fold": [i % 5 for i in range(n)],
        "y_high_risk": y_high,
    })
    return X, y_high, y_risk, folds_df


def test_residual_prediction_formula() -> None:
    latest = np.array([-10.0, -5.0, -20.0])
    delta = np.array([2.0, -1.0, 0.5])
    reconstructed = compute_residual_predictions(latest, delta)
    np.testing.assert_allclose(reconstructed, [-8.0, -6.0, -19.5])


def test_classifiers_and_regressors_cv(
    dummy_train_setup: tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame],
) -> None:
    X, y_high, y_risk, folds_df = dummy_train_setup
    cfg = load_config()

    # 1. Classifiers
    clf_res = fit_predict_classifiers(X, y_high, folds_df, cfg, imbalance_strategy="unweighted")
    for name in ("logreg", "rf", "lgbm"):
        assert name in clf_res
        oof = clf_res[name]["oof_scores"]
        assert len(oof) == len(X)
        assert (oof >= 0.0).all() and (oof <= 1.0).all()
        assert len(clf_res[name]["models"]) == 5

    # 2. Regressors (direct, residual, and conditional)
    reg_res = fit_predict_regressors(X, y_risk, X["latest_log_risk"], folds_df, cfg)
    for name in ("lgbm_direct", "lgbm_residual", "lgbm_conditional"):
        assert name in reg_res
        oof_r = reg_res[name]["oof_pred_risk"]
        assert len(oof_r) == len(X)
        assert not np.isnan(oof_r).any()
        assert len(reg_res[name]["models"]) == 5


def test_augmented_classifiers_cv_and_group_isolation(
    dummy_train_setup: tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame],
) -> None:
    X, y_high, _, folds_df = dummy_train_setup
    cfg = load_config()

    # Augmented rows from earlier cutoffs sharing the same event_ids
    X_aug = X.copy()
    X_aug["feat_1"] = X_aug["feat_1"] + 0.1
    y_aug = y_high.copy()

    res = fit_predict_classifiers_augmented(
        X, y_high, folds_df, X_aug, y_aug, cfg, imbalance_strategy="weighted"
    )
    for name in ("logreg", "rf", "lgbm"):
        assert name in res
        oof = res[name]["oof_scores"]
        assert len(oof) == len(X)
        assert (oof >= 0.0).all() and (oof <= 1.0).all()
        assert len(res[name]["models"]) == 5
