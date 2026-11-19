"""Phase 0 tests: config loading, parquet loaders, and dashboard mock files."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import yaml

from src.load import load_config, load_parquet
from src.recommend import CDM_HISTORY_COLUMNS, PREDICTIONS_COLUMNS


def test_config_loads() -> None:
    """Verify config.yaml exists and contains required top-level sections."""
    cfg_path = Path("config.yaml")
    assert cfg_path.exists(), "config.yaml must exist at repo root"
    cfg = load_config(cfg_path)
    for key in ("project", "data", "eval_filter", "thresholds", "validation"):
        assert key in cfg, f"Missing '{key}' in config.yaml"
    assert cfg["data"]["cutoff_days"] == 2.0


def test_mock_dashboard_files_exist_and_match_schema() -> None:
    """Verify mock parquet files exist and match Section 6 column schemas."""
    pred_path = Path("data/processed/predictions.parquet")
    hist_path = Path("data/processed/cdm_history.parquet")

    assert pred_path.exists(), "predictions.parquet must exist"
    assert hist_path.exists(), "cdm_history.parquet must exist"

    pred_df = load_parquet(pred_path)
    hist_df = load_parquet(hist_path)

    # Assert exact column schemas
    assert list(pred_df.columns) == PREDICTIONS_COLUMNS
    assert list(hist_df.columns) == CDM_HISTORY_COLUMNS

    # Assert row counts and content
    assert len(pred_df) > 0
    assert len(hist_df) > 0
    assert set(pred_df["split"].unique()).issubset({"train", "val", "test"})
    assert set(pred_df["recommendation"].unique()).issubset({"MANOEUVRE", "NO_MANOEUVRE"})
    assert set(pred_df["confidence"].unique()).issubset({"HIGH", "MEDIUM", "LOW"})
    assert pred_df["high_risk_prob"].between(0.0, 1.0).all()
