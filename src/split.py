"""Grouped and stratified dataset splitting by event_id.

Rules:
- Section 12 Rule 2 (Group isolation): never split rows randomly; split by
  event_id and assert zero overlap across splits and folds.
- Section 12 Rule 3 (Test set quarantine): lock test set until final evaluation.
- Section 12 Rule 4 (No magic numbers): read n_splits, holdout_fraction, and
  random_seed from config.yaml.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from src.load import load_config, load_parquet, save_parquet


def check_chronology(
    train_targets: pd.DataFrame,
    test_targets: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Inspect whether event_id or other columns encode chronology.

    Inputs:
        train_targets: Event-level target table for training data.
        test_targets: Optional official test event-level table.
    Outputs:
        Summary dictionary describing chronological findings.
    Rules:
        Section 12 Rule 8 (record findings in reports/assumptions.md).
    """
    train_ids = train_targets["event_id"].values
    monotonic_train = bool(np.all(np.diff(train_ids) >= 0))

    overlap = 0
    test_count = 0
    if test_targets is not None:
        test_ids = test_targets["event_id"].values
        test_count = len(test_ids)
        overlap = len(set(train_ids).intersection(set(test_ids)))

    return {
        "train_event_count": len(train_ids),
        "train_ids_monotonic": monotonic_train,
        "test_event_count": test_count,
        "raw_id_overlap": overlap,
        "chronology_finding": (
            "No absolute timestamps exist in the dataset; space weather reflects varying solar cycles. "
            "ESA paper states test set is chronologically subsequent to train set, but re-indexed from 0."
        ),
    }


def create_splits(
    train_targets: pd.DataFrame,
    cfg: dict[str, Any],
    test_targets: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Assign each event_id to split ('train' or 'test') and CV fold (0..k-1 or -1 for test).

    Inputs:
        train_targets: Event-level targets from training archive.
        cfg: Loaded config dictionary.
        test_targets: Optional official labelled test targets. If None, holds out
            cfg['validation']['holdout_fraction'] stratified by y_high_risk.
    Outputs:
        folds_df: DataFrame with columns ['event_id', 'split', 'fold', 'y_high_risk'].
    Rules:
        Section 12 Rule 2 (Group isolation), Rule 3 (Test set quarantine), Rule 4.
    """
    val_cfg = cfg.get("validation", {})
    n_splits = int(val_cfg.get("n_splits", 5))
    holdout_fraction = float(val_cfg.get("holdout_fraction", 0.20))
    random_seed = int(cfg.get("project", {}).get("random_seed", 42))

    df_train = train_targets.copy()

    if test_targets is not None:
        # We have official test targets
        df_test = test_targets.copy()

        # Check for event_id collision between independent train and test sets
        train_id_set = set(df_train["event_id"])
        test_id_set = set(df_test["event_id"])
        if train_id_set.intersection(test_id_set):
            # To preserve Group Isolation (Rule 2) and avoid collision, offset test IDs
            # or prefix with test_
            if pd.api.types.is_numeric_dtype(df_test["event_id"]):
                max_train = int(df_train["event_id"].max())
                offset = 100_000
                df_test["event_id"] = df_test["event_id"] + offset
            else:
                df_test["event_id"] = df_test["event_id"].apply(lambda x: f"test_{x}")

        # Partition train into n_splits stratified by y_high_risk
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
        df_train["fold"] = -1
        df_train["split"] = "train"

        for fold_idx, (_, val_idx) in enumerate(skf.split(df_train, df_train["y_high_risk"])):
            df_train.iloc[val_idx, df_train.columns.get_loc("fold")] = fold_idx

        df_test["fold"] = -1
        df_test["split"] = "test"

        cols = ["event_id", "split", "fold", "y_high_risk"]
        folds_df = pd.concat([df_train[cols], df_test[cols]], ignore_index=True)

    else:
        # Hold out holdout_fraction from train_targets as test set
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=holdout_fraction, random_state=random_seed
        )
        train_idx, test_idx = next(sss.split(df_train, df_train["y_high_risk"]))

        df_tr = df_train.iloc[train_idx].copy()
        df_te = df_train.iloc[test_idx].copy()

        df_te["split"] = "test"
        df_te["fold"] = -1

        df_tr["split"] = "train"
        df_tr["fold"] = -1
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
        for fold_idx, (_, val_idx) in enumerate(skf.split(df_tr, df_tr["y_high_risk"])):
            df_tr.iloc[val_idx, df_tr.columns.get_loc("fold")] = fold_idx

        cols = ["event_id", "split", "fold", "y_high_risk"]
        folds_df = pd.concat([df_tr[cols], df_te[cols]], ignore_index=True)

    assert_zero_overlap(folds_df)
    return folds_df


def assert_zero_overlap(folds_df: pd.DataFrame) -> None:
    """Assert no event_id appears in multiple splits or multiple CV folds.

    Inputs:
        folds_df: Output of create_splits.
    Outputs:
        None (raises AssertionError on any overlap).
    Rules:
        Section 12 Rule 2 (Group isolation).
    """
    if "event_id" not in folds_df.columns or "split" not in folds_df.columns or "fold" not in folds_df.columns:
        raise ValueError("folds_df must contain 'event_id', 'split', and 'fold'")

    # 1. Total event_ids must equal total rows (each event_id appears exactly once)
    total_rows = len(folds_df)
    unique_events = folds_df["event_id"].nunique()
    if unique_events != total_rows:
        duplicates = folds_df[folds_df["event_id"].duplicated(keep=False)]["event_id"].unique()
        raise AssertionError(
            f"Event ID duplicate detected: {len(duplicates)} events appear multiple times: {duplicates[:5]}"
        )

    # 2. Train vs Test event_id sets must be disjoint
    train_ids = set(folds_df[folds_df["split"] == "train"]["event_id"])
    test_ids = set(folds_df[folds_df["split"] == "test"]["event_id"])
    overlap_train_test = train_ids.intersection(test_ids)
    if overlap_train_test:
        raise AssertionError(
            f"Leakage detected between train and test splits! Overlapping IDs: {list(overlap_train_test)[:5]}"
        )

    # 3. For any pair of CV folds in train, event_id sets must be disjoint
    train_df = folds_df[folds_df["split"] == "train"]
    folds = sorted(train_df["fold"].unique())
    for i in range(len(folds)):
        for j in range(i + 1, len(folds)):
            f_i = set(train_df[train_df["fold"] == folds[i]]["event_id"])
            f_j = set(train_df[train_df["fold"] == folds[j]]["event_id"])
            overlap_folds = f_i.intersection(f_j)
            if overlap_folds:
                raise AssertionError(
                    f"Leakage detected between fold {folds[i]} and fold {folds[j]}! Overlapping IDs: {list(overlap_folds)[:5]}"
                )


def save_folds(folds_df: pd.DataFrame, path: str | Path = "data/processed/folds.parquet") -> Path:
    """Persist folds_df to parquet after asserting zero overlap."""
    assert_zero_overlap(folds_df)
    return save_parquet(folds_df, path)


def load_folds(path: str | Path = "data/processed/folds.parquet") -> pd.DataFrame:
    """Load persisted folds DataFrame from parquet."""
    return load_parquet(path)
