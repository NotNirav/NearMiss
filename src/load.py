"""Data and configuration loading utilities.

Only this module may call pandas.read_csv on files in data/raw/.
Rules:
- Section 12 Rule 1 (Cutoff isolation): only load.py, cutoff.py, and the EDA
  notebook touch raw files; no read_csv outside load.py.
- Section 12 Rule 4 (No magic numbers): load paths and parameters from config.yaml.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def load_config(config_path: str | Path = "config.yaml") -> dict[str, Any]:
    """Load YAML configuration from config_path.

    Inputs:
        config_path: Path to config.yaml.
    Outputs:
        Parsed configuration dictionary.
    Rules:
        Section 12 Rule 4 (No magic numbers).
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Configuration file {path} did not parse as a dictionary.")
    return cfg


def load_raw_cdms(
    raw_dir: str | Path = "data/raw",
    split: str = "train",
) -> pd.DataFrame:
    """Read raw Zenodo CDM CSV file(s) from raw_dir.

    Inputs:
        raw_dir: Directory containing raw Zenodo CSVs (read-only).
        split: One of 'train', 'test', 'test_private', or 'test_merged'.
    Outputs:
        Raw CDM DataFrame with original columns intact.
    Rules:
        Section 12 Rule 1 (Cutoff isolation): only place read_csv is allowed.
    """
    path = Path(raw_dir)
    if not path.exists():
        raise FileNotFoundError(f"Raw directory not found: {path}")

    # Find matching files in raw_dir recursively
    csv_files = list(path.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    file_map: dict[str, Path] = {}
    for f in csv_files:
        file_map[f.name.lower()] = f

    target_file: Path | None = None
    if split == "train":
        for name in ("train_data.csv", "train.csv", "cdm_train.csv"):
            if name in file_map:
                target_file = file_map[name]
                break
    elif split == "test":
        for name in ("test_data.csv", "test.csv", "cdm_test.csv"):
            if name in file_map:
                target_file = file_map[name]
                break
    elif split == "test_private":
        for name in ("test_data_private.csv", "test_private.csv", "test_labels.csv"):
            if name in file_map:
                target_file = file_map[name]
                break
    elif split == "test_merged":
        # Combine visible test CDMs with private final CDMs into one CDM history
        test_file = file_map.get("test_data.csv") or file_map.get("test.csv")
        priv_file = file_map.get("test_data_private.csv") or file_map.get("test_private.csv")
        if test_file and priv_file:
            df_test = pd.read_csv(test_file)
            df_priv = pd.read_csv(priv_file)
            # test_data_private has 'true_risk'; rename to 'risk' for unified format
            if "true_risk" in df_priv.columns and "risk" not in df_priv.columns:
                df_priv = df_priv.rename(columns={"true_risk": "risk"})
            return pd.concat([df_test, df_priv], ignore_index=True)

    if target_file is None:
        # Fallback search
        for name, p in file_map.items():
            if split.lower() in name:
                target_file = p
                break

    if target_file is None or not target_file.exists():
        raise FileNotFoundError(
            f"No raw CDM CSV found for split '{split}' in {raw_dir} (available: {list(file_map.keys())})."
        )

    df = pd.read_csv(target_file)
    if split == "test_private" and "true_risk" in df.columns and "risk" not in df.columns:
        df = df.rename(columns={"true_risk": "risk"})
    return df


def save_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    """Write a DataFrame to parquet, creating parent directories if needed.

    Inputs:
        df: DataFrame to persist.
        path: Destination .parquet path.
    Outputs:
        Resolved Path to the written file.
    Rules:
        Section 12 Rule 4.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return out_path.resolve()


def load_parquet(path: str | Path) -> pd.DataFrame:
    """Read a parquet file into a DataFrame.

    Inputs:
        path: Source .parquet path.
    Outputs:
        Loaded DataFrame.
    Rules:
        Section 12 Rule 1.
    """
    in_path = Path(path)
    if not in_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {in_path}")
    return pd.read_parquet(in_path)
