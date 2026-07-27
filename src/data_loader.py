"""
Phase 1 / Step 2: Dataset Acquisition & Cleaning.

Loaders for both candidate datasets, normalized to a common schema so every
downstream stage (windowing, autoencoder, fast backend, scoring, streaming)
is dataset-agnostic:

    sensor_columns : list[str]
    normal_df      : DataFrame, sensors only, all rows normal operation
    test_df        : DataFrame, sensors only
    test_labels    : np.ndarray of 0/1 aligned row-for-row with test_df

Download the raw files from Kaggle into data/nasa_cmapss/ or data/swat/
as referenced in configs/config.yaml.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class LoadedDataset:
    sensor_columns: list
    normal_df: pd.DataFrame
    test_df: pd.DataFrame
    test_labels: np.ndarray


# --------------------------------------------------------------------------- #
# NASA C-MAPSS
# --------------------------------------------------------------------------- #
def load_nasa_cmapss(cfg: dict) -> LoadedDataset:
    """
    C-MAPSS ships as whitespace-separated .txt with no header and two
    trailing empty columns.

      - TRAIN split = healthy run-to-failure trajectories -> "normal" pool.
      - TEST split  = labeled by marking the final `anomaly_horizon_cycles`
        of each unit's trajectory as anomalous (near-failure degradation).
    """
    c = cfg["nasa_cmapss"]
    columns = c["columns"]

    train_path = Path(c["train_file"])
    test_path = Path(c["test_file"])
    for pth in (train_path, test_path):
        if not pth.exists():
            raise FileNotFoundError(
                f"Expected C-MAPSS file at {pth}. Download from Kaggle and place "
                "it there (see configs/config.yaml and README.md)."
            )

    def _read(path: Path) -> pd.DataFrame:
        df = pd.read_csv(path, sep=r"\s+", header=None)
        df = df.iloc[:, : len(columns)]      # drop trailing empty columns
        df.columns = columns
        return df

    train_df = _read(train_path)
    test_df = _read(test_path)

    keep_cols = [col for col in columns if col not in c["drop_columns"]]
    sensor_cols = [col for col in keep_cols
                   if col not in (c["id_column"], c["time_column"])]

    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        """Missing-value handling: per-unit ffill/bfill, then median fallback."""
        df = df.sort_values([c["id_column"], c["time_column"]]).reset_index(drop=True)
        grp = df.groupby(c["id_column"])[sensor_cols]
        # NOTE: .ffill()/.bfill() on a groupby return index-aligned frames,
        # unlike .apply(), which can silently reorder rows. Assigning the
        # result of .apply() here was a latent alignment bug.
        df[sensor_cols] = grp.ffill()
        df[sensor_cols] = df.groupby(c["id_column"])[sensor_cols].bfill()
        df[sensor_cols] = df[sensor_cols].fillna(df[sensor_cols].median())
        return df

    train_df = _clean(train_df)
    test_df = _clean(test_df)

    horizon = c["anomaly_horizon_cycles"]
    max_cycle = test_df.groupby(c["id_column"])[c["time_column"]].transform("max")
    test_labels = (max_cycle - test_df[c["time_column"]] < horizon).astype(int).to_numpy()

    normal_df = train_df[sensor_cols].reset_index(drop=True)
    test_sensors = test_df[sensor_cols].reset_index(drop=True)

    logger.info("C-MAPSS: %d normal rows, %d test rows (%.1f%% anomalous)",
                len(normal_df), len(test_sensors), 100 * test_labels.mean())
    return LoadedDataset(sensor_cols, normal_df, test_sensors, test_labels)


# --------------------------------------------------------------------------- #
# SWaT
# --------------------------------------------------------------------------- #
def load_swat(cfg: dict) -> LoadedDataset:
    """
    SWaT ships as two CSVs sharing a schema: a Normal-operation file (used as
    the training pool) and an Attack file with a Normal/Attack label column
    (used as the labeled test set).

    SWaT is ~500k rows at 1Hz; `row_subsample` takes every Nth row to keep
    memory tractable. Because sampling is uniform in time, the temporal
    structure each window needs is preserved (at a coarser sample rate).
    """
    c = cfg["swat"]
    normal_path = Path(c["normal_file"])
    attack_path = Path(c["attack_file"])
    for pth in (normal_path, attack_path):
        if not pth.exists():
            raise FileNotFoundError(
                f"Expected SWaT file at {pth}. Download from Kaggle and place "
                "it there (see configs/config.yaml and README.md)."
            )

    normal_raw = pd.read_csv(normal_path)
    attack_raw = pd.read_csv(attack_path)
    normal_raw.columns = [str(col).strip() for col in normal_raw.columns]
    attack_raw.columns = [str(col).strip() for col in attack_raw.columns]

    ts_col, label_col = c["timestamp_column"], c["label_column"]
    if label_col not in attack_raw.columns:
        raise KeyError(
            f"Label column '{label_col}' not found in {attack_path}. "
            f"Columns present: {list(attack_raw.columns)[:10]}... "
            "Update swat.label_column in configs/config.yaml to match."
        )

    drop_cols = [ts_col, label_col]
    sensor_cols = [col for col in normal_raw.columns if col not in drop_cols]

    step = max(1, int(c.get("row_subsample", 1)))
    normal_raw = normal_raw.iloc[::step].reset_index(drop=True)
    attack_raw = attack_raw.iloc[::step].reset_index(drop=True)

    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[sensor_cols] = df[sensor_cols].apply(pd.to_numeric, errors="coerce")
        df[sensor_cols] = df[sensor_cols].ffill().bfill()
        df[sensor_cols] = df[sensor_cols].fillna(df[sensor_cols].median())
        return df

    normal_raw = _clean(normal_raw)
    attack_raw = _clean(attack_raw)

    test_labels = (
        attack_raw[label_col].astype(str).str.strip().str.lower()
        != str(c["normal_label"]).strip().lower()
    ).astype(int).to_numpy()

    normal_df = normal_raw[sensor_cols].reset_index(drop=True)
    test_sensors = attack_raw[sensor_cols].reset_index(drop=True)

    logger.info("SWaT: %d normal rows, %d test rows (%.1f%% anomalous), subsample=%d",
                len(normal_df), len(test_sensors), 100 * test_labels.mean(), step)
    return LoadedDataset(sensor_cols, normal_df, test_sensors, test_labels)


def load_dataset(name: str, cfg: dict) -> LoadedDataset:
    if name == "nasa":
        return load_nasa_cmapss(cfg)
    if name == "swat":
        return load_swat(cfg)
    raise ValueError(f"Unknown dataset '{name}' (expected 'nasa' or 'swat')")
