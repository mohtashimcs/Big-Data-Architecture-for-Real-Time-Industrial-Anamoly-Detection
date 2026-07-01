"""
Dataset Acquisition & Cleaning.

Loaders for the two candidate datasets:
  - NASA C-MAPSS turbofan degradation (regression-style RUL -> recast as
    "near-failure window = anomaly" for this anomaly-detection project)
  - SWaT secure water treatment (native Normal/Attack labels)

Both loaders return a common schema so the rest of the pipeline (windowing,
scaling, autoencoder, fast backend, scoring) is dataset-agnostic:

    df_normal : DataFrame of sensor columns only, label == 0 (normal)
    df_test   : DataFrame of sensor columns only, plus aligned `label` array
                (0 = normal, 1 = anomaly) used purely for evaluation in Phase 4.

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
    sensor_columns: list[str]
    normal_df: pd.DataFrame      # sensors only, all rows are "normal"
    test_df: pd.DataFrame        # sensors only
    test_labels: np.ndarray      # 0/1, aligned row-for-row with test_df


# --------------------------------------------------------------------------- #
# NASA C-MAPSS
# --------------------------------------------------------------------------- #
def load_nasa_cmapss(cfg: dict) -> LoadedDataset:
    """
    C-MAPSS ships as whitespace-separated .txt files with no header and a
    trailing pair of empty columns. We:
      1. Read train + test with the documented 26-column schema.
      2. Drop near-constant operating-setting / sensor columns.
      3. Label the final `anomaly_horizon_cycles` of each unit's run-to-failure
         trajectory in the TEST split as anomalous (near-failure degradation),
         everything else as normal. The TRAIN split is used as pure "normal"
         operation since it represents healthy engines early in life.
    """
    c = cfg["nasa_cmapss"]
    columns = c["columns"]

    train_path = Path(c["train_file"])
    test_path = Path(c["test_file"])
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Expected C-MAPSS files at {train_path} and {test_path}. "
            "Download from Kaggle and place them there (see configs/config.yaml)."
        )

    def _read(path: Path) -> pd.DataFrame:
        df = pd.read_csv(path, sep=r"\s+", header=None)
        df = df.iloc[:, : len(columns)]  # drop trailing empty cols if present
        df.columns = columns
        return df

    train_df = _read(train_path)
    test_df = _read(test_path)

    keep_cols = [col for col in columns if col not in c["drop_columns"]]
    sensor_cols = [
        col for col in keep_cols
        if col not in (c["id_column"], c["time_column"])
    ]

    # Missing-value handling: forward/back fill within each unit, then any
    # remaining gaps with the column median.
    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[sensor_cols] = (
            df.groupby(c["id_column"])[sensor_cols]
            .apply(lambda g: g.ffill().bfill())
            .reset_index(drop=True)
        )
        df[sensor_cols] = df[sensor_cols].fillna(df[sensor_cols].median())
        return df

    train_df = _clean(train_df)
    test_df = _clean(test_df)

    # Label test rows: last N cycles of each unit = anomalous (near failure)
    horizon = c["anomaly_horizon_cycles"]
    max_cycle = test_df.groupby(c["id_column"])[c["time_column"]].transform("max")
    test_labels = (max_cycle - test_df[c["time_column"]] < horizon).astype(int).to_numpy()

    normal_df = train_df[sensor_cols].reset_index(drop=True)
    test_sensors_df = test_df[sensor_cols].reset_index(drop=True)

    logger.info(
        "NASA C-MAPSS loaded: %d normal rows, %d test rows (%d anomalous, %.1f%%)",
        len(normal_df), len(test_sensors_df), test_labels.sum(),
        100 * test_labels.mean(),
    )

    return LoadedDataset(
        sensor_columns=sensor_cols,
        normal_df=normal_df,
        test_df=test_sensors_df,
        test_labels=test_labels,
    )


# --------------------------------------------------------------------------- #
# SWaT
# --------------------------------------------------------------------------- #
def load_swat(cfg: dict) -> LoadedDataset:
    """
    SWaT ships as two CSVs: a 'Normal' file (healthy operation, used for AE /
    fast-backend training) and an 'Attack' file (contains injected attacks,
    used as the labeled test set). Both share the same sensor/actuator schema
    plus a Timestamp column and a Normal/Attack label column.
    """
    c = cfg["swat"]
    normal_path = Path(c["normal_file"])
    attack_path = Path(c["attack_file"])
    if not normal_path.exists() or not attack_path.exists():
        raise FileNotFoundError(
            f"Expected SWaT files at {normal_path} and {attack_path}. "
            "Download from Kaggle and place them there (see configs/config.yaml)."
        )

    normal_raw = pd.read_csv(normal_path)
    attack_raw = pd.read_csv(attack_path)

    normal_raw.columns = [col.strip() for col in normal_raw.columns]
    attack_raw.columns = [col.strip() for col in attack_raw.columns]

    drop_cols = [c["timestamp_column"], c["label_column"]]
    sensor_cols = [col for col in normal_raw.columns if col not in drop_cols]

    # Missing-value handling: forward fill then median fallback.
    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[sensor_cols] = df[sensor_cols].apply(pd.to_numeric, errors="coerce")
        df[sensor_cols] = df[sensor_cols].ffill().bfill()
        df[sensor_cols] = df[sensor_cols].fillna(df[sensor_cols].median())
        return df

    normal_raw = _clean(normal_raw)
    attack_raw = _clean(attack_raw)

    test_labels = (
        attack_raw[c["label_column"]].str.strip().str.lower() != c["normal_label"].lower()
    ).astype(int).to_numpy()

    normal_df = normal_raw[sensor_cols].reset_index(drop=True)
    test_sensors_df = attack_raw[sensor_cols].reset_index(drop=True)

    logger.info(
        "SWaT loaded: %d normal rows, %d test rows (%d anomalous, %.1f%%)",
        len(normal_df), len(test_sensors_df), test_labels.sum(),
        100 * test_labels.mean(),
    )

    return LoadedDataset(
        sensor_columns=sensor_cols,
        normal_df=normal_df,
        test_df=test_sensors_df,
        test_labels=test_labels,
    )


def load_dataset(name: str, cfg: dict) -> LoadedDataset:
    if name == "nasa":
        return load_nasa_cmapss(cfg)
    elif name == "swat":
        return load_swat(cfg)
    raise ValueError(f"Unknown dataset '{name}', expected 'nasa' or 'swat'")
