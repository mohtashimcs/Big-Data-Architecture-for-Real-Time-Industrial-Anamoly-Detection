"""
normalize sensor metrics and structure data into
contiguous time-series windows; split normal data into train/validation.

Critical correctness point: the MinMax scaler is fit
ONLY on the normal-operation data, then applied unchanged to the test split.
This avoids leaking test-set (anomalous) statistics into the normalization,
which would bias reconstruction-error baselines optimistically.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ScalerStats:
    data_min: np.ndarray
    data_max: np.ndarray
    eps: float = 1e-8

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.data_min) / (self.data_max - self.data_min + self.eps)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * (self.data_max - self.data_min + self.eps) + self.data_min


def fit_minmax_on_normal(normal_array: np.ndarray) -> ScalerStats:
    """MinMax Scaling fit exclusively on normal operational data."""
    return ScalerStats(
        data_min=normal_array.min(axis=0),
        data_max=normal_array.max(axis=0),
    )


def make_windows(array: np.ndarray, window_size: int, stride: int = 1) -> np.ndarray:
    """
    structure a (T, F) array of contiguous sensor readings into
    overlapping windows of shape (num_windows, window_size, F).
    """
    t, f = array.shape
    if t < window_size:
        raise ValueError(f"Series length {t} shorter than window_size {window_size}")
    starts = range(0, t - window_size + 1, stride)
    windows = np.stack([array[s:s + window_size] for s in starts], axis=0)
    return windows


def window_labels(labels: np.ndarray, window_size: int, stride: int = 1) -> np.ndarray:
    """
    A window is labeled anomalous if ANY reading inside it is anomalous
    (conservative — favors catching partial-window anomalies, standard
    choice for industrial CPS anomaly detection).
    """
    t = labels.shape[0]
    starts = range(0, t - window_size + 1, stride)
    return np.array([int(labels[s:s + window_size].max()) for s in starts])


def train_val_split(windows: np.ndarray, val_split: float, seed: int = 42):
    """hold out a validation slice of normal windows for the AE."""
    n = windows.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = max(1, int(n * val_split))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    return windows[train_idx], windows[val_idx]


def prepare_pipeline_data(loaded, window_size: int, stride: int, val_split: float):
    """
    scale (fit on normal only) -> window ->
    split normal windows into train/val -> window the labeled test split.

    Returns a dict with everything Phase 2 (Modelling) needs.
    """
    normal_array = loaded.normal_df.to_numpy(dtype=np.float64)
    test_array = loaded.test_df.to_numpy(dtype=np.float64)

    scaler = fit_minmax_on_normal(normal_array)
    normal_scaled = scaler.transform(normal_array)
    test_scaled = scaler.transform(test_array)

    normal_windows = make_windows(normal_scaled, window_size, stride)
    train_windows, val_windows = train_val_split(normal_windows, val_split)

    test_windows = make_windows(test_scaled, window_size, stride)
    test_window_labels = window_labels(loaded.test_labels, window_size, stride)

    return {
        "scaler": scaler,
        "train_windows": train_windows,
        "val_windows": val_windows,
        "test_windows": test_windows,
        "test_window_labels": test_window_labels,
        "sensor_columns": loaded.sensor_columns,
    }
