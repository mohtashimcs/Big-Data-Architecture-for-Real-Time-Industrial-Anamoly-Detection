"""
Phase 1 / Step 2-3: normalization, windowing, and normal/test separation.

Key correctness point for the write-up: the MinMax scaler is fit ONLY on
normal-operation data and then applied unchanged to the test split. Fitting
on the full data would leak anomalous-region statistics into normalization
and optimistically bias every downstream reconstruction-error baseline.
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
    """Step 2: MinMax scaling fit exclusively on normal operational data."""
    return ScalerStats(data_min=normal_array.min(axis=0),
                       data_max=normal_array.max(axis=0))


def make_windows(array: np.ndarray, window_size: int, stride: int = 1) -> np.ndarray:
    """
    Structure a (T, F) array into (num_windows, window_size, F).

    Uses a strided view then copies, which avoids building a Python list of
    thousands of slices (materially faster and lower peak memory on SWaT).
    """
    t, f = array.shape
    if t < window_size:
        raise ValueError(f"Series length {t} < window_size {window_size}")
    array = np.ascontiguousarray(array, dtype=np.float64)
    n_windows = (t - window_size) // stride + 1
    s0, s1 = array.strides
    view = np.lib.stride_tricks.as_strided(
        array, shape=(n_windows, window_size, f), strides=(s0 * stride, s0, s1),
        writeable=False,
    )
    return np.array(view, copy=True)


def window_labels(labels: np.ndarray, window_size: int, stride: int = 1) -> np.ndarray:
    """
    A window is anomalous if ANY reading inside it is anomalous — a
    conservative policy favouring recall on partial-window anomalies, and a
    labeling decision worth stating explicitly in the Methodology chapter
    since it affects reported F1.
    """
    t = labels.shape[0]
    n_windows = (t - window_size) // stride + 1
    starts = np.arange(n_windows) * stride
    return np.array([int(labels[s:s + window_size].max()) for s in starts])


def subsample_windows(windows: np.ndarray, max_count: int,
                       labels: np.ndarray | None = None):
    """
    Evenly subsample windows down to `max_count`, preserving coverage across
    the whole series rather than truncating to a prefix. Returns (windows,
    labels) with labels subsampled identically when provided.

    Without this, SWaT at stride 1 produces hundreds of thousands of windows
    (many GB once materialized as float64).
    """
    n = len(windows)
    if max_count <= 0 or n <= max_count:
        return windows, labels
    idx = np.linspace(0, n - 1, max_count).astype(int)
    return windows[idx], (labels[idx] if labels is not None else None)


def train_val_split(windows: np.ndarray, val_split: float, seed: int = 42):
    """Step 3: hold out a slice of NORMAL windows for AE early stopping and
    for threshold calibration (never uses anomaly labels)."""
    n = windows.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = max(1, int(n * val_split))
    return windows[idx[n_val:]], windows[idx[:n_val]]


def prepare_pipeline_data(loaded, window_size: int, stride: int, val_split: float,
                           max_train_windows: int = 0, max_test_windows: int = 0):
    """
    End-to-end Step 2-3: scale (normal-only fit) -> window -> cap size ->
    split normal into train/val -> window the labeled test split.
    """
    normal_array = loaded.normal_df.to_numpy(dtype=np.float64)
    test_array = loaded.test_df.to_numpy(dtype=np.float64)

    scaler = fit_minmax_on_normal(normal_array)
    normal_scaled = scaler.transform(normal_array)
    test_scaled = scaler.transform(test_array)

    normal_windows = make_windows(normal_scaled, window_size, stride)
    normal_windows, _ = subsample_windows(normal_windows, max_train_windows)
    train_windows, val_windows = train_val_split(normal_windows, val_split)

    test_windows = make_windows(test_scaled, window_size, stride)
    test_lbls = window_labels(loaded.test_labels, window_size, stride)
    test_windows, test_lbls = subsample_windows(test_windows, max_test_windows, test_lbls)

    return {
        "scaler": scaler,
        "train_windows": train_windows,
        "val_windows": val_windows,
        "test_windows": test_windows,
        "test_window_labels": test_lbls,
        "sensor_columns": loaded.sensor_columns,
        "test_scaled_array": test_scaled,   # kept for the Phase 3 stream replay
    }
