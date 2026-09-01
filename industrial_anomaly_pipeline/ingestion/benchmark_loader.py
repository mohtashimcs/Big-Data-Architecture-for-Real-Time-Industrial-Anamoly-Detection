"""
Benchmark Loader: dataset-specific loaders that normalize open-source
industrial anomaly-detection benchmarks into one common schema so the rest
of the pipeline (windowing, analytics engines, evaluation) is
dataset-agnostic. Also exposes a fully synthetic, no-download benchmark
("TCM5") so latency/throughput/stress experiments never depend on network
access.

Supported sources:
  - NASA C-MAPSS turbofan degradation (real data, ships in this repo under
    data/nasa_cmapss/). RUL-style regression is recast as "near-failure
    window = anomaly", following the same labeling policy used by the
    Phase 1/2 offline pipeline in src/.
  - NASA SMAP/MSL telemetry (Hundman et al. "telemanom" layout: per-channel
    train/test .npy files + labeled_anomalies.csv). Loader is implemented
    against that layout; raises a clear, actionable error if the files
    aren't present locally (no bundled copy -- large, license-gated download).
  - MVTec-AD: image-based visual anomaly detection. Explicitly NOT
    compatible with multivariate-time-series windowing, so the loader
    documents the mismatch and raises rather than faking a fit.
  - TCM5: synthetic 5-channel Tool Condition Monitoring stream generator
    (vibration x/y, spindle load, temperature, acoustic emission), fully
    reproducible via a seed and safe to run anywhere, used for the
    benchmarking / stress-testing experiments in benchmarks/.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_THIS_DIR = Path(__file__).resolve().parent
_PACKAGE_ROOT = _THIS_DIR.parent          # industrial_anomaly_pipeline/
_REPO_ROOT = _PACKAGE_ROOT.parent          # repo root


@dataclass
class LoadedDataset:
    sensor_columns: list[str]
    normal_df: pd.DataFrame  # sensors only, all rows are "normal"
    test_df: pd.DataFrame  # sensors only
    test_labels: np.ndarray  # 0/1, aligned row-for-row with test_df

    def to_replay_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Convenience: (normal_array, test_array) ready to hand to a
        `SensorStreamProducer(source=...)` for replay-mode streaming."""
        return (
            self.normal_df.to_numpy(dtype=np.float64),
            self.test_df.to_numpy(dtype=np.float64),
        )


def _resolve_data_root(*candidates: str) -> Path:
    """Return the first existing directory among candidate locations,
    checked relative to both the pipeline package and the repo root, so the
    loader works whether datasets live in `industrial_anomaly_pipeline/data/`
    or the repo-level `data/` used by earlier phases."""
    for rel in candidates:
        for base in (_PACKAGE_ROOT, _REPO_ROOT):
            path = base / rel
            if path.exists():
                return path
    # Nothing found: return the package-local path as the "expected" location
    # so downstream FileNotFoundError messages point somewhere sensible.
    return _PACKAGE_ROOT / candidates[0]


def _sanitize(df: pd.DataFrame, sensor_cols: list[str]) -> pd.DataFrame:
    """Forward/back-fill within-column gaps, then fall back to the column
    median for anything still missing (e.g. a fully-NaN leading run) --
    keeps NaN/Inf out of every downstream matrix operation."""
    df = df.copy()
    df[sensor_cols] = df[sensor_cols].apply(pd.to_numeric, errors="coerce")
    df[sensor_cols] = df[sensor_cols].replace([np.inf, -np.inf], np.nan)
    df[sensor_cols] = df[sensor_cols].ffill().bfill()
    medians = df[sensor_cols].median()
    df[sensor_cols] = df[sensor_cols].fillna(medians).fillna(0.0)
    return df


# --------------------------------------------------------------------------- #
# NASA C-MAPSS (real data, ships with this repo)
# --------------------------------------------------------------------------- #
_CMAPSS_COLUMNS = (
    ["unit_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)
_CMAPSS_DROP_COLUMNS = [
    "op_setting_3", "sensor_1", "sensor_5", "sensor_6",
    "sensor_10", "sensor_16", "sensor_18", "sensor_19",
]


def load_nasa_cmapss(
    subset: str = "FD001",
    data_root: Optional[Path] = None,
    anomaly_horizon_cycles: int = 30,
) -> LoadedDataset:
    """
    C-MAPSS ships as whitespace-separated .txt files with no header. Train
    split (healthy engines early in life) is used as pure "normal"
    operation; the last `anomaly_horizon_cycles` of each test unit's
    run-to-failure trajectory are labeled anomalous (near-failure
    degradation).
    """
    root = data_root or _resolve_data_root("data/nasa_cmapss")
    train_path = root / f"train_{subset}.txt"
    test_path = root / f"test_{subset}.txt"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Expected C-MAPSS files at {train_path} and {test_path}. "
            "Download from Kaggle (NASA Turbofan Engine Degradation "
            "Simulation) and place them under data/nasa_cmapss/."
        )

    def _read(path: Path) -> pd.DataFrame:
        df = pd.read_csv(path, sep=r"\s+", header=None)
        df = df.iloc[:, : len(_CMAPSS_COLUMNS)]
        df.columns = _CMAPSS_COLUMNS
        return df

    train_df = _read(train_path)
    test_df = _read(test_path)

    keep_cols = [c for c in _CMAPSS_COLUMNS if c not in _CMAPSS_DROP_COLUMNS]
    sensor_cols = [c for c in keep_cols if c not in ("unit_id", "cycle")]

    train_df = _sanitize(train_df, sensor_cols)
    test_df = _sanitize(test_df, sensor_cols)

    max_cycle = test_df.groupby("unit_id")["cycle"].transform("max")
    test_labels = (
        (max_cycle - test_df["cycle"] < anomaly_horizon_cycles).astype(int).to_numpy()
    )

    # C-MAPSS train trajectories run every unit to failure -- the raw train
    # split is NOT purely healthy operation, it silently includes the same
    # near-failure degradation pattern the test split labels as anomalous.
    # Fitting "normal" baselines on it lets both engines partially learn to
    # reconstruct/predict degraded dynamics as normal, which measurably
    # depresses recall. Trim the same trailing `anomaly_horizon_cycles`
    # window from train too, so normal-only fitting sees a support disjoint
    # from what test calls anomalous -- consistent with the labeling policy
    # rather than contradicting it.
    train_max_cycle = train_df.groupby("unit_id")["cycle"].transform("max")
    train_df = train_df[train_max_cycle - train_df["cycle"] >= anomaly_horizon_cycles]

    normal_df = train_df[sensor_cols].reset_index(drop=True)
    test_sensors_df = test_df[sensor_cols].reset_index(drop=True)

    logger.info(
        "NASA C-MAPSS %s loaded: %d normal rows, %d test rows (%d anomalous, %.1f%%)",
        subset, len(normal_df), len(test_sensors_df), test_labels.sum(),
        100 * test_labels.mean(),
    )
    return LoadedDataset(
        sensor_columns=sensor_cols,
        normal_df=normal_df,
        test_df=test_sensors_df,
        test_labels=test_labels,
    )


# --------------------------------------------------------------------------- #
# NASA SMAP/MSL (telemanom layout) -- best-effort, requires local download
# --------------------------------------------------------------------------- #
def load_smap_msl(
    spacecraft: str = "SMAP",
    data_root: Optional[Path] = None,
) -> LoadedDataset:
    """
    Expects the standard "telemanom" layout (Hundman et al., 2018):
        <data_root>/train/<chan_id>.npy   -- normal telemetry, (T, n_features)
        <data_root>/test/<chan_id>.npy    -- labeled test telemetry
        <data_root>/labeled_anomalies.csv -- columns: chan_id, spacecraft,
                                              anomaly_sequences (list of
                                              [start, end] index pairs)
    Not bundled with this repo (license-gated download); raises with setup
    instructions if the layout isn't present locally.
    """
    root = data_root or _resolve_data_root("data/smap_msl")
    labels_path = root / "labeled_anomalies.csv"
    train_dir, test_dir = root / "train", root / "test"
    if not labels_path.exists() or not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(
            f"Expected SMAP/MSL telemanom layout under {root} "
            "(train/<chan>.npy, test/<chan>.npy, labeled_anomalies.csv). "
            "Download from https://github.com/khundman/telemanom and place "
            "the extracted `data/` contents there."
        )

    labels_df = pd.read_csv(labels_path)
    labels_df = labels_df[labels_df["spacecraft"] == spacecraft]
    if labels_df.empty:
        raise ValueError(f"No channels found for spacecraft={spacecraft!r} in {labels_path}")

    normal_chunks, test_chunks, label_chunks = [], [], []
    for _, row in labels_df.iterrows():
        chan_id = row["chan_id"]
        train_arr = np.load(train_dir / f"{chan_id}.npy")
        test_arr = np.load(test_dir / f"{chan_id}.npy")

        chan_labels = np.zeros(len(test_arr), dtype=int)
        for start, end in eval(row["anomaly_sequences"]):  # e.g. "[[100,200]]"
            chan_labels[start:end] = 1

        normal_chunks.append(train_arr)
        test_chunks.append(test_arr)
        label_chunks.append(chan_labels)

    # Channels have independent lengths/feature counts in telemanom; align by
    # truncating every channel's test split to the shortest one so rows stay
    # aligned across channels for a joint multivariate window.
    min_test_len = min(len(a) for a in test_chunks)
    n_features = min(a.shape[1] if a.ndim > 1 else 1 for a in normal_chunks)

    def _flatten(chunks: list[np.ndarray], length: int) -> np.ndarray:
        cols = [c[:length, :n_features] if c.ndim > 1 else c[:length, None] for c in chunks]
        return np.concatenate(cols, axis=1)

    min_train_len = min(len(a) for a in normal_chunks)
    normal_array = _flatten(normal_chunks, min_train_len)
    test_array = _flatten(test_chunks, min_test_len)
    test_labels = np.array(
        [chan[:min_test_len] for chan in label_chunks]
    ).max(axis=0)  # a timestep is anomalous if ANY channel flags it

    sensor_cols = [f"{spacecraft.lower()}_chan_{i}" for i in range(normal_array.shape[1])]
    normal_df = pd.DataFrame(normal_array, columns=sensor_cols)
    test_df = pd.DataFrame(test_array, columns=sensor_cols)
    normal_df = _sanitize(normal_df, sensor_cols)
    test_df = _sanitize(test_df, sensor_cols)

    logger.info(
        "%s loaded: %d channels, %d normal rows, %d test rows (%d anomalous)",
        spacecraft, len(labels_df), len(normal_df), len(test_df), test_labels.sum(),
    )
    return LoadedDataset(
        sensor_columns=sensor_cols, normal_df=normal_df, test_df=test_df, test_labels=test_labels
    )


# --------------------------------------------------------------------------- #
# MVTec-AD -- explicitly out of scope for MTS windowing
# --------------------------------------------------------------------------- #
def load_mvtec(*_args, **_kwargs) -> LoadedDataset:
    """
    MVTec-AD is an image-patch visual-defect benchmark, not a
    multivariate-time-series stream: there is no natural notion of a
    contiguous sensor "window" over its samples. Rather than reshaping
    pixels into a fake sensor vector (which would silently misrepresent
    what the model is learning), this loader raises. Swap in a CNN-based
    detector behind `BaseAnomalyDetector` if vision-based inspection is
    ever required alongside the MTS engines.
    """
    raise NotImplementedError(
        "MVTec-AD is image-based and not representable as an MTS window "
        "stream; unsupported by this pipeline's windowing/analytics layer."
    )


# --------------------------------------------------------------------------- #
# TCM5 -- synthetic Tool Condition Monitoring stream (no download required)
# --------------------------------------------------------------------------- #
_TCM5_CHANNELS = [
    "vibration_x", "vibration_y", "spindle_load", "temperature_c", "acoustic_emission",
]


def generate_tcm5_synthetic(
    n_normal: int = 6000,
    n_test: int = 3000,
    anomaly_fraction: float = 0.12,
    seed: int = 7,
) -> LoadedDataset:
    """
    Synthesizes a 5-channel machining-tool telemetry stream:
      - vibration_x/y: rotational-frequency sine waves + noise (seasonal).
      - spindle_load / temperature_c: slow trend (thermal ramp-up) + noise.
      - acoustic_emission: white noise floor, spikes during induced faults.

    The test split contains contiguous "tool wear / spindle overload"
    anomaly segments where amplitude, trend slope, and noise floor all
    shift simultaneously across channels -- a genuinely multivariate
    anomaly signature the DMD/AE engines both key off differently
    (DMD via one-step prediction error, AE via reconstruction error).
    """
    rng = np.random.default_rng(seed)
    n_features = len(_TCM5_CHANNELS)

    def _baseline(n: int, t0: int = 0) -> np.ndarray:
        t = np.arange(t0, t0 + n)
        vib_x = np.sin(0.31 * t) + rng.normal(0, 0.08, n)
        vib_y = np.sin(0.31 * t + np.pi / 4) + rng.normal(0, 0.08, n)
        spindle_load = 0.5 + 0.0002 * t + rng.normal(0, 0.05, n)
        temperature = 40 + 0.0015 * t + rng.normal(0, 0.3, n)
        acoustic = rng.normal(0, 0.15, n)
        return np.stack([vib_x, vib_y, spindle_load, temperature, acoustic], axis=1)

    normal_array = _baseline(n_normal)

    test_array = _baseline(n_test, t0=n_normal)
    test_labels = np.zeros(n_test, dtype=int)

    n_anomalous = int(n_test * anomaly_fraction)
    n_segments = max(1, n_anomalous // 150)
    seg_len = max(20, n_anomalous // n_segments)
    for _ in range(n_segments):
        start = int(rng.integers(0, max(1, n_test - seg_len)))
        end = min(n_test, start + seg_len)
        severity = rng.uniform(2.5, 5.0)
        test_array[start:end, 0] += severity * np.sin(1.4 * np.arange(end - start))  # vib_x
        test_array[start:end, 1] += severity * np.sin(1.4 * np.arange(end - start) + 1.0)
        test_array[start:end, 2] += severity * 0.3  # spindle load spike
        test_array[start:end, 3] += severity * 2.0  # temperature spike
        test_array[start:end, 4] += rng.normal(0, severity * 0.5, end - start)  # acoustic burst
        test_labels[start:end] = 1

    normal_df = pd.DataFrame(normal_array, columns=_TCM5_CHANNELS)
    test_df = pd.DataFrame(test_array, columns=_TCM5_CHANNELS)

    logger.info(
        "TCM5 synthetic generated: %d normal rows, %d test rows (%d anomalous, %.1f%%)",
        len(normal_df), len(test_df), test_labels.sum(), 100 * test_labels.mean(),
    )
    return LoadedDataset(
        sensor_columns=_TCM5_CHANNELS, normal_df=normal_df, test_df=test_df,
        test_labels=test_labels,
    )


def load_benchmark(name: str, **kwargs) -> LoadedDataset:
    """Dispatch by benchmark name: 'nasa_cmapss' | 'smap_msl' | 'tcm5' | 'mvtec'."""
    loaders = {
        "nasa_cmapss": load_nasa_cmapss,
        "smap_msl": load_smap_msl,
        "tcm5": generate_tcm5_synthetic,
        "mvtec": load_mvtec,
    }
    if name not in loaders:
        raise ValueError(f"Unknown benchmark {name!r}, expected one of {sorted(loaders)}")
    return loaders[name](**kwargs)
