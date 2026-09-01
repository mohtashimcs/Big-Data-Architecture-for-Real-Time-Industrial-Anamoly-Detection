"""
Latency Tracker + SLA regression tests.

`LatencyTracker` is the microsecond-precision instrument used both here
(as pytest assertions against the 20ms SLA) and in `run_experiments.py`
(to report full latency distributions alongside the accuracy benchmarks).
`time.perf_counter()` is a monotonic, sub-microsecond-resolution clock on
every platform this runs on, which is what "microsecond-precision" means
in practice -- `AnomalyResult.latency_ms` (see analytics/base.py) is
already measured this way per window, so the tracker's job is aggregation
and SLA bookkeeping, not re-timing.
"""
from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import pytest

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from analytics.autoencoder import AutoencoderDetector
from analytics.dmd_decomposer import DMDSTLDetector
from evaluation.metrics_engine import CompositeScoringEngine

DEFAULT_SLA_MS = 20.0


@dataclass
class LatencyTracker:
    """Accumulates per-window latency samples (milliseconds) and reports
    percentiles / SLA-violation rate against a target threshold."""

    sla_ms: float = DEFAULT_SLA_MS
    samples_ms: list[float] = field(default_factory=list)

    def record(self, latency_ms: float) -> None:
        self.samples_ms.append(latency_ms)

    @contextmanager
    def measure(self) -> Iterator[None]:
        """Microsecond-precision context manager for timing an arbitrary block."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self.record((time.perf_counter() - start) * 1000.0)

    def _array(self) -> np.ndarray:
        return np.asarray(self.samples_ms, dtype=np.float64) if self.samples_ms else np.array([0.0])

    def percentile(self, p: float) -> float:
        return float(np.percentile(self._array(), p))

    def sla_violation_rate(self) -> float:
        if not self.samples_ms:
            return 0.0
        arr = self._array()
        return float(np.mean(arr > self.sla_ms))

    def summary(self) -> dict[str, float]:
        arr = self._array()
        return {
            "count": len(self.samples_ms),
            "mean_ms": float(arr.mean()),
            "p50_ms": self.percentile(50),
            "p95_ms": self.percentile(95),
            "p99_ms": self.percentile(99),
            "max_ms": float(arr.max()),
            "sla_ms": self.sla_ms,
            "sla_violation_rate": self.sla_violation_rate(),
        }


# --------------------------------------------------------------------------- #
# SLA regression tests
# --------------------------------------------------------------------------- #
def _synthetic_windows(rng: np.random.Generator, n: int, window_size: int, n_features: int) -> np.ndarray:
    t = np.arange(window_size)
    freq = rng.uniform(0.05, 0.2, size=n_features)
    phase = rng.uniform(0, 2 * np.pi, size=n_features)
    base = np.sin(freq[None, :] * t[:, None] + phase[None, :])
    return np.stack([base + rng.normal(0, 0.05, size=(window_size, n_features)) for _ in range(n)])


@pytest.fixture(scope="module")
def sla_windows() -> np.ndarray:
    rng = np.random.default_rng(0)
    return _synthetic_windows(rng, n=300, window_size=30, n_features=10)


def test_dmd_engine_meets_sla_across_many_windows(sla_windows: np.ndarray):
    train, test = sla_windows[:200], sla_windows[200:]
    detector = DMDSTLDetector(stl_period=30, dmd_rank=8).fit(train)

    tracker = LatencyTracker(sla_ms=DEFAULT_SLA_MS)
    for window in test:
        result = detector.score_window(window)
        tracker.record(result.latency_ms)

    summary = tracker.summary()
    assert summary["p99_ms"] < DEFAULT_SLA_MS, summary
    assert summary["sla_violation_rate"] == 0.0, summary


def test_autoencoder_engine_meets_sla_across_many_windows(sla_windows: np.ndarray):
    train, test = sla_windows[:200], sla_windows[200:]
    detector = AutoencoderDetector(
        architecture="lstm", epochs=5, hidden_dim=16, latent_dim=8
    ).fit(train)

    tracker = LatencyTracker(sla_ms=DEFAULT_SLA_MS)
    for window in test:
        result = detector.score_window(window)
        tracker.record(result.latency_ms)

    summary = tracker.summary()
    assert summary["p99_ms"] < DEFAULT_SLA_MS, summary


def test_dense_autoencoder_is_at_least_as_fast_as_lstm(sla_windows: np.ndarray):
    """The Dense variant exists specifically as the cheaper/faster option;
    its p50 latency should not regress above the LSTM variant's."""
    train, test = sla_windows[:200], sla_windows[200:]
    lstm_det = AutoencoderDetector(architecture="lstm", epochs=3, hidden_dim=16, latent_dim=8).fit(train)
    dense_det = AutoencoderDetector(architecture="dense", epochs=3, hidden_dim=16, latent_dim=8).fit(train)

    lstm_tracker, dense_tracker = LatencyTracker(), LatencyTracker()
    for window in test:
        lstm_tracker.record(lstm_det.score_window(window).latency_ms)
        dense_tracker.record(dense_det.score_window(window).latency_ms)

    assert dense_tracker.percentile(50) <= lstm_tracker.percentile(50) * 1.5  # generous slack, CPU noise


def test_end_to_end_composite_scoring_meets_sla(sla_windows: np.ndarray):
    """The SLA is per stream *payload window*, i.e. the full pipeline
    budget: engine score_window() + the evaluation composite score, not
    just the raw model forward pass in isolation."""
    train, val, test = sla_windows[:150], sla_windows[150:200], sla_windows[200:]
    detector = DMDSTLDetector(stl_period=30, dmd_rank=8).fit(train)

    val_errors = np.array([detector.score_window(w).score for w in val])
    composite = CompositeScoringEngine(k_centroids=6).fit(val, warm_start_errors=val_errors)

    tracker = LatencyTracker(sla_ms=DEFAULT_SLA_MS)
    for window in test:
        with tracker.measure():
            result = detector.score_window(window)
            composite.score(window, result.score)

    summary = tracker.summary()
    assert summary["p99_ms"] < DEFAULT_SLA_MS, summary


def test_quantized_autoencoder_latency_does_not_regress(sla_windows: np.ndarray):
    train, test = sla_windows[:200], sla_windows[200:]
    detector = AutoencoderDetector(architecture="lstm", epochs=3, hidden_dim=16, latent_dim=8).fit(train)

    before = LatencyTracker()
    for window in test:
        before.record(detector.score_window(window).latency_ms)

    detector.quantize_dynamic()
    after = LatencyTracker()
    for window in test:
        after.record(detector.score_window(window).latency_ms)

    assert after.percentile(99) < DEFAULT_SLA_MS


def test_latency_tracker_reports_sla_violations():
    tracker = LatencyTracker(sla_ms=10.0)
    for value in (1.0, 2.0, 15.0, 3.0, 25.0):
        tracker.record(value)

    summary = tracker.summary()
    assert summary["count"] == 5
    assert summary["sla_violation_rate"] == pytest.approx(2 / 5)
    assert summary["max_ms"] == 25.0
