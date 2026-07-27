"""
Strategy-pattern interface shared by every anomaly-detection engine in the
pipeline: the mathematical fast-track (STL + DMD) and the deep
reconstruction model (LSTM / Dense Autoencoder). Ingestion, evaluation, and
benchmarking code depend only on `BaseAnomalyDetector`, never on a concrete
engine, so new engines can be dropped in without touching the rest of the
system.
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnomalyResult:
    """Outcome of scoring a single multivariate-time-series window."""

    score: float
    is_anomaly: bool
    latency_ms: float
    components: dict[str, float] = field(default_factory=dict)


class BaseAnomalyDetector(ABC):
    """
    Common interface for all detection engines.

    Concrete engines implement `fit` (train exclusively on normal-operation
    windows) and `_raw_score` (the engine-specific anomaly score for one
    window). `score_window` wraps `_raw_score` with latency instrumentation
    and threshold comparison, since per-window latency against the 20ms SLA
    is a first-class metric for this system, not an afterthought.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.threshold: float = float("inf")
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @abstractmethod
    def fit(self, normal_windows: np.ndarray) -> "BaseAnomalyDetector":
        """
        Fit the engine exclusively on normal-operation data.

        normal_windows: array of shape (n_windows, window_size, n_features).
        Must set self._is_fitted = True and return self.
        """
        raise NotImplementedError

    @abstractmethod
    def _raw_score(self, window: np.ndarray) -> tuple[float, dict[str, float]]:
        """
        Engine-specific anomaly score for a single window.

        window: array of shape (window_size, n_features).
        Returns (score, component_breakdown) where component_breakdown
        holds any intermediate values (e.g. reconstruction_error,
        dmd_prediction_error) useful for downstream composite scoring.
        """
        raise NotImplementedError

    def score_window(self, window: np.ndarray) -> AnomalyResult:
        """
        Latency-instrumented single-window scoring — the hot path that must
        clear the sub-20ms SLA in real-time streaming use.
        """
        if not self._is_fitted:
            raise RuntimeError(f"{self.name} must be fit() before scoring")

        start = time.perf_counter()
        score, components = self._raw_score(window)
        latency_ms = (time.perf_counter() - start) * 1000.0

        return AnomalyResult(
            score=score,
            is_anomaly=score > self.threshold,
            latency_ms=latency_ms,
            components=components,
        )

    def score_batch(self, windows: np.ndarray) -> list[AnomalyResult]:
        """Convenience wrapper for offline evaluation; the hot path remains score_window."""
        return [self.score_window(window) for window in windows]

    def set_threshold(self, threshold: float) -> None:
        self.threshold = threshold
        logger.info("%s: threshold set to %.6f", self.name, threshold)

    def get_params(self) -> dict[str, Any]:
        """Hyperparameters/metadata worth logging alongside benchmark results."""
        return {"name": self.name, "threshold": self.threshold}

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"{self.__class__.__name__}(name={self.name!r}, fitted={self._is_fitted})"
