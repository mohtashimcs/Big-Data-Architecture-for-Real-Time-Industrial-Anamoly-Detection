"""
Evaluation Matrix & Anomaly Engine.

1. Composite Scoring Function -- fuses two independent normal-cy signals
   per window into one anomaly score via the (epsilon-guarded) **harmonic
   mean**:
     - a Euclidean-distance term: how far the window's sensor vectors sit
       from the nearest known-normal operating-regime centroid.
     - a localized-reconstruction-error term: whatever per-window error an
       analytics engine already produced (`AnomalyResult.score` from either
       `DMDSTLDetector` or `AutoencoderDetector` -- this module is engine
       agnostic and only consumes a scalar).
   The harmonic mean is used rather than the arithmetic mean because it is
   dominated by the smaller of the two inputs: a window only scores low if
   it is *both* close to a normal regime *and* reconstructs/predicts well,
   so neither signal alone can mask a genuine anomaly.

2. Dynamic Thresholding -- a rolling, robust baseline of recent composite
   scores (deliberately excluding scores already flagged anomalous, so the
   boundary can't be dragged upward by a sustained anomaly run) yields a
   threshold that tracks the stream's *local* deviation, not a single
   static global cutoff.

3. Classification metrics (F1 / AUC-ROC / Precision / Recall) against
   historical ground-truth labels, for offline benchmarking.
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

logger = logging.getLogger(__name__)

_EPS = 1e-8


# --------------------------------------------------------------------------- #
# Reference centroids (normal operating regimes)
# --------------------------------------------------------------------------- #
@dataclass
class ReferenceCentroids:
    centroids: np.ndarray  # (k, n_features)


def fit_reference_centroids(
    normal_windows: np.ndarray, k: int = 8, seed: int = 42, n_iter: int = 20
) -> ReferenceCentroids:
    """Lightweight k-means++ (pure NumPy) over per-timestep normal vectors,
    giving k representative "normal operating regime" centroids."""
    flat = normal_windows.reshape(-1, normal_windows.shape[-1])
    flat = flat[~np.isnan(flat).any(axis=1)]
    if len(flat) == 0:
        raise ValueError("fit_reference_centroids: normal_windows contained no valid (non-NaN) rows")
    k = max(1, min(k, len(flat)))  # guard: can't have more clusters than samples

    rng = np.random.default_rng(seed)
    idx = [int(rng.integers(len(flat)))]
    for _ in range(k - 1):
        d2 = np.min([np.sum((flat - flat[i]) ** 2, axis=1) for i in idx], axis=0)
        total = d2.sum()
        probs = d2 / total if total > _EPS else np.full(len(flat), 1.0 / len(flat))
        idx.append(int(rng.choice(len(flat), p=probs)))
    centroids = flat[idx].copy()

    for _ in range(n_iter):
        dists = np.linalg.norm(flat[:, None, :] - centroids[None, :, :], axis=2)
        assign = np.argmin(dists, axis=1)
        new_centroids = np.array(
            [flat[assign == j].mean(axis=0) if np.any(assign == j) else centroids[j]
             for j in range(k)]
        )
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids

    return ReferenceCentroids(centroids=centroids)


def nearest_centroid_distance(window: np.ndarray, ref: ReferenceCentroids) -> float:
    """Mean (over the window's timesteps) Euclidean distance to the nearest
    reference centroid -- a "how far from any known-normal regime" score."""
    window = np.asarray(window, dtype=np.float64)
    if np.isnan(window).any():
        logger.debug("nearest_centroid_distance: NaN in window, imputing with column means")
        col_mean = np.nanmean(window, axis=0)
        window = np.where(np.isnan(window), col_mean, window)
    dists = np.linalg.norm(window[:, None, :] - ref.centroids[None, :, :], axis=2)  # (T, k)
    return float(np.mean(dists.min(axis=1)))


# --------------------------------------------------------------------------- #
# Composite scoring: harmonic mean of distance + localized error
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CompositeScore:
    distance: float
    local_error: float
    composite: float
    is_anomaly: bool
    threshold: float


def harmonic_composite(distance: float, local_error: float, eps: float = _EPS) -> float:
    """
    Epsilon-guarded harmonic mean of two non-negative scalars.

    2ab/(a+b) is undefined at a=b=0 (both signals say "perfectly normal"):
    the +eps in the denominator turns that into a safe 0/eps = 0.0 instead
    of a ZeroDivisionError / NaN -- the fix for the "division by zero on a
    zero-variance baseline window" failure mode.
    """
    a, b = max(distance, 0.0), max(local_error, 0.0)
    return float(2.0 * a * b / (a + b + eps))


class DynamicThresholder:
    """
    Rolling, robust anomaly threshold: mean + k*std (or a percentile) over a
    bounded window of the *most recent normal-looking* composite scores.
    Deliberately does not fold already-flagged scores back into the
    baseline, so a sustained anomaly run can't drag the boundary up and
    mask itself ("threshold drift").
    """

    def __init__(
        self,
        lookback: int = 500,
        k_sigma: float = 3.0,
        method: str = "zscore",
        percentile: float = 99.5,
        min_history: int = 30,
        eps: float = _EPS,
        stuck_run_length: int = 50,
    ) -> None:
        if method not in ("zscore", "percentile"):
            raise ValueError(f"method must be 'zscore' or 'percentile', got {method!r}")
        self.lookback = lookback
        self.k_sigma = k_sigma
        self.method = method
        self.percentile = percentile
        self.min_history = min_history
        self.eps = eps
        # A real anomaly is episodic; `stuck_run_length` consecutive flags in a
        # row is far more consistent with a *stale baseline* (the stream's
        # normal operating point has drifted since calibration) than with one
        # sustained anomaly. Past this run length, force a recalibration --
        # otherwise the "don't let anomalies drag the boundary" rule below
        # becomes self-defeating: if drift makes every genuinely-normal score
        # score above threshold, none of them ever update the baseline, and
        # the threshold stays wrong forever.
        self.stuck_run_length = stuck_run_length
        self._history: deque[float] = deque(maxlen=lookback)
        self._recent_scores: deque[float] = deque(maxlen=stuck_run_length)
        self._consecutive_flags = 0

    def warm_start(self, scores: np.ndarray) -> None:
        """Seed the rolling baseline directly from a known-normal validation split."""
        for s in scores:
            self._history.append(float(s))

    def current_threshold(self) -> float:
        if len(self._history) < self.min_history:
            return float("inf")  # not enough baseline yet -> never flag
        arr = np.asarray(self._history, dtype=np.float64)
        if self.method == "percentile":
            return float(np.percentile(arr, self.percentile))
        mean = float(arr.mean())
        std = float(arr.std())
        std_safe = std if std > self.eps else self.eps  # zero-variance baseline guard
        return mean + self.k_sigma * std_safe

    def evaluate(self, score: float) -> tuple[bool, float]:
        threshold = self.current_threshold()
        is_anomaly = score > threshold
        self._recent_scores.append(score)  # tracked regardless of flag, for recalibration below

        if not is_anomaly:
            self._history.append(score)
            self._consecutive_flags = 0
        else:
            self._consecutive_flags += 1
            if self._consecutive_flags >= self.stuck_run_length:
                # Replace the stale baseline outright with the recent run: a
                # single drip-fed sample would barely nudge a large rolling
                # mean/std, so a full swap is what actually lets the
                # threshold catch up to a genuinely shifted regime.
                logger.info(
                    "DynamicThresholder: %d consecutive flags, forcing baseline "
                    "recalibration (stale threshold vs. a genuine sustained anomaly "
                    "are indistinguishable without this)",
                    self._consecutive_flags,
                )
                self._history.clear()
                self._history.extend(self._recent_scores)
                self._consecutive_flags = 0
        return is_anomaly, threshold


class CompositeScoringEngine:
    """Ties reference centroids + harmonic composite + dynamic thresholding together."""

    def __init__(self, k_centroids: int = 8, thresholder: DynamicThresholder | None = None) -> None:
        self.k_centroids = k_centroids
        self.thresholder = thresholder or DynamicThresholder()
        self._ref: ReferenceCentroids | None = None

    @property
    def is_fitted(self) -> bool:
        return self._ref is not None

    def fit(self, normal_windows: np.ndarray, warm_start_errors: np.ndarray | None = None) -> "CompositeScoringEngine":
        self._ref = fit_reference_centroids(normal_windows, k=self.k_centroids)
        if warm_start_errors is not None:
            distances = np.array([nearest_centroid_distance(w, self._ref) for w in normal_windows])
            n = min(len(distances), len(warm_start_errors))
            warm_scores = [
                harmonic_composite(distances[i], warm_start_errors[i]) for i in range(n)
            ]
            self.thresholder.warm_start(np.array(warm_scores))
        return self

    def score(self, window: np.ndarray, local_error: float) -> CompositeScore:
        if not self.is_fitted:
            raise RuntimeError("CompositeScoringEngine must be fit() before scoring")
        distance = nearest_centroid_distance(window, self._ref)
        composite = harmonic_composite(distance, local_error)
        is_anomaly, threshold = self.thresholder.evaluate(composite)
        return CompositeScore(
            distance=distance, local_error=local_error, composite=composite,
            is_anomaly=is_anomaly, threshold=threshold,
        )


# --------------------------------------------------------------------------- #
# Offline classification metrics
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ClassificationMetrics:
    f1: float
    auc_roc: float
    precision: float
    recall: float

    def as_dict(self) -> dict[str, float]:
        return {"f1": self.f1, "auc_roc": self.auc_roc, "precision": self.precision, "recall": self.recall}


def compute_classification_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_pred: np.ndarray | None = None,
    threshold: float | None = None,
) -> ClassificationMetrics:
    """F1 / AUC-ROC / Precision / Recall against ground-truth labels.

    Either pass `y_pred` directly (e.g. from streaming `is_anomaly` flags) or
    a `threshold` to derive predictions from `y_score`. AUC-ROC is undefined
    for a single-class label set (e.g. an all-normal calibration slice) and
    is reported as NaN in that case rather than raising.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=np.float64)
    if y_pred is None:
        if threshold is None:
            raise ValueError("compute_classification_metrics: pass either y_pred or threshold")
        y_pred = (y_score > threshold).astype(int)
    else:
        y_pred = np.asarray(y_pred).astype(int)

    auc = float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else float("nan")
    return ClassificationMetrics(
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        auc_roc=auc,
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
    )
