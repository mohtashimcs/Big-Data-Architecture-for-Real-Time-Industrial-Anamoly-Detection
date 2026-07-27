"""
Evaluation engine tests, including the Phase 3 bug-audit regression for
division-by-zero in the harmonic-mean composite score / dynamic threshold.
"""
from __future__ import annotations

import numpy as np
import pytest

from evaluation.metrics_engine import (
    CompositeScoringEngine,
    DynamicThresholder,
    ReferenceCentroids,
    compute_classification_metrics,
    fit_reference_centroids,
    harmonic_composite,
    nearest_centroid_distance,
)


# --------------------------------------------------------------------------- #
# Harmonic composite score
# --------------------------------------------------------------------------- #
def test_harmonic_composite_zero_variance_window_does_not_divide_by_zero():
    """Bug audit #4: both terms exactly 0 (a perfectly-explained window --
    zero distance to a centroid, zero reconstruction error) is the classic
    a+b=0 harmonic-mean singularity. Must return 0.0, not NaN/inf/raise."""
    score = harmonic_composite(0.0, 0.0)
    assert score == 0.0
    assert np.isfinite(score)


def test_harmonic_composite_dominated_by_smaller_term():
    assert harmonic_composite(0.0, 5.0) == 0.0
    small = harmonic_composite(0.1, 100.0)
    assert small < 1.0  # harmonic mean stays near the smaller input


def test_harmonic_composite_equal_terms_returns_that_value():
    assert harmonic_composite(4.0, 4.0) == pytest.approx(4.0, rel=1e-6)


def test_harmonic_composite_negative_inputs_are_clamped_not_erroring():
    # Defensive: a caller passing a stray negative value should not corrupt
    # the harmonic mean or divide by a negative denominator.
    score = harmonic_composite(-1.0, -1.0)
    assert score == 0.0


# --------------------------------------------------------------------------- #
# Dynamic thresholding
# --------------------------------------------------------------------------- #
def test_dynamic_thresholder_zero_variance_baseline_does_not_divide_by_zero():
    """Bug audit #4: a constant baseline (std == 0) must not make the
    z-score threshold blow up to inf/NaN; the eps floor on std keeps the
    threshold a small, finite offset above the constant baseline value."""
    thresholder = DynamicThresholder(min_history=5, lookback=50, k_sigma=3.0)
    thresholder.warm_start(np.full(20, 2.5))

    threshold = thresholder.current_threshold()
    assert np.isfinite(threshold)
    assert threshold > 2.5

    is_anomaly, _ = thresholder.evaluate(2.5)
    assert is_anomaly is False
    is_anomaly, _ = thresholder.evaluate(1000.0)
    assert is_anomaly is True


def test_dynamic_thresholder_returns_infinity_before_enough_history():
    thresholder = DynamicThresholder(min_history=10)
    thresholder.warm_start(np.ones(3))
    assert thresholder.current_threshold() == float("inf")


def test_dynamic_thresholder_does_not_let_anomalies_drag_baseline_up():
    thresholder = DynamicThresholder(min_history=5, lookback=50, k_sigma=3.0)
    thresholder.warm_start(np.ones(20))
    baseline_threshold = thresholder.current_threshold()

    for _ in range(10):
        thresholder.evaluate(500.0)  # sustained anomaly run

    assert thresholder.current_threshold() == pytest.approx(baseline_threshold, rel=1e-6)


def test_dynamic_thresholder_rejects_unknown_method():
    with pytest.raises(ValueError):
        DynamicThresholder(method="bogus")


def test_dynamic_thresholder_recovers_from_sustained_baseline_drift():
    """A stale threshold that every post-drift 'normal' score exceeds would,
    without recovery, never receive a baseline update again (the
    not-anomalous-only update rule is self-defeating under real drift, not
    just under a genuine attack). After `stuck_run_length` consecutive
    flags the thresholder must force a recalibration and start passing
    the new, drifted-but-normal regime again."""
    thresholder = DynamicThresholder(min_history=5, lookback=200, k_sigma=3.0, stuck_run_length=20)
    thresholder.warm_start(np.full(30, 1.0))  # calibrated at the old regime

    # New regime settles at ~10.0 -- every one of these would otherwise be
    # flagged forever since none pass the stale ~1.0-based threshold.
    flags = [thresholder.evaluate(10.0)[0] for _ in range(20)]
    assert all(flags)  # stuck: the whole run is (wrongly) flagged first

    # Recalibration should have kicked in by now; scores at the new regime
    # should pass again.
    post_recovery_flags = [thresholder.evaluate(10.0)[0] for _ in range(5)]
    assert not any(post_recovery_flags)


# --------------------------------------------------------------------------- #
# Reference centroids
# --------------------------------------------------------------------------- #
def test_fit_reference_centroids_handles_k_greater_than_sample_count():
    rng = np.random.default_rng(0)
    tiny = rng.normal(size=(3, 4, 2))  # 12 total timesteps
    ref = fit_reference_centroids(tiny, k=100)
    assert ref.centroids.shape[0] <= 12


def test_fit_reference_centroids_raises_on_all_nan_input():
    all_nan = np.full((5, 4, 2), np.nan)
    with pytest.raises(ValueError):
        fit_reference_centroids(all_nan, k=3)


def test_nearest_centroid_distance_imputes_nan_window():
    ref = ReferenceCentroids(centroids=np.zeros((3, 2)))
    window = np.array([[1.0, np.nan], [2.0, 2.0]])
    distance = nearest_centroid_distance(window, ref)
    assert np.isfinite(distance)


# --------------------------------------------------------------------------- #
# Full CompositeScoringEngine, end-to-end
# --------------------------------------------------------------------------- #
def test_composite_scoring_engine_flags_injected_anomalies():
    rng = np.random.default_rng(1)
    normal_windows = rng.normal(size=(200, 10, 4))
    warm_errors = np.abs(rng.normal(0.5, 0.05, size=200))

    engine = CompositeScoringEngine(k_centroids=6)
    engine.fit(normal_windows, warm_start_errors=warm_errors)

    test_windows = rng.normal(size=(60, 10, 4))
    errors = np.abs(rng.normal(0.5, 0.05, size=60))
    labels = np.zeros(60, dtype=int)

    test_windows[45:50] += 12
    errors[45:50] += 15
    labels[45:50] = 1

    scores, preds = [], []
    for window, error in zip(test_windows, errors):
        result = engine.score(window, error)
        scores.append(result.composite)
        preds.append(int(result.is_anomaly))

    metrics = compute_classification_metrics(labels, np.array(scores), y_pred=np.array(preds))
    assert metrics.recall >= 0.8  # the injected block should be caught
    assert metrics.f1 > 0.5


def test_composite_scoring_engine_requires_fit_before_scoring():
    engine = CompositeScoringEngine()
    with pytest.raises(RuntimeError):
        engine.score(np.zeros((5, 3)), 0.1)


# --------------------------------------------------------------------------- #
# Classification metrics
# --------------------------------------------------------------------------- #
def test_compute_classification_metrics_single_class_auc_is_nan_not_error():
    y_true = np.zeros(10, dtype=int)
    y_score = np.random.default_rng(0).normal(size=10)
    metrics = compute_classification_metrics(y_true, y_score, threshold=0.0)
    assert np.isnan(metrics.auc_roc)
    assert np.isfinite(metrics.f1)


def test_compute_classification_metrics_perfect_separation():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.1, 0.9, 0.8, 0.95])
    metrics = compute_classification_metrics(y_true, y_score, threshold=0.5)
    assert metrics.f1 == 1.0
    assert metrics.auc_roc == 1.0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0


def test_compute_classification_metrics_requires_pred_or_threshold():
    with pytest.raises(ValueError):
        compute_classification_metrics(np.array([0, 1]), np.array([0.1, 0.9]))
