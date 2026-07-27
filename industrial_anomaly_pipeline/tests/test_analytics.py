"""
Analytics engine tests, including the Phase 3 bug-audit regressions:
  - Dimension-mismatch guards in DMD/SVD and the autoencoder (missing
    sensor channels, variable window length for the Dense architecture).
  - NaN/missing-channel imputation rather than propagation.
  - PyTorch inference runs under torch.no_grad() and is deterministic
    (i.e. dropout is genuinely disabled at inference, not leaking
    train-mode stochasticity/graph-building into the streaming hot path).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from analytics.dmd_decomposer import DMDSTLDetector
from analytics.autoencoder import AutoencoderDetector


def _windows(rng: np.random.Generator, n: int, window_size: int, n_features: int) -> np.ndarray:
    return rng.normal(size=(n, window_size, n_features))


# --------------------------------------------------------------------------- #
# DMD + STL fast-track engine
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def fitted_dmd() -> DMDSTLDetector:
    rng = np.random.default_rng(0)
    train = _windows(rng, 60, window_size=20, n_features=5)
    det = DMDSTLDetector(stl_period=20, dmd_rank=4)
    det.fit(train)
    return det


def test_dmd_fits_quickly_and_scores_below_sla(fitted_dmd: DMDSTLDetector):
    assert fitted_dmd.fit_time_sec is not None
    assert fitted_dmd.fit_time_sec < 60  # far under the "<20 min init" target

    rng = np.random.default_rng(1)
    window = rng.normal(size=(20, 5))
    result = fitted_dmd.score_window(window)
    assert result.latency_ms < 20.0  # SLA
    assert "stl_residual_energy" in result.components
    assert "dmd_prediction_error" in result.components


def test_dmd_raises_clear_error_on_channel_mismatch_instead_of_crashing(fitted_dmd: DMDSTLDetector):
    """Bug audit #2: dimension mismatch (e.g. a dropped sensor channel) must
    fail predictably, not with an opaque numpy broadcast/shape error."""
    with pytest.raises(ValueError, match="sensor channel"):
        fitted_dmd.score_window(np.zeros((20, 3)))


def test_dmd_rejects_too_short_window(fitted_dmd: DMDSTLDetector):
    with pytest.raises(ValueError, match="window_size"):
        fitted_dmd.score_window(np.zeros((1, 5)))


def test_dmd_fit_rejects_wrong_ndim():
    det = DMDSTLDetector()
    with pytest.raises(ValueError):
        det.fit(np.zeros((10, 5)))  # missing the window dimension


def test_dmd_imputes_nan_channel_instead_of_propagating(fitted_dmd: DMDSTLDetector):
    """Bug audit #2/#5: a missing sensor reading (NaN) inside an
    otherwise-valid window must not poison the whole score with NaN."""
    rng = np.random.default_rng(2)
    window = rng.normal(size=(20, 5))
    window[3, 2] = np.nan

    result = fitted_dmd.score_window(window)
    assert np.isfinite(result.score)
    assert all(np.isfinite(v) for v in result.components.values())


def test_dmd_fit_imputes_nan_in_training_data():
    rng = np.random.default_rng(3)
    train = _windows(rng, 40, window_size=15, n_features=4)
    train[5, 2, 1] = np.nan

    det = DMDSTLDetector(stl_period=15, dmd_rank=3)
    det.fit(train)  # must not raise, and downstream operator must stay finite
    assert np.isfinite(det._dmd.A).all()


def test_dmd_handles_zero_variance_channel_without_error():
    """A constant sensor channel drives some SVD singular values to ~0;
    the epsilon-guarded inverse in _fit_dmd must not blow up to inf/NaN."""
    rng = np.random.default_rng(4)
    train = _windows(rng, 30, window_size=15, n_features=4)
    train[:, :, 0] = 5.0  # constant channel

    det = DMDSTLDetector(stl_period=15, dmd_rank=4)
    det.fit(train)
    assert np.isfinite(det._dmd.A).all()

    result = det.score_window(train[0])
    assert np.isfinite(result.score)


# --------------------------------------------------------------------------- #
# Autoencoder engine
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def fitted_ae() -> AutoencoderDetector:
    rng = np.random.default_rng(0)
    train = _windows(rng, 80, window_size=10, n_features=4)
    det = AutoencoderDetector(architecture="lstm", epochs=3, hidden_dim=8, latent_dim=4)
    det.fit(train)
    return det


def test_autoencoder_trains_and_scores_below_sla(fitted_ae: AutoencoderDetector):
    rng = np.random.default_rng(5)
    window = rng.normal(size=(10, 4)).astype(np.float32)
    result = fitted_ae.score_window(window)
    assert result.latency_ms < 20.0
    assert "reconstruction_error" in result.components


def test_autoencoder_inference_runs_without_building_autograd_graph(fitted_ae: AutoencoderDetector):
    """Bug audit #3: every _raw_score forward pass must run under
    torch.no_grad(); otherwise each streamed window retains a full
    autograd graph it will never backward() through -- wasted compute and
    a growing memory footprint under sustained throughput."""
    grad_states: list[bool] = []
    original_forward = fitted_ae.model.forward

    def spy_forward(x):
        grad_states.append(torch.is_grad_enabled())
        return original_forward(x)

    fitted_ae.model.forward = spy_forward
    try:
        rng = np.random.default_rng(6)
        fitted_ae.score_window(rng.normal(size=(10, 4)).astype(np.float32))
    finally:
        fitted_ae.model.forward = original_forward

    assert grad_states == [False]


def test_autoencoder_inference_is_deterministic_across_repeated_calls(fitted_ae: AutoencoderDetector):
    """If dropout/train-mode ever leaked into the inference path, repeated
    scoring of the identical window would not be bit-identical."""
    rng = np.random.default_rng(7)
    window = rng.normal(size=(10, 4)).astype(np.float32)
    first = fitted_ae.score_window(window).score
    second = fitted_ae.score_window(window).score
    assert first == second


def test_autoencoder_raises_clear_error_on_channel_mismatch(fitted_ae: AutoencoderDetector):
    with pytest.raises(ValueError, match="sensor channel"):
        fitted_ae.score_window(np.zeros((10, 2), dtype=np.float32))


def test_dense_autoencoder_rejects_variable_window_length():
    """Bug audit #2: the Dense architecture flattens window_size * n_features
    at fit time, so a variable-length window at inference must fail
    predictably rather than crash inside a reshape()."""
    rng = np.random.default_rng(8)
    train = _windows(rng, 40, window_size=12, n_features=3)
    det = AutoencoderDetector(architecture="dense", epochs=2, hidden_dim=8, latent_dim=4)
    det.fit(train)

    with pytest.raises(ValueError, match="window_size"):
        det.score_window(np.zeros((8, 3), dtype=np.float32))


def test_lstm_autoencoder_tolerates_variable_window_length():
    """Unlike Dense, the LSTM architecture has no fixed sequence length."""
    rng = np.random.default_rng(9)
    train = _windows(rng, 40, window_size=12, n_features=3)
    det = AutoencoderDetector(architecture="lstm", epochs=2, hidden_dim=8, latent_dim=4)
    det.fit(train)

    result = det.score_window(rng.normal(size=(8, 3)).astype(np.float32))
    assert np.isfinite(result.score)


def test_autoencoder_imputes_nan_channel_instead_of_propagating(fitted_ae: AutoencoderDetector):
    rng = np.random.default_rng(10)
    window = rng.normal(size=(10, 4)).astype(np.float32)
    window[2, 1] = np.nan

    result = fitted_ae.score_window(window)
    assert np.isfinite(result.score)


def test_autoencoder_fit_rejects_wrong_ndim():
    det = AutoencoderDetector(architecture="lstm", epochs=1)
    with pytest.raises(ValueError):
        det.fit(np.zeros((10, 4)))


def test_autoencoder_requires_fit_before_scoring():
    det = AutoencoderDetector(architecture="lstm", epochs=1)
    with pytest.raises(RuntimeError):
        det.score_window(np.zeros((10, 4), dtype=np.float32))


def test_quantize_dynamic_preserves_scoring_capability():
    # Uses its own detector instance (rather than the shared module-scoped
    # fixture) since quantization mutates `.model` in place.
    rng = np.random.default_rng(11)
    train = _windows(rng, 60, window_size=10, n_features=4)
    det = AutoencoderDetector(architecture="lstm", epochs=2, hidden_dim=8, latent_dim=4)
    det.fit(train)

    window = rng.normal(size=(10, 4)).astype(np.float32)
    det.quantize_dynamic()
    result = det.score_window(window)
    assert np.isfinite(result.score)
    assert det.get_params()["quantized"] is True
