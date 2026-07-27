"""
Engine 1: Mathematical Fast-Track (VersaGuardian-inspired).

Two closed-form ingredients, both O(window_size * n_features^2) or cheaper
at inference time (a handful of matrix multiplies + one small SVD at fit
time only), which is what lets this engine clear the sub-20ms SLA even on
a single CPU core:

  1. Seasonal-Trend decomposition (STL), fit once on the normal baseline to
     learn the expected per-channel residual-energy distribution. Each
     incoming window is decomposed independently at scoring time and
     compared to that baseline -- a streaming approximation of continuous
     STL without ever needing to refit a global decomposition.
  2. Dynamic Mode Decomposition (DMD): a reduced-order linear operator
     A (n_features x n_features), fit via SVD-truncated least squares over
     paired snapshots (x_t -> x_{t+1}) from the normal baseline. Large
     one-step prediction error from A flags anomalous dynamics.

No gradient descent anywhere in `fit()`, so initialization is bounded by a
couple of matrix factorizations rather than an optimizer loop -- the
"<20 min init" target in practice is single-digit seconds even on large
baselines.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

from analytics.base import BaseAnomalyDetector

logger = logging.getLogger(__name__)

try:
    from statsmodels.tsa.seasonal import STL

    _HAS_STATSMODELS = True
except ImportError:  # statsmodels optional at import time; see requirements.txt
    _HAS_STATSMODELS = False


# --------------------------------------------------------------------------- #
# Seasonal-Trend Decomposition (STL)
# --------------------------------------------------------------------------- #
@dataclass
class STLBaseline:
    period: int
    residual_std: np.ndarray  # per-channel std of residuals on normal data

    def residual_energy(self, window: np.ndarray) -> float:
        """window: (window_size, n_features), already shape/NaN-validated by the caller."""
        n_features = window.shape[1]
        energies = np.empty(n_features)
        for f in range(n_features):
            resid = _stl_residual(window[:, f], self.period)
            std = self.residual_std[f] if self.residual_std[f] > 1e-8 else 1e-8
            energies[f] = np.mean((resid / std) ** 2)
        return float(np.mean(energies))


def _stl_residual(series: np.ndarray, period: int) -> np.ndarray:
    if _HAS_STATSMODELS and len(series) >= 2 * period:
        try:
            return STL(series, period=period, robust=True).fit().resid
        except Exception:
            logger.debug("STL fit failed on a %d-length series; using moving-average fallback",
                         len(series))
    # Fallback: cheap moving-average detrend/deseasonalize when STL can't run
    # (short windows, or statsmodels unavailable) -- keeps this engine
    # fast-initializing and dependency-light, true to its "fast-track" role.
    trend = _moving_average(series, max(3, period // 3))
    detrended = series - trend
    return detrended - detrended.mean()


def _moving_average(series: np.ndarray, k: int) -> np.ndarray:
    k = max(1, min(k, len(series)))
    kernel = np.ones(k) / k
    return np.convolve(series, kernel, mode="same")


def _fit_stl_baseline(normal_windows: np.ndarray, period: int) -> STLBaseline:
    n_features = normal_windows.shape[2]
    sample = normal_windows[:: max(1, len(normal_windows) // 200)]  # cap fit cost
    residual_std = np.empty(n_features)
    for f in range(n_features):
        resids = [_stl_residual(w[:, f], period) for w in sample]
        residual_std[f] = np.std(np.concatenate(resids)) if resids else 1.0
    return STLBaseline(period=period, residual_std=residual_std)


# --------------------------------------------------------------------------- #
# Dynamic Mode Decomposition (DMD)
# --------------------------------------------------------------------------- #
@dataclass
class DMDOperator:
    """Linear operator A (n_features x n_features) such that x_{t+1} ~= A x_t."""

    A: np.ndarray
    prediction_error_std: float


def _fit_dmd(normal_windows: np.ndarray, rank: int) -> DMDOperator:
    """
    Closed-form DMD fit: stack paired snapshots X (t) -> X' (t+1) across all
    normal windows (columns = time-steps, rows = sensor channels), truncate
    via SVD to `rank`, solve for the operator in the reduced subspace, then
    lift back to full sensor space.
    """
    X = np.concatenate([w[:-1].T for w in normal_windows], axis=1)
    Xp = np.concatenate([w[1:].T for w in normal_windows], axis=1)

    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    r = max(1, min(rank, U.shape[1]))
    Ur, Sr, Vr = U[:, :r], S[:r], Vt[:r, :].T

    # epsilon-guarded inverse: near-zero singular values (constant/collinear
    # sensor channels in the baseline) would otherwise blow up 1/Sr.
    Sr_inv = np.diag(1.0 / np.where(Sr > 1e-10, Sr, 1e-10))
    A_tilde = Ur.T @ Xp @ Vr @ Sr_inv  # reduced operator (r x r)
    A = Ur @ A_tilde @ Ur.T  # lift back to full space

    preds = A @ X
    errs = np.linalg.norm(Xp - preds, axis=0)
    return DMDOperator(A=A, prediction_error_std=float(np.std(errs)) or 1.0)


def _dmd_anomaly_score(op: DMDOperator, window: np.ndarray) -> float:
    """One-step-ahead prediction error, normalized by the baseline error std.
    Cheap: a single (n_features x n_features) @ (n_features x T) matmul."""
    X, Xp = window[:-1].T, window[1:].T
    preds = op.A @ X
    err = np.linalg.norm(Xp - preds, axis=0).mean()
    return float(err / op.prediction_error_std)


# --------------------------------------------------------------------------- #
# BaseAnomalyDetector adapter
# --------------------------------------------------------------------------- #
class DMDSTLDetector(BaseAnomalyDetector):
    """
    Fast mathematical-track engine: combines STL residual energy and DMD
    one-step prediction error (both already baseline-normalized to ~O(1)
    for normal data) into a single lightweight anomaly score.
    """

    def __init__(
        self,
        stl_period: int = 30,
        dmd_rank: int = 10,
        stl_weight: float = 0.5,
        name: str = "dmd-stl-fast-track",
    ) -> None:
        super().__init__(name=name)
        self.stl_period = stl_period
        self.dmd_rank = dmd_rank
        self.stl_weight = stl_weight
        self._stl: STLBaseline | None = None
        self._dmd: DMDOperator | None = None
        self._n_features: int | None = None
        self._channel_mean: np.ndarray | None = None
        self.fit_time_sec: float | None = None

    def fit(self, normal_windows: np.ndarray) -> "DMDSTLDetector":
        normal_windows = np.asarray(normal_windows, dtype=np.float64)
        if normal_windows.ndim != 3:
            raise ValueError(
                f"{self.name}: fit() expects (n_windows, window_size, n_features), "
                f"got shape {normal_windows.shape}"
            )
        if normal_windows.shape[1] < 2:
            raise ValueError(
                f"{self.name}: window_size must be >= 2 for one-step DMD differencing, "
                f"got {normal_windows.shape[1]}"
            )
        if np.isnan(normal_windows).any():
            # Fill NaNs with the per-channel mean over all *observed* values before
            # any statistic (residual std, DMD operator) is estimated from them.
            flat = normal_windows.reshape(-1, normal_windows.shape[-1])
            col_mean = np.nanmean(flat, axis=0)
            nan_mask = np.isnan(normal_windows)
            normal_windows = np.where(nan_mask, col_mean, normal_windows)
            logger.warning(
                "%s: fit() input contained NaNs; imputed %d cell(s) with per-channel means",
                self.name, int(nan_mask.sum()),
            )

        start = time.perf_counter()
        self._n_features = normal_windows.shape[2]
        self._channel_mean = normal_windows.reshape(-1, self._n_features).mean(axis=0)
        self._stl = _fit_stl_baseline(normal_windows, period=self.stl_period)
        self._dmd = _fit_dmd(normal_windows, rank=self.dmd_rank)
        self.fit_time_sec = time.perf_counter() - start
        self._is_fitted = True
        logger.info(
            "%s: fit on %d windows in %.3fs (n_features=%d)",
            self.name, len(normal_windows), self.fit_time_sec, self._n_features,
        )
        return self

    def _prepare_window(self, window: np.ndarray) -> np.ndarray:
        """Dimension/NaN guard: fail predictably on shape mismatch rather than
        letting a bare numpy broadcast error surface deep inside SVD/matmul,
        and impute rather than propagate NaN from a dropped sensor channel."""
        window = np.asarray(window, dtype=np.float64)
        if window.ndim != 2:
            raise ValueError(
                f"{self.name}: expected a 2D window (window_size, n_features), "
                f"got shape {window.shape}"
            )
        if window.shape[0] < 2:
            raise ValueError(
                f"{self.name}: window_size must be >= 2 for one-step DMD differencing, "
                f"got {window.shape[0]}"
            )
        if window.shape[1] != self._n_features:
            raise ValueError(
                f"{self.name}: window has {window.shape[1]} sensor channel(s) but this "
                f"engine was fit on {self._n_features}; upstream ingestion must emit a "
                "fixed sensor schema, or realign/impute channels before scoring."
            )
        if np.isnan(window).any():
            window = window.copy()
            rows, cols = np.where(np.isnan(window))
            window[rows, cols] = self._channel_mean[cols]
            logger.debug(
                "%s: imputed %d missing-channel value(s) in-window using fitted means",
                self.name, len(rows),
            )
        return window

    def _raw_score(self, window: np.ndarray) -> tuple[float, dict[str, float]]:
        window = self._prepare_window(window)
        stl_energy = self._stl.residual_energy(window)
        dmd_error = _dmd_anomaly_score(self._dmd, window)
        combined = self.stl_weight * stl_energy + (1 - self.stl_weight) * dmd_error
        return combined, {
            "stl_residual_energy": stl_energy,
            "dmd_prediction_error": dmd_error,
        }

    def get_params(self) -> dict:
        params = super().get_params()
        params.update(
            stl_period=self.stl_period,
            dmd_rank=self.dmd_rank,
            stl_weight=self.stl_weight,
            fit_time_sec=self.fit_time_sec,
            n_features=self._n_features,
        )
        return params
