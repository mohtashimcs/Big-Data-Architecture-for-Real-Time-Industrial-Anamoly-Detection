"""
Phase 2 / Step 5: the fast mathematical backend (VersaGuardian-style).

A low-overhead, fast-initializing alternative to the deep Autoencoder, built
from two closed-form components fit once on normal data:

  1. Seasonal-Trend decomposition (STL) per sensor channel, learning the
     residual scale of healthy operation. At scoring time, a window's
     normalized residual energy indicates deviation from the expected
     trend/seasonal signature.
  2. Dynamic Mode Decomposition (DMD): a linear operator A with
     x_{t+1} ~= A x_t, obtained from a single rank-truncated SVD over
     stacked snapshot pairs. Large one-step prediction error flags dynamics
     inconsistent with normal operation.

Neither component uses gradient descent, so initialization is a single SVD
rather than a training loop — the property that makes this backend the
low-latency comparator in the Phase 4 benchmark.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from statsmodels.tsa.seasonal import STL
    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False


# --------------------------------------------------------------------------- #
# Seasonal-Trend Decomposition
# --------------------------------------------------------------------------- #
@dataclass
class STLBaseline:
    period: int
    residual_std: np.ndarray     # per-channel residual std on normal data

    def residual_energy(self, window: np.ndarray) -> float:
        """Mean channel-normalized residual energy over a (T, F) window."""
        n_features = window.shape[1]
        energies = np.empty(n_features)
        for f in range(n_features):
            resid = _stl_residual(window[:, f], self.period)
            std = self.residual_std[f] if self.residual_std[f] > 1e-8 else 1e-8
            energies[f] = np.mean((resid / std) ** 2)
        return float(np.mean(energies))


def _moving_average(series: np.ndarray, k: int) -> np.ndarray:
    k = max(1, min(k, len(series)))
    return np.convolve(series, np.ones(k) / k, mode="same")


def _stl_residual(series: np.ndarray, period: int) -> np.ndarray:
    """
    Full STL when the window is long enough for it; otherwise a cheap
    moving-average detrend. The fallback keeps the backend dependency-light
    and fast-initializing (its whole selling point) rather than failing on
    short windows.
    """
    if _HAS_STATSMODELS and len(series) >= 2 * period:
        try:
            return STL(series, period=period, robust=True).fit().resid
        except Exception:
            pass
    detrended = series - _moving_average(series, max(3, period // 3))
    return detrended - detrended.mean()


def fit_stl_baseline(normal_windows: np.ndarray, period: int,
                      sample_windows: int = 200) -> STLBaseline:
    n_features = normal_windows.shape[2]
    step = max(1, len(normal_windows) // max(1, sample_windows))
    sample = normal_windows[::step]
    residual_std = np.empty(n_features)
    for f in range(n_features):
        resids = [_stl_residual(w[:, f], period) for w in sample]
        residual_std[f] = float(np.std(np.concatenate(resids))) if resids else 1.0
    return STLBaseline(period=period, residual_std=residual_std)


# --------------------------------------------------------------------------- #
# Dynamic Mode Decomposition
# --------------------------------------------------------------------------- #
@dataclass
class DMDOperator:
    """Linear operator A (n_features x n_features) with x_{t+1} ~= A x_t."""
    A: np.ndarray
    prediction_error_std: float


def fit_dmd(normal_windows: np.ndarray, rank: int,
             sample_windows: int = 2000) -> DMDOperator:
    """
    Closed-form DMD fit. Builds snapshot pairs X (t) -> X' (t+1) with sensor
    channels as rows, truncates via SVD to `rank`, solves for the reduced
    operator, then lifts back to full sensor space.

    `sample_windows` caps how many windows enter the SVD; on SWaT the full
    set would make X enormous with no accuracy benefit for a linear fit.
    """
    step = max(1, len(normal_windows) // max(1, sample_windows))
    sample = normal_windows[::step]

    X = np.concatenate([w[:-1].T for w in sample], axis=1)
    Xp = np.concatenate([w[1:].T for w in sample], axis=1)

    U, S, _ = np.linalg.svd(X, full_matrices=False)
    r = int(min(rank, U.shape[1], np.sum(S > 1e-10)))
    r = max(r, 1)
    Ur, Sr = U[:, :r], S[:r]

    # A_tilde = Ur^T Xp V Sr^-1 ; recover V from the same SVD via V = X^T U S^-1
    Vr = X.T @ Ur @ np.diag(1.0 / np.where(Sr > 1e-10, Sr, 1e-10))
    A_tilde = Ur.T @ Xp @ Vr @ np.diag(1.0 / np.where(Sr > 1e-10, Sr, 1e-10))
    A = Ur @ A_tilde @ Ur.T

    errs = np.linalg.norm(Xp - A @ X, axis=0)
    std = float(np.std(errs))
    return DMDOperator(A=A, prediction_error_std=std if std > 1e-12 else 1.0)


def dmd_anomaly_score(op: DMDOperator, window: np.ndarray) -> float:
    """
    One-step-ahead prediction error over a (T, F) window, normalized by the
    normal-data baseline. Cost is a single (F x F) @ (F x T-1) matmul, which
    is what keeps this backend inside a tight latency budget.
    """
    X, Xp = window[:-1].T, window[1:].T
    err = np.linalg.norm(Xp - op.A @ X, axis=0).mean()
    return float(err / op.prediction_error_std)


@dataclass
class FastBackend:
    stl: STLBaseline
    dmd: DMDOperator

    def score(self, window: np.ndarray) -> dict:
        """Both raw components; combined downstream in scoring.py."""
        return {
            "stl_residual_energy": self.stl.residual_energy(window),
            "dmd_prediction_error": dmd_anomaly_score(self.dmd, window),
        }


def fit_fast_backend(normal_windows: np.ndarray, cfg: dict) -> FastBackend:
    fb = cfg["fast_backend"]
    n_sample = fb.get("fit_sample_windows", 2000)
    return FastBackend(
        stl=fit_stl_baseline(normal_windows, fb["stl_period"],
                              sample_windows=min(200, n_sample)),
        dmd=fit_dmd(normal_windows, fb["dmd_rank"], sample_windows=n_sample),
    )
