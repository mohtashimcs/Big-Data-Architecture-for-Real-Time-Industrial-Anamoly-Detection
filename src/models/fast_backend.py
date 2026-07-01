"""
lightweight, low-overhead, fast-initializing alternative
to the deep Autoencoder ("VersaGuardian-style" fast mathematical backend).

Two ingredients:
  1. Seasonal-Trend decomposition (STL) per sensor channel, fit once on the
     normal baseline to learn the expected seasonal + trend signature.
     At scoring time we decompose the incoming window and compare its
     residual energy to the learned baseline residual distribution.
  2. Dynamic Mode Decomposition (DMD) over short multivariate snapshot
     windows, which gives a fast, closed-form (no gradient descent) linear
     operator approximating the sensor dynamics. Large one-step prediction
     error from the DMD operator flags anomalous dynamics.

Both are O(window_size * n_features^2) or cheaper at inference time (matrix
multiplies / one small SVD), which is why this backend can hit a sub-20ms
SLA where a full deep model might struggle at high throughput.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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
    residual_std: np.ndarray   # per-channel std of residuals on normal data

    def residual_energy(self, window: np.ndarray) -> float:
        """
        window: (window_size, n_features). Decompose each channel, compute
        the channel-normalized residual energy, average across channels.
        """
        n_features = window.shape[1]
        energies = np.empty(n_features)
        for f in range(n_features):
            series = window[:, f]
            resid = _stl_residual(series, self.period)
            std = self.residual_std[f] if self.residual_std[f] > 1e-8 else 1e-8
            energies[f] = np.mean((resid / std) ** 2)
        return float(np.mean(energies))


def _stl_residual(series: np.ndarray, period: int) -> np.ndarray:
    if _HAS_STATSMODELS and len(series) >= 2 * period:
        try:
            result = STL(series, period=period, robust=True).fit()
            return result.resid
        except Exception:
            pass
    # Fallback: cheap moving-average detrend/deseasonalize when STL can't run
    # (short windows, or statsmodels unavailable) — keeps the backend
    # "fast-initializing" and dependency-light as a true VersaGuardian-style
    # fallback.
    trend = _moving_average(series, max(3, period // 3))
    detrended = series - trend
    return detrended - detrended.mean()


def _moving_average(series: np.ndarray, k: int) -> np.ndarray:
    k = max(1, min(k, len(series)))
    kernel = np.ones(k) / k
    return np.convolve(series, kernel, mode="same")


def fit_stl_baseline(normal_windows: np.ndarray, period: int) -> STLBaseline:
    """Fit on normal windows: learn per-channel residual std after STL."""
    n_features = normal_windows.shape[2]
    sample = normal_windows[:: max(1, len(normal_windows) // 200)]  # cap cost
    residual_std = np.empty(n_features)
    for f in range(n_features):
        resids = []
        for w in sample:
            resids.append(_stl_residual(w[:, f], period))
        residual_std[f] = np.std(np.concatenate(resids)) if resids else 1.0
    return STLBaseline(period=period, residual_std=residual_std)


# --------------------------------------------------------------------------- #
# Dynamic Mode Decomposition (DMD)
# --------------------------------------------------------------------------- #
@dataclass
class DMDOperator:
    """Linear operator A (n_features x n_features) s.t. x_{t+1} ~= A x_t."""
    A: np.ndarray
    prediction_error_std: float


def fit_dmd(normal_windows: np.ndarray, rank: int) -> DMDOperator:
    """
    closed-form DMD fit on normal snapshot windows.
    Stack windows as paired snapshots X (t) -> X' (t+1), truncate via SVD to
    `rank`, and solve for the best-fit linear operator A in the reduced
    subspace, then lift back to full sensor space.
    """
    # Build snapshot pairs across all normal windows, transposed so columns
    # are time-steps and rows are sensor channels: shape (n_features, T)
    X_list, Xp_list = [], []
    for w in normal_windows:
        X_list.append(w[:-1].T)
        Xp_list.append(w[1:].T)
    X = np.concatenate(X_list, axis=1)
    Xp = np.concatenate(Xp_list, axis=1)

    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    r = min(rank, U.shape[1])
    Ur, Sr, Vr = U[:, :r], S[:r], Vt[:r, :].T

    Sr_inv = np.diag(1.0 / np.where(Sr > 1e-10, Sr, 1e-10))
    A_tilde = Ur.T @ Xp @ Vr @ Sr_inv          # reduced operator (r x r)
    A = Ur @ A_tilde @ Ur.T                     # lift back to full space

    # Baseline one-step prediction error on normal data, for normalization.
    preds = (A @ X)
    errs = np.linalg.norm(Xp - preds, axis=0)
    return DMDOperator(A=A, prediction_error_std=float(np.std(errs)) or 1.0)


def dmd_anomaly_score(op: DMDOperator, window: np.ndarray) -> float:
    """
    window: (window_size, n_features). One-step-ahead prediction error using
    the fitted linear operator, normalized by the baseline error std.
    Cheap: a single (n_features x n_features) @ (n_features x T) matmul.
    """
    X = window[:-1].T
    Xp = window[1:].T
    preds = op.A @ X
    err = np.linalg.norm(Xp - preds, axis=0).mean()
    return float(err / op.prediction_error_std)


@dataclass
class FastBackend:
    stl: STLBaseline
    dmd: DMDOperator

    def score(self, window: np.ndarray) -> dict:
        """Returns both raw component scores; combined in scoring.py."""
        return {
            "stl_residual_energy": self.stl.residual_energy(window),
            "dmd_prediction_error": dmd_anomaly_score(self.dmd, window),
        }


def fit_fast_backend(normal_windows: np.ndarray, cfg: dict) -> FastBackend:
    fb_cfg = cfg["fast_backend"]
    stl = fit_stl_baseline(normal_windows, period=fb_cfg["stl_period"])
    dmd = fit_dmd(normal_windows, rank=fb_cfg["dmd_rank"])
    return FastBackend(stl=stl, dmd=dmd)
