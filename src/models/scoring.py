"""
Phase 2 / Step 6: the Scoring Node (evaluation matrix).

Combines, per window:
  (a) a localized error term — the Autoencoder's reconstruction error, or
      the fast backend's DMD one-step prediction error; and
  (b) the Harmonic Mean of Euclidean Distance from each in-window reading to
      a set of reference "normal operating regime" centroids.

Why the harmonic mean rather than the arithmetic mean: the harmonic mean is
dominated by the SMALLEST inputs, so a window's distance term stays low only
if its readings sit close to every relevant reference regime. Under an
arithmetic mean, proximity to a single centroid could mask a genuinely
anomalous vector. This makes the distance term stricter and regime-aware.

Both terms are z-normalized against normal-data baselines before combination
so neither dominates purely through scale.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import hmean


@dataclass
class ReferenceCentroids:
    centroids: np.ndarray        # (k, n_features)


def _pairwise_sq_dists(points: np.ndarray, centroids: np.ndarray,
                        chunk: int = 4096) -> np.ndarray:
    """
    Squared distances (N, k) computed in row chunks.

    A naive points[:, None, :] - centroids[None, :, :] allocates an
    (N, k, F) intermediate, which on SWaT-scale inputs is tens of GB.
    """
    n = points.shape[0]
    out = np.empty((n, centroids.shape[0]), dtype=np.float64)
    c_sq = np.sum(centroids ** 2, axis=1)
    for start in range(0, n, chunk):
        block = points[start:start + chunk]
        out[start:start + chunk] = (
            np.sum(block ** 2, axis=1)[:, None] - 2.0 * block @ centroids.T + c_sq[None, :]
        )
    return np.maximum(out, 0.0)


def fit_reference_centroids(normal_windows: np.ndarray, k: int = 8,
                             max_samples: int = 20000, seed: int = 42,
                             max_iter: int = 20) -> ReferenceCentroids:
    """
    Lightweight k-means (pure NumPy) over per-timestep normal vectors, giving
    k representative normal-operating-regime centroids. Input is subsampled
    to `max_samples` vectors to bound memory and fit time.
    """
    flat = normal_windows.reshape(-1, normal_windows.shape[-1])
    rng = np.random.default_rng(seed)
    if len(flat) > max_samples:
        flat = flat[rng.choice(len(flat), max_samples, replace=False)]

    k = int(min(k, len(flat)))
    # k-means++ style seeding for stability
    idx = [int(rng.integers(len(flat)))]
    for _ in range(k - 1):
        d2 = _pairwise_sq_dists(flat, flat[idx]).min(axis=1)
        total = d2.sum()
        probs = (d2 / total) if total > 0 else None
        idx.append(int(rng.choice(len(flat), p=probs)))
    centroids = flat[idx].copy()

    for _ in range(max_iter):
        assign = np.argmin(_pairwise_sq_dists(flat, centroids), axis=1)
        new_centroids = np.array([
            flat[assign == j].mean(axis=0) if np.any(assign == j) else centroids[j]
            for j in range(k)
        ])
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids

    return ReferenceCentroids(centroids=centroids)


def harmonic_mean_distance(window: np.ndarray, ref: ReferenceCentroids) -> float:
    """
    Per timestep: Euclidean distance to every centroid, then the harmonic
    mean across centroids; averaged over the window.
    """
    dists = np.sqrt(_pairwise_sq_dists(window, ref.centroids))
    dists = np.maximum(dists, 1e-8)          # hmean is undefined at zero
    return float(np.mean(hmean(dists, axis=1)))


def _normalize(x: np.ndarray, ref_mean: float, ref_std: float) -> np.ndarray:
    return (x - ref_mean) / (ref_std + 1e-8)


def fit_normalization_stats(normal_errors: np.ndarray, normal_windows: np.ndarray,
                             ref: ReferenceCentroids) -> dict:
    """Baseline mean/std of both score components, from normal data only."""
    normal_distances = np.array([harmonic_mean_distance(w, ref) for w in normal_windows])
    return {
        "recon_mean": float(np.mean(normal_errors)),
        "recon_std": float(np.std(normal_errors)) or 1.0,
        "dist_mean": float(np.mean(normal_distances)),
        "dist_std": float(np.std(normal_distances)) or 1.0,
    }


@dataclass
class ScoringResult:
    score: np.ndarray
    reconstruction_component: np.ndarray
    distance_component: np.ndarray
    threshold: float
    predictions: np.ndarray


def compute_scores(errors: np.ndarray, windows: np.ndarray, ref: ReferenceCentroids,
                    alpha: float, norm_stats: dict):
    distances = np.array([harmonic_mean_distance(w, ref) for w in windows])
    recon_z = _normalize(errors, norm_stats["recon_mean"], norm_stats["recon_std"])
    dist_z = _normalize(distances, norm_stats["dist_mean"], norm_stats["dist_std"])
    return alpha * recon_z + (1 - alpha) * dist_z, recon_z, dist_z


def score_single_window(error: float, window: np.ndarray, ref: ReferenceCentroids,
                         norm_stats: dict, alpha: float,
                         threshold: float) -> tuple:
    """
    Single-window version of the same formula, for the Phase 3 streaming
    pipeline. Vectors are scored one at a time there — batching would
    understate the true per-vector SLA latency being measured.
    """
    distance = harmonic_mean_distance(window, ref)
    recon_z = (error - norm_stats["recon_mean"]) / (norm_stats["recon_std"] + 1e-8)
    dist_z = (distance - norm_stats["dist_mean"]) / (norm_stats["dist_std"] + 1e-8)
    combined = alpha * recon_z + (1 - alpha) * dist_z
    return float(combined), int(combined > threshold)


def calibrate_threshold(normal_scores: np.ndarray, percentile: float) -> float:
    """Threshold from the NORMAL validation split only — no anomaly labels,
    keeping the method genuinely unsupervised."""
    return float(np.percentile(normal_scores, percentile))


def run_scoring_node(errors: np.ndarray, windows: np.ndarray, ref: ReferenceCentroids,
                      norm_stats: dict, alpha: float, threshold: float) -> ScoringResult:
    combined, recon_z, dist_z = compute_scores(errors, windows, ref, alpha, norm_stats)
    return ScoringResult(
        score=combined, reconstruction_component=recon_z, distance_component=dist_z,
        threshold=threshold, predictions=(combined > threshold).astype(int),
    )
