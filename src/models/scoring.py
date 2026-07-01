"""
Scoring Node.

Combines, per window:
  (a) localized reconstruction error from the Autoencoder (or, in the fast
      backend's case, its DMD one-step prediction error), and
  (b) the Harmonic Mean of Euclidean Distance across the multivariate
      sensor vectors within the window, relative to a set of normal
      reference centroids.

The harmonic mean is used (rather than the arithmetic mean) because it is
dominated by the *smallest* distances, i.e. it stays low only if a vector is
close to *every* reference centroid along multiple operating regimes — a
single nearby centroid isn't enough to mask anomalies the way it would
under an arithmetic mean. This makes it a stricter, regime-aware multivariate
distance score.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import hmean


@dataclass
class ReferenceCentroids:
    centroids: np.ndarray  # (k, n_features)


def fit_reference_centroids(normal_windows: np.ndarray, k: int = 8,
                             seed: int = 42) -> ReferenceCentroids:
    """
    Lightweight k-means (pure NumPy, no sklearn dependency) over the
    per-timestep normal vectors, to get k representative "normal operating
    regime" centroids for the harmonic-mean distance score.
    """
    flat = normal_windows.reshape(-1, normal_windows.shape[-1])
    rng = np.random.default_rng(seed)
    # k-means++-style init for stability
    idx = [rng.integers(len(flat))]
    for _ in range(k - 1):
        d2 = np.min(
            [np.sum((flat - flat[i]) ** 2, axis=1) for i in idx], axis=0
        )
        probs = d2 / (d2.sum() + 1e-12)
        idx.append(rng.choice(len(flat), p=probs))
    centroids = flat[idx].copy()

    for _ in range(20):
        dists = np.linalg.norm(flat[:, None, :] - centroids[None, :, :], axis=2)
        assign = np.argmin(dists, axis=1)
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
    For each timestep vector in the window, compute Euclidean distance to
    every reference centroid, then take the harmonic mean across centroids
    (per timestep), then average across the window.
    """
    # window: (T, F), centroids: (k, F)
    dists = np.linalg.norm(
        window[:, None, :] - ref.centroids[None, :, :], axis=2
    )  # (T, k)
    dists = np.where(dists < 1e-8, 1e-8, dists)  # avoid div-by-zero in hmean
    per_timestep_hmean = hmean(dists, axis=1)     # (T,)
    return float(np.mean(per_timestep_hmean))


@dataclass
class ScoringResult:
    score: np.ndarray
    reconstruction_component: np.ndarray
    distance_component: np.ndarray
    threshold: float
    predictions: np.ndarray


def _normalize(x: np.ndarray, ref_mean: float, ref_std: float) -> np.ndarray:
    return (x - ref_mean) / (ref_std + 1e-8)


def compute_scores(reconstruction_errors: np.ndarray, windows: np.ndarray,
                    ref: ReferenceCentroids, alpha: float,
                    norm_stats: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    build the evaluation matrix. Both components are z-normalized
    against their normal-data baseline statistics (norm_stats) before being
    combined, so neither term dominates purely from scale differences.
    """
    distances = np.array([harmonic_mean_distance(w, ref) for w in windows])

    recon_z = _normalize(reconstruction_errors, norm_stats["recon_mean"], norm_stats["recon_std"])
    dist_z = _normalize(distances, norm_stats["dist_mean"], norm_stats["dist_std"])

    combined = alpha * recon_z + (1 - alpha) * dist_z
    return combined, recon_z, dist_z


def fit_normalization_stats(normal_reconstruction_errors: np.ndarray,
                             normal_windows: np.ndarray,
                             ref: ReferenceCentroids) -> dict:
    """Baseline mean/std for both components, computed on normal data only."""
    normal_distances = np.array([harmonic_mean_distance(w, ref) for w in normal_windows])
    return {
        "recon_mean": float(np.mean(normal_reconstruction_errors)),
        "recon_std": float(np.std(normal_reconstruction_errors)) or 1.0,
        "dist_mean": float(np.mean(normal_distances)),
        "dist_std": float(np.std(normal_distances)) or 1.0,
    }


def calibrate_threshold(normal_scores: np.ndarray, percentile: float) -> float:
    return float(np.percentile(normal_scores, percentile))


def run_scoring_node(reconstruction_errors: np.ndarray, windows: np.ndarray,
                      ref: ReferenceCentroids, norm_stats: dict,
                      alpha: float, threshold: float) -> ScoringResult:
    combined, recon_z, dist_z = compute_scores(
        reconstruction_errors, windows, ref, alpha, norm_stats
    )
    predictions = (combined > threshold).astype(int)
    return ScoringResult(
        score=combined,
        reconstruction_component=recon_z,
        distance_component=dist_z,
        threshold=threshold,
        predictions=predictions,
    )
