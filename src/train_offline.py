"""
Usage:
    python src/train_offline.py --dataset nasa --config configs/config.yaml
    python src/train_offline.py --dataset swat --config configs/config.yaml
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import f1_score, roc_auc_score

from data_loader import load_dataset
from preprocessing import prepare_pipeline_data
from models.autoencoder import train_autoencoder, reconstruction_error
from models.fast_backend import fit_fast_backend
from models.scoring import (
    fit_reference_centroids,
    fit_normalization_stats,
    calibrate_threshold,
    run_scoring_node,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["nasa", "swat"], required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--outdir", default="outputs")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    outdir = Path(args.outdir) / args.dataset
    outdir.mkdir(parents=True, exist_ok=True)

    # ---------------- 1: Environment Setup & Dataset Prep ----------------
    print(f"[Phase 1] Loading and cleaning '{args.dataset}' dataset...")
    loaded = load_dataset(args.dataset, cfg)

    print("[Phase 1] Scaling (fit on normal data only) and windowing...")
    pp_cfg = cfg["preprocessing"]
    data = prepare_pipeline_data(
        loaded,
        window_size=pp_cfg["window_size"],
        stride=pp_cfg["window_stride"],
        val_split=pp_cfg["val_split"],
    )
    print(f"[Phase 1] train_windows={data['train_windows'].shape} "
          f"val_windows={data['val_windows'].shape} "
          f"test_windows={data['test_windows'].shape} "
          f"test_anomaly_rate={data['test_window_labels'].mean():.3f}")

    # ----------------2: Autoencoder ----------------
    print("[Phase 2] Training Autoencoder on normal data only...")
    ae_result = train_autoencoder(data["train_windows"], data["val_windows"], cfg)
    print(f"[Phase 2] AE trained in {ae_result.train_time_sec:.1f}s, "
          f"final val_loss={ae_result.val_losses[-1]:.6f}")

    # ----------------3: Fast mathematical backend ----------------
    print("[Phase 2] Fitting fast mathematical backend (STL + DMD)...")
    t0 = time.perf_counter()
    fast_backend = fit_fast_backend(data["train_windows"], cfg)
    print(f"[Phase 2] Fast backend fit in {time.perf_counter() - t0:.2f}s "
          "(no gradient descent — closed-form)")

    # ----------------4: Scoring node ----------------
    print("[Phase 2] Building scoring node (reconstruction error + "
          "harmonic mean of Euclidean distance)...")
    ref = fit_reference_centroids(data["train_windows"], k=8)

    # --- Autoencoder-backed score ---
    val_recon_err = reconstruction_error(ae_result.model, data["val_windows"])
    norm_stats_ae = fit_normalization_stats(val_recon_err, data["val_windows"], ref)
    val_scoring_ae = run_scoring_node(
        val_recon_err, data["val_windows"], ref, norm_stats_ae,
        cfg["scoring"]["alpha"], threshold=0,  # threshold calibrated from this run's scores next
    )
    threshold_ae = calibrate_threshold(val_scoring_ae.score, cfg["scoring"]["threshold_percentile"])

    test_recon_err = reconstruction_error(ae_result.model, data["test_windows"])
    ae_scoring = run_scoring_node(
        test_recon_err, data["test_windows"], ref, norm_stats_ae,
        cfg["scoring"]["alpha"], threshold_ae,
    )

    # --- Fast-backend score (DMD prediction error in place of AE recon error) ---
    from models.fast_backend import dmd_anomaly_score
    val_dmd_err = np.array([dmd_anomaly_score(fast_backend.dmd, w) for w in data["val_windows"]])
    norm_stats_fb = fit_normalization_stats(val_dmd_err, data["val_windows"], ref)
    val_scoring_fb = run_scoring_node(
        val_dmd_err, data["val_windows"], ref, norm_stats_fb,
        cfg["scoring"]["alpha"], threshold=0,
    )
    threshold_fb = calibrate_threshold(val_scoring_fb.score, cfg["scoring"]["threshold_percentile"])

    test_dmd_err = np.array([dmd_anomaly_score(fast_backend.dmd, w) for w in data["test_windows"]])
    fb_scoring = run_scoring_node(
        test_dmd_err, data["test_windows"], ref, norm_stats_fb,
        cfg["scoring"]["alpha"], threshold_fb,
    )

    # ----------------offline metrics ----------------
    labels = data["test_window_labels"]

    def _safe_auc(y, s):
        return roc_auc_score(y, s) if len(np.unique(y)) > 1 else float("nan")

    # print("\n=== Offline evaluation (Phase 4 — full benchmarking in Phase 4) ===")
    print(f"Autoencoder backend : F1={f1_score(labels, ae_scoring.predictions):.3f}  "
          f"AUCROC={_safe_auc(labels, ae_scoring.score):.3f}")
    print(f"Fast (STL+DMD) backend: F1={f1_score(labels, fb_scoring.predictions):.3f}  "
          f"AUCROC={_safe_auc(labels, fb_scoring.score):.3f}")

    # ---------------- Persist artifacts for streaming pipeline ----------------
    import torch
    torch.save(ae_result.model.state_dict(), outdir / "autoencoder.pt")
    with open(outdir / "fast_backend.pkl", "wb") as fh:
        pickle.dump(fast_backend, fh)
    with open(outdir / "reference_centroids.pkl", "wb") as fh:
        pickle.dump(ref, fh)
    with open(outdir / "norm_stats.pkl", "wb") as fh:
        pickle.dump({"ae": norm_stats_ae, "fast_backend": norm_stats_fb,
                     "threshold_ae": threshold_ae, "threshold_fb": threshold_fb}, fh)
    with open(outdir / "scaler.pkl", "wb") as fh:
        pickle.dump(data["scaler"], fh)
    print(f"\nArtifacts saved to {outdir}/ (ready for Phase 3 streaming integration)")


if __name__ == "__main__":
    main()
