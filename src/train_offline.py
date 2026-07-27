"""
Phase 1 + Phase 2 orchestration: offline modeling on static data.

Per the project's own pro-tip, this runs the complete offline path first —
load, clean, scale, window (Phase 1), then train the Autoencoder, fit the
fast mathematical backend, and run both through the Step 6 scoring node
(Phase 2). Phase 3 (benchmark.py) then wraps these saved artifacts in the
streaming pipeline without changing the models at all.

Usage (from the PROJECT ROOT):
    python src/train_offline.py --dataset nasa
    python src/train_offline.py --dataset swat
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import time
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import f1_score, roc_auc_score

from data_loader import load_dataset
from preprocessing import prepare_pipeline_data
from models.autoencoder import train_autoencoder, reconstruction_error
from models.fast_backend import fit_fast_backend, dmd_anomaly_score
from models.scoring import (
    fit_reference_centroids, fit_normalization_stats,
    calibrate_threshold, run_scoring_node,
)


def _safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) > 1 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["nasa", "swat"], required=True)
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    outdir = Path(args.outdir) / args.dataset
    outdir.mkdir(parents=True, exist_ok=True)

    pp = cfg["preprocessing"]
    stride = pp["swat_window_stride"] if args.dataset == "swat" else pp["window_stride"]

    # ---------------- Phase 1 ----------------
    print(f"[Phase 1] Loading + cleaning '{args.dataset}'...")
    loaded = load_dataset(args.dataset, cfg)

    print(f"[Phase 1] Scaling (normal-only fit) + windowing (stride={stride})...")
    data = prepare_pipeline_data(
        loaded, window_size=pp["window_size"], stride=stride,
        val_split=pp["val_split"],
        max_train_windows=pp.get("max_train_windows", 0),
        max_test_windows=pp.get("max_test_windows", 0),
    )
    print(f"[Phase 1] train={data['train_windows'].shape} "
          f"val={data['val_windows'].shape} test={data['test_windows'].shape} "
          f"test_anomaly_rate={data['test_window_labels'].mean():.3f}")

    # ---------------- Phase 2 / Step 4 ----------------
    print("[Phase 2] Training Autoencoder on normal data only...")
    ae = train_autoencoder(data["train_windows"], data["val_windows"], cfg)
    print(f"[Phase 2] AE trained in {ae.train_time_sec:.1f}s "
          f"(best val_loss={min(ae.val_losses):.6f})")

    # ---------------- Phase 2 / Step 5 ----------------
    print("[Phase 2] Fitting fast backend (STL + DMD, closed-form)...")
    t0 = time.perf_counter()
    fast_backend = fit_fast_backend(data["train_windows"], cfg)
    fb_fit_time = time.perf_counter() - t0
    print(f"[Phase 2] Fast backend fit in {fb_fit_time:.2f}s (no gradient descent)")

    # ---------------- Phase 2 / Step 6 ----------------
    print("[Phase 2] Building scoring node (error + harmonic-mean distance)...")
    sc = cfg["scoring"]
    ref = fit_reference_centroids(
        data["train_windows"], k=sc["n_centroids"],
        max_samples=sc.get("centroid_fit_samples", 20000),
    )

    # -- Autoencoder path: calibrate threshold on the NORMAL validation split --
    val_err_ae = reconstruction_error(ae.model, data["val_windows"])
    norm_ae = fit_normalization_stats(val_err_ae, data["val_windows"], ref)
    val_scores_ae = run_scoring_node(val_err_ae, data["val_windows"], ref,
                                      norm_ae, sc["alpha"], threshold=0.0).score
    thr_ae = calibrate_threshold(val_scores_ae, sc["threshold_percentile"])

    test_err_ae = reconstruction_error(ae.model, data["test_windows"])
    res_ae = run_scoring_node(test_err_ae, data["test_windows"], ref,
                               norm_ae, sc["alpha"], thr_ae)

    # -- Fast backend path (DMD prediction error replaces reconstruction error) --
    val_err_fb = np.array([dmd_anomaly_score(fast_backend.dmd, w)
                            for w in data["val_windows"]])
    norm_fb = fit_normalization_stats(val_err_fb, data["val_windows"], ref)
    val_scores_fb = run_scoring_node(val_err_fb, data["val_windows"], ref,
                                      norm_fb, sc["alpha"], threshold=0.0).score
    thr_fb = calibrate_threshold(val_scores_fb, sc["threshold_percentile"])

    test_err_fb = np.array([dmd_anomaly_score(fast_backend.dmd, w)
                             for w in data["test_windows"]])
    res_fb = run_scoring_node(test_err_fb, data["test_windows"], ref,
                               norm_fb, sc["alpha"], thr_fb)

    # ---------------- Offline metrics ----------------
    y = data["test_window_labels"]
    f1_ae, auc_ae = f1_score(y, res_ae.predictions, zero_division=0), _safe_auc(y, res_ae.score)
    f1_fb, auc_fb = f1_score(y, res_fb.predictions, zero_division=0), _safe_auc(y, res_fb.score)

    print("\n=== Offline evaluation (streaming benchmark follows in Phase 4) ===")
    print(f"Autoencoder     : F1={f1_ae:.3f}  AUCROC={auc_ae:.3f}")
    print(f"Fast (STL+DMD)  : F1={f1_fb:.3f}  AUCROC={auc_fb:.3f}")

    metrics = {
        "dataset": args.dataset,
        "n_test_windows": int(len(y)),
        "test_anomaly_rate": float(y.mean()),
        "window_size": int(pp["window_size"]),
        "stride": int(stride),
        "autoencoder": {"f1": f1_ae, "aucroc": auc_ae, "threshold": thr_ae,
                        "train_time_sec": ae.train_time_sec},
        "fast_backend": {"f1": f1_fb, "aucroc": auc_fb, "threshold": thr_fb,
                         "fit_time_sec": fb_fit_time},
    }
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    with open(outdir / "scores.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["window_idx", "true_label", "ae_score", "ae_prediction",
                     "fb_score", "fb_prediction"])
        for i in range(len(y)):
            w.writerow([i, int(y[i]), float(res_ae.score[i]), int(res_ae.predictions[i]),
                        float(res_fb.score[i]), int(res_fb.predictions[i])])

    # ---------------- Persist artifacts for Phase 3 ----------------
    import torch
    torch.save(ae.model.state_dict(), outdir / "autoencoder.pt")
    (outdir / "fast_backend.pkl").write_bytes(pickle.dumps(fast_backend))
    (outdir / "reference_centroids.pkl").write_bytes(pickle.dumps(ref))
    (outdir / "scaler.pkl").write_bytes(pickle.dumps(data["scaler"]))
    (outdir / "norm_stats.pkl").write_bytes(pickle.dumps({
        "ae": norm_ae, "fast_backend": norm_fb,
        "threshold_ae": thr_ae, "threshold_fb": thr_fb,
    }))
    (outdir / "model_meta.json").write_text(json.dumps({
        "dataset": args.dataset,
        "n_features": int(data["train_windows"].shape[2]),
        "window_size": int(data["train_windows"].shape[1]),
        "stride": int(stride),
        "sensor_columns": list(data["sensor_columns"]),
        "autoencoder_cfg": cfg["autoencoder"],
    }, indent=2))

    print(f"\nMetrics  -> {outdir/'metrics.json'}")
    print(f"Scores   -> {outdir/'scores.csv'}")
    print(f"Artifacts-> {outdir}/  (ready for: python src/benchmark.py --dataset {args.dataset})")


if __name__ == "__main__":
    main()
