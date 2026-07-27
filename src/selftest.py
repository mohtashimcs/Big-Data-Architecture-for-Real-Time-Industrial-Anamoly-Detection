"""
End-to-end self-test on synthetic data — no Kaggle download required.

Generates a synthetic multivariate sensor stream with injected anomalies and
runs the COMPLETE pipeline over it: preprocessing -> fast backend -> scoring
node -> streaming ingestion -> SLA latency measurement -> stress test. If
PyTorch is installed it also trains a small Autoencoder and includes it in
the comparison; if not, it skips that part and says so.

Use this to verify your environment and the code paths before spending time
on the real datasets.

Usage (from the PROJECT ROOT):
    python src/selftest.py
"""
from __future__ import annotations

import sys
import time

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from preprocessing import make_windows, window_labels, fit_minmax_on_normal, train_val_split
from models.fast_backend import fit_fast_backend, dmd_anomaly_score
from models.scoring import (
    fit_reference_centroids, fit_normalization_stats,
    calibrate_threshold, run_scoring_node, score_single_window,
)
from streaming.ingestion import StreamingProducer, make_ingestion_queue
from streaming.pipeline import StreamingConsumer

WINDOW, STRIDE, N_FEATURES = 30, 5, 6
SLA_MS = 20.0


def synth_series(n: int, n_features: int, rng, anomalous_tail: int = 0):
    """Smooth periodic multivariate signal; optional anomalous tail."""
    t = np.arange(n)
    base = np.stack([np.sin(0.05 * t + i) + 0.3 * np.cos(0.013 * t + 2 * i)
                     for i in range(n_features)], axis=1)
    series = base + 0.05 * rng.standard_normal((n, n_features))
    labels = np.zeros(n, dtype=int)
    if anomalous_tail:
        series[-anomalous_tail:] += 3.0 + 2.0 * rng.standard_normal(
            (anomalous_tail, n_features))
        labels[-anomalous_tail:] = 1
    return series, labels


def main() -> int:
    rng = np.random.default_rng(0)
    print("=== Synthetic end-to-end self-test ===\n")

    # ---------------- Phase 1 ----------------
    normal_raw, _ = synth_series(4000, N_FEATURES, rng)
    test_raw, test_lbl = synth_series(1500, N_FEATURES, rng, anomalous_tail=300)

    scaler = fit_minmax_on_normal(normal_raw)
    normal_windows = make_windows(scaler.transform(normal_raw), WINDOW, STRIDE)
    test_scaled = scaler.transform(test_raw)
    test_windows = make_windows(test_scaled, WINDOW, STRIDE)
    test_wlbl = window_labels(test_lbl, WINDOW, STRIDE)
    train_w, val_w = train_val_split(normal_windows, 0.1)
    print(f"[Phase 1] train={train_w.shape} val={val_w.shape} test={test_windows.shape} "
          f"anomaly_rate={test_wlbl.mean():.3f}  OK")

    cfg = {
        "fast_backend": {"stl_period": WINDOW, "dmd_rank": 5, "fit_sample_windows": 500},
        "scoring": {"alpha": 0.5, "threshold_percentile": 99.0, "n_centroids": 6},
        "autoencoder": {"type": "lstm", "hidden_dim": 32, "latent_dim": 8,
                         "num_layers": 1, "dropout": 0.0, "epochs": 5,
                         "batch_size": 64, "learning_rate": 0.005,
                         "early_stopping_patience": 3},
    }

    # ---------------- Phase 2: fast backend ----------------
    t0 = time.perf_counter()
    fb = fit_fast_backend(train_w, cfg)
    print(f"[Phase 2] Fast backend fit in {time.perf_counter()-t0:.2f}s "
          f"(DMD operator {fb.dmd.A.shape})  OK")

    ref = fit_reference_centroids(train_w, k=cfg["scoring"]["n_centroids"])
    val_err = np.array([dmd_anomaly_score(fb.dmd, w) for w in val_w])
    norm = fit_normalization_stats(val_err, val_w, ref)
    thr = calibrate_threshold(
        run_scoring_node(val_err, val_w, ref, norm, 0.5, 0.0).score,
        cfg["scoring"]["threshold_percentile"])

    test_err = np.array([dmd_anomaly_score(fb.dmd, w) for w in test_windows])
    res = run_scoring_node(test_err, test_windows, ref, norm, 0.5, thr)
    f1 = f1_score(test_wlbl, res.predictions, zero_division=0)
    auc = roc_auc_score(test_wlbl, res.score)
    print(f"[Phase 2] Fast backend offline: F1={f1:.3f} AUCROC={auc:.3f}  "
          f"{'OK' if f1 > 0.5 else 'WARN (low F1 on synthetic data)'}")

    # ---------------- Phase 2: Autoencoder (optional) ----------------
    ae_scorer = None
    try:
        import torch  # noqa: F401
        from models.autoencoder import train_autoencoder, reconstruction_error
        print("\n[Phase 2] PyTorch found - training a small Autoencoder...")
        ae = train_autoencoder(train_w, val_w, cfg)
        val_err_ae = reconstruction_error(ae.model, val_w)
        norm_ae = fit_normalization_stats(val_err_ae, val_w, ref)
        thr_ae = calibrate_threshold(
            run_scoring_node(val_err_ae, val_w, ref, norm_ae, 0.5, 0.0).score,
            cfg["scoring"]["threshold_percentile"])
        test_err_ae = reconstruction_error(ae.model, test_windows)
        res_ae = run_scoring_node(test_err_ae, test_windows, ref, norm_ae, 0.5, thr_ae)
        print(f"[Phase 2] Autoencoder offline: "
              f"F1={f1_score(test_wlbl, res_ae.predictions, zero_division=0):.3f} "
              f"AUCROC={roc_auc_score(test_wlbl, res_ae.score):.3f}  OK")

        import torch as _t
        model = ae.model

        def ae_scorer(window):
            with _t.no_grad():
                x = _t.tensor(window, dtype=_t.float32).unsqueeze(0)
                err = _t.mean((model(x) - x) ** 2).item()
            return score_single_window(err, window, ref, norm_ae, 0.5, thr_ae)
    except ImportError:
        print("\n[Phase 2] PyTorch NOT installed - skipping the Autoencoder path.")
        print("          Install it (`pip install torch`) to test that backend.")

    # ---------------- Phase 3 + 4: streaming ----------------
    def fb_scorer(window):
        return score_single_window(dmd_anomaly_score(fb.dmd, window),
                                    window, ref, norm, 0.5, thr)

    backends = [("fast_backend", fb_scorer)]
    if ae_scorer is not None:
        backends.append(("autoencoder", ae_scorer))

    print("\n[Phase 3-4] Steady-state streaming @ 500 vec/s:")
    for name, fn in backends:
        q = make_ingestion_queue(maxsize=200)
        prod = StreamingProducer(test_scaled, test_lbl, WINDOW, STRIDE, q,
                                  target_hz=500, max_vectors=300)
        cons = StreamingConsumer(q, fn, n_consumers=1)
        prod.start(); cons.run(); prod.join()
        lat = cons.stats.latencies_ms()
        viol = 100 * np.mean(lat > SLA_MS)
        print(f"  {name:13s} scored={len(lat):4d} mean={lat.mean():.3f}ms "
              f"p99={np.percentile(lat, 99):.3f}ms SLA_violations={viol:.2f}%  "
              f"{'OK' if viol == 0 else 'over SLA'}")

    print("\n[Phase 4/Step 11] Stress test (unthrottled, queue=10, drop-on-full):")
    for name, fn in backends:
        q = make_ingestion_queue(maxsize=10)
        prod = StreamingProducer(test_scaled, test_lbl, WINDOW, STRIDE, q,
                                  target_hz=None, drop_on_full=True, max_vectors=300)
        cons = StreamingConsumer(q, fn, n_consumers=1)
        t0 = time.perf_counter()
        prod.start(); cons.run(); prod.join()
        wall = time.perf_counter() - t0
        print(f"  {name:13s} produced={prod.stats.total_produced} "
              f"scored={len(cons.stats.results)} dropped={prod.stats.dropped} "
              f"throughput={prod.stats.total_produced/wall:.0f} vec/s  OK")

    print("\n=== SELF-TEST PASSED ===")
    print("The pipeline works end to end. Next: download the datasets, then run")
    print("  python src/train_offline.py --dataset nasa")
    print("  python src/benchmark.py    --dataset nasa")
    return 0


if __name__ == "__main__":
    sys.exit(main())
