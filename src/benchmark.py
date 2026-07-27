"""
Phase 4: Testing, Benchmarking & Evaluation (Steps 9-11).

Loads the Phase 2 artifacts, wraps each backend as an interchangeable
`scorer_fn`, and drives BOTH through the same Phase 3 streaming pipeline —
identical data, identical queue mechanics, only the scoring function differs.

  Step 9  SLA & latency tracking: per-vector "queue-exit -> score-produced"
          timing, reported as mean/p50/p95/p99/max vs the 20ms SLA.
  Step 10 Performance comparison: detection accuracy (F1, AUCROC) and
          computational efficiency (latency) for both backends.
  Step 11 Stress & elasticity: unthrottled burst into a small drop-on-full
          queue, reporting throughput, drop rate and tail latency.

Usage (from the PROJECT ROOT):
    python src/benchmark.py --dataset nasa --steady-hz 200
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score, roc_auc_score

from data_loader import load_dataset
from models.autoencoder import build_autoencoder
from models.fast_backend import dmd_anomaly_score
from models.scoring import score_single_window
from streaming.ingestion import StreamingProducer, make_ingestion_queue
from streaming.pipeline import StreamingConsumer


def load_artifacts(outdir: Path, device: str = "cpu") -> dict:
    required = ["model_meta.json", "norm_stats.pkl", "reference_centroids.pkl",
                "fast_backend.pkl", "scaler.pkl", "autoencoder.pt"]
    missing = [f for f in required if not (outdir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing artifacts in {outdir}: {missing}. "
            f"Run `python src/train_offline.py --dataset {outdir.name}` first."
        )

    meta = json.loads((outdir / "model_meta.json").read_text())
    model = build_autoencoder(meta["n_features"], meta["window_size"],
                               {"autoencoder": meta["autoencoder_cfg"]})
    model.load_state_dict(torch.load(outdir / "autoencoder.pt", map_location=device))
    model.to(device).eval()

    return {
        "meta": meta,
        "ae_model": model,
        "norm_stats": pickle.loads((outdir / "norm_stats.pkl").read_bytes()),
        "ref": pickle.loads((outdir / "reference_centroids.pkl").read_bytes()),
        "fast_backend": pickle.loads((outdir / "fast_backend.pkl").read_bytes()),
        "scaler": pickle.loads((outdir / "scaler.pkl").read_bytes()),
    }


def make_ae_scorer(artifacts: dict, alpha: float, device: str = "cpu"):
    model, ref = artifacts["ae_model"], artifacts["ref"]
    stats = artifacts["norm_stats"]["ae"]
    thr = artifacts["norm_stats"]["threshold_ae"]

    def scorer(window: np.ndarray):
        with torch.no_grad():
            x = torch.tensor(window, dtype=torch.float32, device=device).unsqueeze(0)
            err = torch.mean((model(x) - x) ** 2).item()
        return score_single_window(err, window, ref, stats, alpha, thr)
    return scorer


def make_fast_scorer(artifacts: dict, alpha: float):
    fb, ref = artifacts["fast_backend"], artifacts["ref"]
    stats = artifacts["norm_stats"]["fast_backend"]
    thr = artifacts["norm_stats"]["threshold_fb"]

    def scorer(window: np.ndarray):
        err = dmd_anomaly_score(fb.dmd, window)
        return score_single_window(err, window, ref, stats, alpha, thr)
    return scorer


def _safe_auc(y, s):
    return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float("nan")


def run_steady_state(name: str, array: np.ndarray, labels: np.ndarray,
                      window_size: int, stride: int, scorer_fn,
                      target_hz: float, sla_ms: float, max_vectors: int) -> dict:
    """Steps 9-10: fixed-rate run with full latency + accuracy reporting."""
    q = make_ingestion_queue(maxsize=500)
    producer = StreamingProducer(array, labels, window_size, stride, q,
                                  target_hz=target_hz, max_vectors=max_vectors)
    consumer = StreamingConsumer(q, scorer_fn, n_consumers=1)

    producer.start()
    consumer.run()
    producer.join()

    lat = consumer.stats.latencies_ms()
    y, pred, score = (consumer.stats.true_labels(), consumer.stats.predictions(),
                      consumer.stats.scores())
    return {
        "backend": name,
        "n_scored": int(len(lat)),
        "f1": float(f1_score(y, pred, zero_division=0)) if len(np.unique(y)) > 1 else float("nan"),
        "aucroc": _safe_auc(y, score),
        "latency_ms_mean": float(np.mean(lat)),
        "latency_ms_p50": float(np.percentile(lat, 50)),
        "latency_ms_p95": float(np.percentile(lat, 95)),
        "latency_ms_p99": float(np.percentile(lat, 99)),
        "latency_ms_max": float(np.max(lat)),
        "sla_ms": sla_ms,
        "sla_violation_rate": float(np.mean(lat > sla_ms)),
        "queue_high_watermark": producer.stats.queue_high_watermark,
    }


def run_stress(name: str, array: np.ndarray, labels: np.ndarray, window_size: int,
                stride: int, scorer_fn, queue_maxsize: int, max_vectors: int) -> dict:
    """Step 11: unthrottled burst into a small drop-on-full queue."""
    q = make_ingestion_queue(maxsize=queue_maxsize)
    producer = StreamingProducer(array, labels, window_size, stride, q,
                                  target_hz=None, drop_on_full=True,
                                  max_vectors=max_vectors)
    consumer = StreamingConsumer(q, scorer_fn, n_consumers=1)

    t0 = time.perf_counter()
    producer.start()
    consumer.run()
    producer.join()
    wall = time.perf_counter() - t0

    lat = consumer.stats.latencies_ms()
    produced = max(1, producer.stats.total_produced)
    return {
        "backend": name,
        "n_produced": producer.stats.total_produced,
        "n_scored": int(len(lat)),
        "n_dropped": producer.stats.dropped,
        "drop_rate": producer.stats.dropped / produced,
        "wall_time_sec": wall,
        "effective_throughput_hz": producer.stats.total_produced / max(1e-9, wall),
        "queue_high_watermark": producer.stats.queue_high_watermark,
        "latency_ms_mean": float(np.mean(lat)) if len(lat) else float("nan"),
        "latency_ms_p99": float(np.percentile(lat, 99)) if len(lat) else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["nasa", "swat"], required=True)
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--steady-hz", type=float, default=200.0,
                     help="Simulated steady-state production rate (vectors/sec)")
    ap.add_argument("--max-vectors", type=int, default=5000,
                     help="Cap vectors per run to keep benchmarks quick (0 = all)")
    ap.add_argument("--stress-queue", type=int, default=50,
                     help="Queue size for the Step 11 burst test")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    outdir = Path(args.outdir) / args.dataset
    sla_ms = cfg["sla"]["target_latency_ms"]
    alpha = cfg["scoring"]["alpha"]

    print(f"[Phase 4] Loading Phase 2 artifacts from {outdir}/ ...")
    artifacts = load_artifacts(outdir)
    meta = artifacts["meta"]
    window_size, stride = meta["window_size"], meta["stride"]

    print(f"[Phase 4] Re-loading '{args.dataset}' test split for stream replay...")
    loaded = load_dataset(args.dataset, cfg)
    test_array = artifacts["scaler"].transform(loaded.test_df.to_numpy(dtype=np.float64))
    test_labels = loaded.test_labels

    backends = [
        ("autoencoder", make_ae_scorer(artifacts, alpha)),
        ("fast_backend", make_fast_scorer(artifacts, alpha)),
    ]

    print(f"\n[Step 9-10] Steady state @ {args.steady_hz:.0f} vec/s (SLA={sla_ms}ms)")
    steady = []
    for name, fn in backends:
        r = run_steady_state(name, test_array, test_labels, window_size, stride,
                              fn, args.steady_hz, sla_ms, args.max_vectors)
        steady.append(r)
        print(f"  {name:13s} F1={r['f1']:.3f} AUC={r['aucroc']:.3f} | "
              f"mean={r['latency_ms_mean']:.3f}ms p99={r['latency_ms_p99']:.3f}ms | "
              f"SLA violations={100*r['sla_violation_rate']:.2f}%")

    print(f"\n[Step 11] Stress test (unthrottled burst, queue={args.stress_queue}, drop-on-full)")
    stress = []
    for name, fn in backends:
        r = run_stress(name, test_array, test_labels, window_size, stride, fn,
                        args.stress_queue, args.max_vectors)
        stress.append(r)
        print(f"  {name:13s} throughput={r['effective_throughput_hz']:.1f} vec/s | "
              f"dropped={r['n_dropped']}/{r['n_produced']} ({100*r['drop_rate']:.2f}%) | "
              f"p99={r['latency_ms_p99']:.3f}ms")

    report = {"dataset": args.dataset, "sla_ms": sla_ms,
              "steady_state_hz": args.steady_hz,
              "steady_state": steady, "stress_test": stress}
    (outdir / "benchmark_report.json").write_text(json.dumps(report, indent=2))
    print(f"\n[Phase 4] Report -> {outdir/'benchmark_report.json'}")


if __name__ == "__main__":
    main()
