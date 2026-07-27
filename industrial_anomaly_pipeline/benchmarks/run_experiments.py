"""
Phase 4: full empirical benchmark harness.

For each benchmark dataset (real NASA C-MAPSS + synthetic TCM5) and each
analytics engine (DMD+STL fast-track, LSTM Autoencoder, Dense Autoencoder):
  1. Fit the engine on normal-operation windows only, timing initialization.
  2. Calibrate the composite scoring node (reference centroids + dynamic
     threshold) on a held-out normal validation split.
  3. Stream the labeled test split through the fitted engine + composite
     scorer *in order* (a realistic streaming simulation, not a shuffled
     batch), recording per-window latency (LatencyTracker) and classifying
     against ground truth (F1 / AUC-ROC / Precision / Recall).
Also runs a burst-throughput stress test through the real asyncio
ingestion pipeline (producer -> bounded queue -> consumer -> engine) to
evaluate elasticity: throughput, drop rate, and whether latency stays
stable under a sudden load spike.

All results are written to benchmarks/results/experiment_results.json.

Usage:
    python benchmarks/run_experiments.py
    python benchmarks/run_experiments.py --quick   # smaller run for a fast smoke test
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from analytics.autoencoder import AutoencoderDetector
from analytics.base import BaseAnomalyDetector
from analytics.dmd_decomposer import DMDSTLDetector
from benchmarks.test_latency import LatencyTracker
from evaluation.metrics_engine import CompositeScoringEngine, compute_classification_metrics
from ingestion.benchmark_loader import LoadedDataset, load_benchmark
from ingestion.stream_consumer import StreamConsumer
from ingestion.stream_producer import SensorStreamProducer, StreamProducerConfig, run_producers

logger = logging.getLogger("benchmarks.run_experiments")
RESULTS_DIR = Path(__file__).resolve().parent / "results"
SLA_MS = 20.0


# --------------------------------------------------------------------------- #
# Windowing helpers (self-contained; no cross-import from the Phase 1/2 src/)
# --------------------------------------------------------------------------- #
def make_windows(array: np.ndarray, window_size: int, stride: int = 1) -> np.ndarray:
    t = array.shape[0]
    if t < window_size:
        raise ValueError(f"series length {t} shorter than window_size {window_size}")
    starts = range(0, t - window_size + 1, stride)
    return np.stack([array[s:s + window_size] for s in starts])


def window_labels(labels: np.ndarray, window_size: int, stride: int = 1) -> np.ndarray:
    starts = range(0, len(labels) - window_size + 1, stride)
    return np.array([int(labels[s:s + window_size].max()) for s in starts])


def train_val_split(windows: np.ndarray, val_frac: float, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(windows))
    n_val = max(1, int(len(windows) * val_frac))
    return windows[idx[n_val:]], windows[idx[:n_val]]


# --------------------------------------------------------------------------- #
# Per-engine, per-dataset evaluation: fit -> calibrate -> stream test set
# --------------------------------------------------------------------------- #
def evaluate_engine(
    engine_name: str,
    detector: BaseAnomalyDetector,
    train_windows: np.ndarray,
    val_windows: np.ndarray,
    test_windows: np.ndarray,
    test_labels: np.ndarray,
) -> dict[str, Any]:
    logger.info("[%s] fitting on %d normal windows...", engine_name, len(train_windows))
    fit_start = time.perf_counter()
    detector.fit(train_windows)
    fit_time_sec = time.perf_counter() - fit_start

    val_errors = np.array([detector.score_window(w).score for w in val_windows])
    composite = CompositeScoringEngine(k_centroids=8).fit(val_windows, warm_start_errors=val_errors)

    tracker = LatencyTracker(sla_ms=SLA_MS)
    composite_scores, predictions = [], []
    for window in test_windows:
        result = detector.score_window(window)
        tracker.record(result.latency_ms)
        comp = composite.score(window, result.score)
        composite_scores.append(comp.composite)
        predictions.append(int(comp.is_anomaly))

    metrics = compute_classification_metrics(
        test_labels, np.array(composite_scores), y_pred=np.array(predictions)
    )

    result = {
        "engine": engine_name,
        "fit_time_sec": fit_time_sec,
        "n_train_windows": len(train_windows),
        "n_val_windows": len(val_windows),
        "n_test_windows": len(test_windows),
        "test_anomaly_rate": float(test_labels.mean()),
        "metrics": metrics.as_dict(),
        "latency": tracker.summary(),
        "params": detector.get_params(),
    }
    logger.info(
        "[%s] fit=%.2fs F1=%.3f AUC=%.3f P=%.3f R=%.3f p50=%.3fms p99=%.3fms sla_violations=%.2f%%",
        engine_name, fit_time_sec, metrics.f1, metrics.auc_roc, metrics.precision, metrics.recall,
        result["latency"]["p50_ms"], result["latency"]["p99_ms"],
        100 * result["latency"]["sla_violation_rate"],
    )
    return result


# --------------------------------------------------------------------------- #
# Stress / elasticity test: burst throughput through the real async pipeline
# --------------------------------------------------------------------------- #
async def run_burst_stress_test(
    detector: BaseAnomalyDetector,
    replay_source: np.ndarray,
    window_size: int,
    n_producers: int = 8,
    burst_rate_hz: float = 5000.0,
    messages_per_producer: int = 600,
    queue_maxsize: int = 500,
    max_workers: int = 8,
    max_inflight: int = 32,
) -> dict[str, Any]:
    tracker = LatencyTracker(sla_ms=SLA_MS)
    lock = threading.Lock()

    def handler(_producer_id: str, window: np.ndarray, _meta) -> None:
        result = detector.score_window(window)
        with lock:
            tracker.record(result.latency_ms)

    queue: asyncio.Queue = asyncio.Queue(maxsize=queue_maxsize)
    stop_event = asyncio.Event()
    producers = [
        SensorStreamProducer(
            f"burst-{i}",
            StreamProducerConfig(
                n_features=replay_source.shape[1], sample_rate_hz=burst_rate_hz,
                overflow_policy="drop_new", seed=i,
            ),
            source=replay_source,
        )
        for i in range(n_producers)
    ]
    consumer = StreamConsumer(
        window_size=window_size, stride=window_size, handler=handler,
        max_workers=max_workers, max_inflight=max_inflight,
    )

    consumer_task = asyncio.create_task(consumer.run(queue, stop_event=stop_event))
    start = time.perf_counter()
    emitted = await run_producers(queue, producers, max_messages_per_producer=messages_per_producer)
    stop_event.set()
    received = await consumer_task
    elapsed = time.perf_counter() - start
    dropped_total = sum(p.dropped_count for p in producers)

    result = {
        "n_producers": n_producers,
        "target_rate_hz_per_producer": burst_rate_hz,
        "messages_per_producer": messages_per_producer,
        "emitted": emitted,
        "received": received,
        "dropped": dropped_total,
        "drop_rate": dropped_total / max(emitted, 1),
        "elapsed_sec": elapsed,
        "throughput_pkts_per_sec": received / max(elapsed, 1e-9),
        "windows_scored": tracker.summary()["count"],
        "latency": tracker.summary(),
    }
    logger.info(
        "[stress] %d producers @ %.0fHz -> %d pkts in %.2fs (%.0f pkts/s), dropped=%d (%.1f%%), "
        "windows_scored=%d, p99_latency=%.3fms",
        n_producers, burst_rate_hz, received, elapsed, result["throughput_pkts_per_sec"],
        dropped_total, 100 * result["drop_rate"], result["windows_scored"], result["latency"]["p99_ms"],
    )
    return result


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def build_engines(window_size: int, quick: bool) -> dict[str, BaseAnomalyDetector]:
    ae_kwargs = dict(
        epochs=6 if quick else 15,
        hidden_dim=16 if quick else 32,
        latent_dim=8 if quick else 12,
        batch_size=128,
        early_stopping_patience=3 if quick else 4,
    )
    return {
        "dmd_stl_fast_track": DMDSTLDetector(stl_period=window_size, dmd_rank=8),
        "lstm_autoencoder": AutoencoderDetector(architecture="lstm", **ae_kwargs),
        "dense_autoencoder": AutoencoderDetector(architecture="dense", **ae_kwargs),
    }


def run_dataset_experiments(
    dataset_name: str, loaded: LoadedDataset, window_size: int, stride: int, quick: bool,
) -> list[dict[str, Any]]:
    normal_array, test_array = loaded.to_replay_arrays()
    normal_windows = make_windows(normal_array, window_size, stride)
    train_windows, val_windows = train_val_split(normal_windows, val_frac=0.15)
    test_windows = make_windows(test_array, window_size, stride)
    test_labels = window_labels(loaded.test_labels, window_size, stride)

    logger.info(
        "[%s] windows: train=%d val=%d test=%d (anomaly_rate=%.1f%%)",
        dataset_name, len(train_windows), len(val_windows), len(test_windows),
        100 * test_labels.mean(),
    )

    results = []
    for engine_name, detector in build_engines(window_size, quick).items():
        try:
            outcome = evaluate_engine(
                f"{dataset_name}/{engine_name}", detector,
                train_windows, val_windows, test_windows, test_labels,
            )
            outcome["dataset"] = dataset_name
            outcome["engine_type"] = engine_name
            results.append(outcome)
        except Exception:
            logger.exception("engine %s failed on dataset %s", engine_name, dataset_name)
    return results


def run_stress_suite(
    datasets: dict[str, LoadedDataset], window_size: int, quick: bool,
) -> list[dict[str, Any]]:
    """
    Two complementary burst scenarios per (dataset, engine):
      - "moderate": a sustained rate the queue/thread-pool capacity is sized
        for -- shows the healthy operating envelope (near-zero drops, SLA held).
      - "extreme": a deliberately adversarial spike well beyond provisioned
        capacity -- shows *how* the system degrades (graceful load-shedding
        via the overflow policy + bounded memory, not a crash or unbounded
        queue growth) and exposes the latency gap between the two engines
        once the handler itself becomes the bottleneck.
    """
    stress_results = []
    messages = 150 if quick else 600
    scenarios = (
        {"label": "moderate", "n_producers": 4, "burst_rate_hz": 400.0, "queue_maxsize": 500, "max_inflight": 32},
        {"label": "extreme", "n_producers": 8, "burst_rate_hz": 5000.0, "queue_maxsize": 500, "max_inflight": 32},
    )
    for dataset_name, loaded in datasets.items():
        normal_array, _ = loaded.to_replay_arrays()
        train_windows = make_windows(normal_array, window_size, stride=window_size)
        for engine_name, builder in (
            ("dmd_stl_fast_track", lambda: DMDSTLDetector(stl_period=window_size, dmd_rank=8)),
            ("lstm_autoencoder", lambda: AutoencoderDetector(
                architecture="lstm", epochs=4 if quick else 8, hidden_dim=16, latent_dim=8)),
        ):
            detector = builder()
            detector.fit(train_windows)
            for scenario in scenarios:
                label = scenario["label"]
                kwargs = {k: v for k, v in scenario.items() if k != "label"}
                outcome = asyncio.run(
                    run_burst_stress_test(
                        detector, normal_array, window_size,
                        messages_per_producer=messages, **kwargs,
                    )
                )
                outcome["dataset"] = dataset_name
                outcome["engine_type"] = engine_name
                outcome["scenario"] = label
                stress_results.append(outcome)
    return stress_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--quick", action="store_true", help="smaller run for a fast smoke test")
    parser.add_argument("--skip-stress", action="store_true")
    parser.add_argument("--out", default=str(RESULTS_DIR / "experiment_results.json"))
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    datasets: dict[str, LoadedDataset] = {}
    try:
        datasets["nasa_cmapss"] = load_benchmark("nasa_cmapss")
    except FileNotFoundError as exc:
        logger.warning("skipping nasa_cmapss: %s", exc)
    datasets["tcm5"] = load_benchmark(
        "tcm5", n_normal=3000 if args.quick else 6000, n_test=1500 if args.quick else 3000
    )

    all_results: dict[str, Any] = {"sla_ms": SLA_MS, "window_size": args.window_size, "stride": args.stride}
    classification_results = []
    for name, loaded in datasets.items():
        classification_results.extend(
            run_dataset_experiments(name, loaded, args.window_size, args.stride, args.quick)
        )
    all_results["classification_and_latency"] = classification_results

    if not args.skip_stress:
        all_results["stress_tests"] = run_stress_suite(datasets, args.window_size, args.quick)
    else:
        all_results["stress_tests"] = []

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2))
    logger.info("results written to %s", out_path)

    print("\n=== Summary ===")
    for r in classification_results:
        print(
            f"{r['dataset']:>12s} / {r['engine_type']:<20s} "
            f"F1={r['metrics']['f1']:.3f} AUC={r['metrics']['auc_roc']:.3f} "
            f"P={r['metrics']['precision']:.3f} R={r['metrics']['recall']:.3f} "
            f"p50={r['latency']['p50_ms']:.3f}ms p99={r['latency']['p99_ms']:.3f}ms "
            f"fit={r['fit_time_sec']:.2f}s"
        )
    for s in all_results["stress_tests"]:
        print(
            f"[stress:{s['scenario']:<8s}] {s['dataset']:>12s} / {s['engine_type']:<20s} "
            f"{s['throughput_pkts_per_sec']:.0f} pkts/s drop_rate={s['drop_rate']:.1%} "
            f"p99={s['latency']['p99_ms']:.3f}ms"
        )


if __name__ == "__main__":
    main()
