"""
CLI runner: wires the asynchronous stream producer(s) -> shared queue ->
non-blocking stream consumer -> a `BaseAnomalyDetector` together, so the
ingestion layer can be exercised end to end.

Usage:
    python main.py --n-producers 5 --n-features 20 --messages-per-producer 500

The detector used here (`_BaselineDistanceDetector`) is a minimal
`BaseAnomalyDetector` implementation for wiring/smoke-testing purposes only.
The real analytical engines (Seasonal-Trend + DMD fast-track, LSTM/Dense
Autoencoder) live in `analytics/dmd_decomposer.py` and
`analytics/autoencoder.py`.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import threading
import time

import numpy as np

from analytics.base import BaseAnomalyDetector
from ingestion.stream_consumer import StreamConsumer, WindowMeta
from ingestion.stream_producer import (
    SensorStreamProducer,
    StreamProducerConfig,
    run_producers,
)

logger = logging.getLogger("industrial_anomaly_pipeline.main")


class _BaselineDistanceDetector(BaseAnomalyDetector):
    """Minimal concrete detector: per-channel z-distance from the normal-data mean.

    Exists purely to demonstrate the ingestion layer driving a real
    `BaseAnomalyDetector` end to end; not one of the two production engines.
    """

    def __init__(self) -> None:
        super().__init__(name="baseline-distance")
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    def fit(self, normal_windows: np.ndarray) -> "_BaselineDistanceDetector":
        flat = normal_windows.reshape(-1, normal_windows.shape[-1])
        self._mean = flat.mean(axis=0)
        self._std = flat.std(axis=0) + 1e-8
        self._is_fitted = True
        return self

    def _raw_score(self, window: np.ndarray) -> tuple[float, dict[str, float]]:
        z = (window - self._mean) / self._std
        score = float(np.mean(np.linalg.norm(z, axis=1)))
        return score, {"mean_z_distance": score}


class RunStats:
    """Thread-safe aggregation of per-window results (handler runs off the event loop)."""

    def __init__(self, sla_ms: float) -> None:
        self.sla_ms = sla_ms
        self._lock = threading.Lock()
        self.window_count = 0
        self.anomaly_count = 0
        self.sla_violations = 0
        self.latencies_ms: list[float] = []

    def record(self, latency_ms: float, is_anomaly: bool) -> None:
        with self._lock:
            self.window_count += 1
            self.anomaly_count += int(is_anomaly)
            self.sla_violations += int(latency_ms > self.sla_ms)
            self.latencies_ms.append(latency_ms)

    def summary(self) -> dict:
        with self._lock:
            lat = np.array(self.latencies_ms) if self.latencies_ms else np.array([0.0])
            return {
                "windows_scored": self.window_count,
                "anomalies_flagged": self.anomaly_count,
                "sla_violations_gt_%gms" % self.sla_ms: self.sla_violations,
                "latency_ms_p50": float(np.percentile(lat, 50)),
                "latency_ms_p99": float(np.percentile(lat, 99)),
                "latency_ms_max": float(lat.max()),
            }


def build_handler(detector: BaseAnomalyDetector, stats: RunStats):
    def handle(producer_id: str, window: np.ndarray, meta: WindowMeta) -> None:
        result = detector.score_window(window)
        stats.record(result.latency_ms, result.is_anomaly)
        if result.is_anomaly:
            logger.warning(
                "anomaly: producer=%s window_id=%d score=%.3f latency_ms=%.3f "
                "synthetic_ground_truth=%s",
                producer_id, meta.window_id, result.score, result.latency_ms,
                meta.contains_synthetic_anomaly,
            )

    return handle


async def run_pipeline(args: argparse.Namespace) -> RunStats:
    queue: "asyncio.Queue" = asyncio.Queue(maxsize=args.queue_maxsize)
    stop_event = asyncio.Event()

    producer_config = StreamProducerConfig(
        n_features=args.n_features,
        sample_rate_hz=args.sample_rate_hz,
        anomaly_probability=args.anomaly_probability,
    )
    producers = [
        SensorStreamProducer(
            producer_id=f"rig-{i}",
            config=StreamProducerConfig(**{**producer_config.__dict__, "seed": i}),
        )
        for i in range(args.n_producers)
    ]

    # Fit the demo detector on a quick synthetic baseline so score_window()
    # has something to compare against (no real "normal" corpus in step 1).
    # Mirrors the producers' own signal shape (sine + Gaussian noise) without
    # reaching into their internals.
    rng = np.random.default_rng(0)
    t = np.arange(500)
    freq = rng.uniform(0.05, 0.2, size=args.n_features)
    phase = rng.uniform(0, 2 * np.pi, size=args.n_features)
    bootstrap = np.sin(freq[None, :] * t[:, None] + phase[None, :]) + rng.normal(
        0.0, 0.05, size=(500, args.n_features)
    )
    detector = _BaselineDistanceDetector()
    detector.fit(bootstrap.reshape(1, *bootstrap.shape))
    detector.set_threshold(3.0)

    stats = RunStats(sla_ms=args.sla_ms)
    consumer = StreamConsumer(
        window_size=args.window_size,
        stride=args.stride,
        handler=build_handler(detector, stats),
        max_workers=args.consumer_workers,
    )

    consumer_task = asyncio.create_task(
        consumer.run(queue, stop_event=stop_event)
    )

    start = time.perf_counter()
    total_emitted = await run_producers(
        queue, producers, max_messages_per_producer=args.messages_per_producer
    )
    stop_event.set()
    total_received = await consumer_task
    elapsed = time.perf_counter() - start

    logger.info(
        "ingestion complete: emitted=%d received=%d elapsed=%.2fs (%.0f pkts/s)",
        total_emitted, total_received, elapsed, total_received / max(elapsed, 1e-9),
    )
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-producers", type=int, default=20)
    parser.add_argument("--n-features", type=int, default=100)
    parser.add_argument("--sample-rate-hz", type=float, default=1000.0)
    parser.add_argument("--messages-per-producer", type=int, default=10000)
    parser.add_argument("--anomaly-probability", type=float, default=0.02)
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--consumer-workers", type=int, default=16)
    parser.add_argument("--queue-maxsize", type=int, default=10000)
    parser.add_argument("--sla-ms", type=float, default=20.0)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    stats = asyncio.run(run_pipeline(args))
    print("\n=== Ingestion smoke-run summary ===")
    for key, value in stats.summary().items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
