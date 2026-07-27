"""
Phase 3 / Step 8: Integrate Processing & Analytics.
Phase 4 / Step 9: SLA & Latency Tracking.

Consumer thread(s) pull SensorVector windows off the ingestion queue and run
them through a scoring backend. Model artifacts are loaded once at startup,
so there is no blocking I/O in the hot path — each vector is unpacked and
scored directly from in-memory arrays.

The module has one job: consume -> score -> record. Because the backend is
injected as `scorer_fn`, swapping the Autoencoder for the fast backend
changes nothing else in the pipeline, which is what makes the Step 10
comparison genuinely apples-to-apples.
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from streaming.ingestion import STREAM_END, SensorVector


@dataclass
class ScoredResult:
    seq_id: int
    score: float
    prediction: int
    true_label: int
    latency_ms: float          # queue-exit -> score-produced (the SLA metric)


@dataclass
class ConsumerStats:
    results: list = field(default_factory=list)

    def latencies_ms(self) -> np.ndarray:
        return np.array([r.latency_ms for r in self.results])

    def scores(self) -> np.ndarray:
        return np.array([r.score for r in self.results])

    def predictions(self) -> np.ndarray:
        return np.array([r.prediction for r in self.results])

    def true_labels(self) -> np.ndarray:
        return np.array([r.true_label for r in self.results])


class StreamingConsumer:
    """
    Step 9's timer wraps exactly the queue-exit -> score-produced span, per
    the roadmap's definition. Queue WAIT time is deliberately excluded: that
    is a property of the ingestion/back-pressure subsystem, whereas the 20ms
    SLA targets the scoring engine's own computational cost.
    """

    def __init__(self, in_queue: "queue.Queue", scorer_fn, n_consumers: int = 1):
        self.in_queue = in_queue
        self.scorer_fn = scorer_fn
        self.n_consumers = n_consumers
        self.stats = ConsumerStats()
        self._lock = threading.Lock()
        self._threads: list = []

    def _worker(self) -> None:
        while True:
            item = self.in_queue.get()
            if item is STREAM_END:
                self.in_queue.put(STREAM_END)   # re-broadcast for peers
                break
            vec: SensorVector = item

            t0 = time.perf_counter()            # timer START: vector leaves queue
            score, prediction = self.scorer_fn(vec.values)
            t1 = time.perf_counter()            # timer STOP: score produced

            result = ScoredResult(seq_id=vec.seq_id, score=score,
                                  prediction=prediction,
                                  true_label=vec.is_anomaly_true,
                                  latency_ms=(t1 - t0) * 1000.0)
            with self._lock:
                self.stats.results.append(result)

    def run(self) -> None:
        """Blocking: runs consumers until the stream is exhausted."""
        self._threads = [
            threading.Thread(target=self._worker, name=f"consumer-{i}", daemon=True)
            for i in range(self.n_consumers)
        ]
        for t in self._threads:
            t.start()
        for t in self._threads:
            t.join()
        # Multi-consumer results can interleave; sort for reproducible reports.
        self.stats.results.sort(key=lambda r: r.seq_id)
