"""
Phase 3 / Step 7: Ingestion Layer.

Multi-threaded simulation of a high-throughput industrial data stream. A
producer thread reads the cleaned, pre-scaled sensor array and pushes
multivariate windows into a bounded queue, standing in for a real IIoT
message bus (MQTT/Kafka/OPC-UA) for benchmarking purposes.

Design points worth citing in the Methodology chapter:
  - `queue.Queue` (thread-safe) rather than multiprocessing: the hot path is
    dominated by NumPy/PyTorch ops that release the GIL, so threads suffice
    and avoid per-vector serialization costs across process boundaries —
    exactly the I/O bottleneck the proposal warns against.
  - The queue is BOUNDED, so producer back-pressure is an explicit,
    observable property rather than an implicit assumption.
  - Two overflow policies: blocking (steady state) and drop-on-full
    (Step 11 elasticity test, mimicking a broker's overflow behaviour).
  - A sentinel (STREAM_END) signals completion, so consumers terminate
    without a separately-synchronized shutdown flag.
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass

import numpy as np

STREAM_END = object()      # sentinel: no more vectors


@dataclass
class SensorVector:
    seq_id: int
    values: np.ndarray          # (window_size, n_features)
    enqueue_time: float
    is_anomaly_true: int = -1   # ground truth if known, else -1


@dataclass
class IngestionStats:
    total_produced: int = 0
    dropped: int = 0
    produce_time_sec: float = 0.0
    queue_high_watermark: int = 0


def make_ingestion_queue(maxsize: int = 1000) -> "queue.Queue":
    return queue.Queue(maxsize=maxsize)


class StreamingProducer:
    """Pushes windows into the ingestion queue at a configurable rate."""

    def __init__(self, array: np.ndarray, labels, window_size: int, stride: int,
                 out_queue: "queue.Queue", target_hz: float | None = None,
                 drop_on_full: bool = False, max_vectors: int = 0):
        """
        target_hz    : fixed production rate (vectors/sec). None = unthrottled
                       (used by the Step 11 burst test).
        drop_on_full : non-blocking put; drop and count on overflow instead of
                       stalling the producer.
        max_vectors  : optional cap on how many vectors to emit (0 = all).
        """
        self.array = array
        self.labels = labels
        self.window_size = window_size
        self.stride = stride
        self.out_queue = out_queue
        self.target_hz = target_hz
        self.drop_on_full = drop_on_full
        self.max_vectors = max_vectors
        self.stats = IngestionStats()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        start = time.perf_counter()
        interval = (1.0 / self.target_hz) if self.target_hz else 0.0
        n_windows = (self.array.shape[0] - self.window_size) // self.stride + 1
        if self.max_vectors:
            n_windows = min(n_windows, self.max_vectors)

        for seq_id in range(n_windows):
            s = seq_id * self.stride
            window = self.array[s:s + self.window_size]
            true_label = (int(self.labels[s:s + self.window_size].max())
                          if self.labels is not None else -1)
            vec = SensorVector(seq_id=seq_id, values=window,
                               enqueue_time=time.perf_counter(),
                               is_anomaly_true=true_label)
            if self.drop_on_full:
                try:
                    self.out_queue.put_nowait(vec)
                except queue.Full:
                    self.stats.dropped += 1
            else:
                self.out_queue.put(vec, block=True)   # back-pressure
            self.stats.queue_high_watermark = max(
                self.stats.queue_high_watermark, self.out_queue.qsize())
            if interval:
                time.sleep(interval)

        self.stats.total_produced = n_windows
        self.stats.produce_time_sec = time.perf_counter() - start
        self.out_queue.put(STREAM_END)

    def start(self) -> threading.Thread:
        self._thread = threading.Thread(target=self._run, name="ingestion-producer",
                                         daemon=True)
        self._thread.start()
        return self._thread

    def join(self) -> None:
        if self._thread is not None:
            self._thread.join()
