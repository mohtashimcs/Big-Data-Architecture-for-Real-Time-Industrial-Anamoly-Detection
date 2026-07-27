"""
Non-blocking asynchronous stream consumer.

Continuously drains `SensorPacket` messages from the shared queue,
assembles per-producer contiguous windows, and dispatches each completed
window to a handler (typically `BaseAnomalyDetector.score_window`). The
handler runs in a `ThreadPoolExecutor` via `loop.run_in_executor`, so a
slow or CPU-bound handler call (model inference) can never stall the
asyncio event loop or the packet-intake path -- this is the multi-threaded
half of the ingestion layer, paired with the asyncio-driven producer side.

In-flight handler tasks are bounded by an `asyncio.Semaphore`
(`max_inflight`). This matters because `ThreadPoolExecutor`'s own internal
work queue has no size limit: submitting to it unconditionally (via
`run_in_executor`) as packets arrive would let queued-but-not-yet-run
handler calls -- each holding a reference to its window array -- grow
without bound whenever handlers can't keep up with intake, which is a
genuine memory-growth/"thread lockup" failure mode under sustained high
throughput. Bounding in-flight tasks means a saturated handler pool instead
suspends `_dispatch` (awaiting the semaphore), which stops packet intake,
which fills the shared queue, which finally backpressures the producers --
one coherent, visible backpressure chain instead of an unbounded internal
buffer.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Deque, Dict, Optional

import numpy as np

from ingestion.stream_producer import SensorPacket

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WindowMeta:
    producer_id: str
    window_id: int
    start_sequence_id: int
    end_sequence_id: int
    ingest_latency_ms: float  # wall-clock: first packet emission -> window completion
    contains_synthetic_anomaly: bool


WindowHandler = Callable[[str, np.ndarray, WindowMeta], None]


class StreamConsumer:
    """
    Buffers packets per producer into fixed-size sliding windows of shape
    (window_size, n_features) and hands each completed window to `handler`,
    off the event loop, without blocking further packet intake.
    """

    def __init__(
        self,
        window_size: int,
        handler: WindowHandler,
        stride: int = 1,
        max_workers: int = 4,
        max_inflight: Optional[int] = None,
    ) -> None:
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        if stride < 1:
            raise ValueError("stride must be >= 1")
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        if max_inflight is not None and max_inflight < 1:
            raise ValueError("max_inflight must be >= 1")

        self.window_size = window_size
        self.stride = stride
        self.handler = handler
        self.max_inflight = max_inflight or max_workers * 4

        self._buffers: Dict[str, Deque[SensorPacket]] = defaultdict(
            lambda: deque(maxlen=window_size)
        )
        self._since_last_window: Dict[str, int] = defaultdict(int)
        self._window_counts: Dict[str, int] = defaultdict(int)
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="anomaly-detector"
        )
        self._inflight = asyncio.Semaphore(self.max_inflight)
        self._pending: list[asyncio.Future] = []
        self._prune_threshold = max(64, self.max_inflight * 4)
        self._processed = 0

    @property
    def processed_count(self) -> int:
        """Number of packets received (not windows emitted)."""
        return self._processed

    async def run(
        self,
        queue: "asyncio.Queue[SensorPacket]",
        max_messages: Optional[int] = None,
        stop_event: Optional[asyncio.Event] = None,
        poll_timeout: float = 0.5,
    ) -> int:
        """
        Drain `queue` until `max_messages` packets have been received or
        `stop_event` is set and the queue has been fully drained. Waits on
        `queue.get()` with a timeout so it can periodically re-check
        `stop_event` instead of blocking forever on an empty queue.
        """
        loop = asyncio.get_running_loop()
        received = 0

        while max_messages is None or received < max_messages:
            if stop_event is not None and stop_event.is_set() and queue.empty():
                break
            try:
                packet = await asyncio.wait_for(queue.get(), timeout=poll_timeout)
            except asyncio.TimeoutError:
                continue

            received += 1
            self._processed += 1
            queue.task_done()
            await self._dispatch(packet, loop)
            self._prune_pending()

        if self._pending:
            await asyncio.gather(*self._pending, return_exceptions=True)
        self._executor.shutdown(wait=True)
        return received

    async def _dispatch(self, packet: SensorPacket, loop: asyncio.AbstractEventLoop) -> None:
        """Append `packet` to its producer's window buffer, firing the handler when ready."""
        buf = self._buffers[packet.producer_id]
        buf.append(packet)
        self._since_last_window[packet.producer_id] += 1

        window_ready = (
            len(buf) == self.window_size
            and self._since_last_window[packet.producer_id] >= self.stride
        )
        if not window_ready:
            return
        self._since_last_window[packet.producer_id] = 0

        window_packets = list(buf)
        window = np.stack([p.values for p in window_packets], axis=0)
        meta = WindowMeta(
            producer_id=packet.producer_id,
            window_id=self._window_counts[packet.producer_id],
            start_sequence_id=window_packets[0].sequence_id,
            end_sequence_id=window_packets[-1].sequence_id,
            ingest_latency_ms=(time.time() - window_packets[0].timestamp) * 1000.0,
            contains_synthetic_anomaly=any(p.is_synthetic_anomaly for p in window_packets),
        )
        self._window_counts[packet.producer_id] += 1

        # See module docstring: bounding in-flight handler tasks here is what
        # turns "unbounded executor queue growth under overload" into
        # coherent, visible backpressure all the way back to the producers.
        await self._inflight.acquire()
        future = loop.run_in_executor(
            self._executor, self._safe_handle, packet.producer_id, window, meta
        )
        future.add_done_callback(lambda _f: self._inflight.release())
        self._pending.append(future)

    def _safe_handle(self, producer_id: str, window: np.ndarray, meta: WindowMeta) -> None:
        try:
            self.handler(producer_id, window, meta)
        except Exception:
            logger.exception(
                "handler failed for producer=%s window_id=%d", producer_id, meta.window_id
            )

    def _prune_pending(self) -> None:
        """Drop completed futures so long-running streams don't grow this list unbounded."""
        if len(self._pending) <= self._prune_threshold:
            return
        self._pending = [f for f in self._pending if not f.done()]
