"""
Asynchronous, non-blocking simulator for high-throughput multi-sensor
industrial telemetry.

Each `SensorStreamProducer` represents one independent sensor rig that
continuously emits high-dimensional multivariate sensor vectors ("packets")
onto a shared `asyncio.Queue`. Many producers run concurrently as asyncio
tasks (`run_producers`), emulating dozens of sensor rigs streaming in
parallel without spinning up an OS thread per rig. The queue provides
backpressure (bounded `maxsize`), so a slow consumer never causes unbounded
memory growth, and every wait point uses `await` (queue.put, asyncio.sleep)
so a single producer never blocks the event loop or its siblings.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SensorPacket:
    """One high-dimensional multivariate sensor snapshot."""

    sequence_id: int
    producer_id: str
    timestamp: float  # time.time() at emission, seconds
    values: np.ndarray  # shape (n_features,)
    is_synthetic_anomaly: bool = False
    was_sanitized: bool = False  # True if NaN/Inf channel(s) were imputed at unpacking time


@dataclass
class StreamProducerConfig:
    n_features: int = 20
    sample_rate_hz: float = 200.0  # target packets/sec for this producer
    jitter_frac: float = 0.15  # +/- fractional jitter applied to inter-arrival time
    anomaly_probability: float = 0.0  # probability a packet is a synthetic spike anomaly
    anomaly_magnitude: float = 6.0  # std-devs of the injected spike
    seed: int = 42
    # "block" applies natural backpressure (await queue.put); "drop_oldest" evicts the
    # queue head to make room for fresh telemetry; "drop_new" discards the incoming
    # packet. Both drop modes catch asyncio.QueueFull explicitly rather than letting a
    # burst propagate an unhandled exception out of the producer loop.
    overflow_policy: str = "block"


class SensorStreamProducer:
    """
    Simulates one continuously-emitting multi-sensor rig.

    Without a `source`, values are synthesized as a per-channel sine wave
    (steady-state operating cycle) plus Gaussian sensor noise, so downstream
    STL/DMD engines see a plausible seasonal-trend signal rather than pure
    noise. Pass `source` (e.g. rows from a benchmark loader) to replay real
    data instead.
    """

    def __init__(
        self,
        producer_id: str,
        config: Optional[StreamProducerConfig] = None,
        source: Optional[Sequence[np.ndarray]] = None,
    ) -> None:
        self.producer_id = producer_id
        self.config = config or StreamProducerConfig()
        if self.config.overflow_policy not in ("block", "drop_oldest", "drop_new"):
            raise ValueError(
                f"overflow_policy must be 'block', 'drop_oldest', or 'drop_new', "
                f"got {self.config.overflow_policy!r}"
            )
        self._source = source
        self._rng = np.random.default_rng(self.config.seed)
        self._phase = self._rng.uniform(0, 2 * np.pi, size=self.config.n_features)
        self._freq = self._rng.uniform(0.05, 0.2, size=self.config.n_features)
        self._sequence_id = 0
        self._dropped = 0
        # Causal running mean per channel (EMA), used only to impute NaN/Inf
        # readings -- never peeks at future samples.
        self._running_mean = np.zeros(self.config.n_features)
        self._running_mean_ready = False

    @property
    def dropped_count(self) -> int:
        return self._dropped

    def _synthesize(self, t: int) -> np.ndarray:
        seasonal = np.sin(self._freq * t + self._phase)
        noise = self._rng.normal(0.0, 0.05, size=self.config.n_features)
        return seasonal + noise

    def _sanitize(self, vec: np.ndarray) -> tuple[np.ndarray, bool]:
        """Impute non-finite (NaN/Inf) sensor channels at the ingestion boundary
        -- the "unpacking" step -- so no downstream matrix operation ever sees
        a NaN. Uses a causal running mean per channel rather than 0.0 so an
        imputed reading doesn't itself look like an anomalous spike."""
        bad = ~np.isfinite(vec)
        had_bad = bool(bad.any())
        if had_bad:
            vec = vec.copy()
            fill = self._running_mean if self._running_mean_ready else 0.0
            vec[bad] = fill[bad] if isinstance(fill, np.ndarray) else fill
            logger.debug(
                "producer %s: sanitized %d non-finite channel(s) at seq=%d",
                self.producer_id, int(bad.sum()), self._sequence_id,
            )
        observed = np.where(bad, self._running_mean, vec)
        if not self._running_mean_ready:
            self._running_mean = observed
            self._running_mean_ready = True
        else:
            self._running_mean = 0.99 * self._running_mean + 0.01 * observed
        return vec, had_bad

    def _next_values(self) -> tuple[np.ndarray, bool, bool]:
        if self._source is not None:
            vec = np.asarray(
                self._source[self._sequence_id % len(self._source)], dtype=np.float64
            )
        else:
            vec = self._synthesize(self._sequence_id)

        vec, was_sanitized = self._sanitize(vec)

        is_anomaly = False
        if (
            self.config.anomaly_probability > 0
            and self._rng.random() < self.config.anomaly_probability
        ):
            vec = vec.copy()
            spike_channel = self._rng.integers(0, len(vec))
            vec[spike_channel] += self.config.anomaly_magnitude * self._rng.choice([-1.0, 1.0])
            is_anomaly = True
        return vec, is_anomaly, was_sanitized

    async def _enqueue(self, queue: "asyncio.Queue[SensorPacket]", packet: SensorPacket) -> None:
        """
        Apply this producer's overflow policy. "block" relies on
        `asyncio.Queue`'s own bounded backpressure (awaiting `put` only
        suspends this producer task, never the event loop). The drop modes
        explicitly catch `asyncio.QueueFull` from `put_nowait` -- a burst
        that would otherwise raise out of the producer loop is instead a
        handled, counted, logged event.
        """
        if self.config.overflow_policy == "block":
            await queue.put(packet)
            return
        try:
            queue.put_nowait(packet)
        except asyncio.QueueFull:
            if self.config.overflow_policy == "drop_oldest":
                try:
                    queue.get_nowait()  # evict the stalest packet to make room
                except asyncio.QueueEmpty:
                    pass
                try:
                    queue.put_nowait(packet)
                except asyncio.QueueFull:
                    self._dropped += 1  # consumer is fully saturated; drop this one too
            else:  # "drop_new"
                self._dropped += 1
            # DEBUG per-event, not WARNING: under a real burst this can fire thousands
            # of times/sec, and log I/O itself would become the bottleneck. `run()`
            # emits one summary line with the final dropped_total when it exits, and
            # `dropped_count` is available to callers (e.g. the stress-test harness)
            # at any time without relying on log parsing.
            logger.debug(
                "producer %s: queue overflow (policy=%s), dropped_total=%d",
                self.producer_id, self.config.overflow_policy, self._dropped,
            )

    async def run(
        self,
        queue: "asyncio.Queue[SensorPacket]",
        max_messages: Optional[int] = None,
        stop_event: Optional[asyncio.Event] = None,
    ) -> int:
        """
        Emit packets onto `queue` until `max_messages` is reached or
        `stop_event` is set. Never blocks the event loop: enqueueing is
        either an awaited `put` (suspends only this task while the queue is
        full) or a non-blocking `put_nowait` under a drop policy, and
        inter-arrival waits use `asyncio.sleep`, never `time.sleep`.
        """
        base_interval = 1.0 / self.config.sample_rate_hz
        emitted = 0
        while max_messages is None or emitted < max_messages:
            if stop_event is not None and stop_event.is_set():
                break

            values, is_anomaly, was_sanitized = self._next_values()
            packet = SensorPacket(
                sequence_id=self._sequence_id,
                producer_id=self.producer_id,
                timestamp=time.time(),
                values=values,
                is_synthetic_anomaly=is_anomaly,
                was_sanitized=was_sanitized,
            )
            await self._enqueue(queue, packet)
            self._sequence_id += 1
            emitted += 1

            jitter = self._rng.uniform(-self.config.jitter_frac, self.config.jitter_frac)
            await asyncio.sleep(max(0.0, base_interval * (1 + jitter)))

        logger.info(
            "producer %s: emitted %d packets (dropped=%d)",
            self.producer_id, emitted, self._dropped,
        )
        return emitted


async def run_producers(
    queue: "asyncio.Queue[SensorPacket]",
    producers: Sequence[SensorStreamProducer],
    max_messages_per_producer: Optional[int] = None,
    stop_event: Optional[asyncio.Event] = None,
) -> int:
    """Fan multiple concurrent sensor rigs into one shared queue."""
    results = await asyncio.gather(
        *(p.run(queue, max_messages_per_producer, stop_event) for p in producers)
    )
    return sum(results)
