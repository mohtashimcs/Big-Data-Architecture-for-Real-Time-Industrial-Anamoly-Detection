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


@dataclass
class StreamProducerConfig:
    n_features: int = 20
    sample_rate_hz: float = 200.0  # target packets/sec for this producer
    jitter_frac: float = 0.15  # +/- fractional jitter applied to inter-arrival time
    anomaly_probability: float = 0.0  # probability a packet is a synthetic spike anomaly
    anomaly_magnitude: float = 6.0  # std-devs of the injected spike
    seed: int = 42


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
        self._source = source
        self._rng = np.random.default_rng(self.config.seed)
        self._phase = self._rng.uniform(0, 2 * np.pi, size=self.config.n_features)
        self._freq = self._rng.uniform(0.05, 0.2, size=self.config.n_features)
        self._sequence_id = 0

    def _synthesize(self, t: int) -> np.ndarray:
        seasonal = np.sin(self._freq * t + self._phase)
        noise = self._rng.normal(0.0, 0.05, size=self.config.n_features)
        return seasonal + noise

    def _next_values(self) -> tuple[np.ndarray, bool]:
        if self._source is not None:
            vec = np.asarray(
                self._source[self._sequence_id % len(self._source)], dtype=np.float64
            )
        else:
            vec = self._synthesize(self._sequence_id)

        is_anomaly = False
        if (
            self.config.anomaly_probability > 0
            and self._rng.random() < self.config.anomaly_probability
        ):
            vec = vec.copy()
            spike_channel = self._rng.integers(0, len(vec))
            vec[spike_channel] += self.config.anomaly_magnitude * self._rng.choice([-1.0, 1.0])
            is_anomaly = True
        return vec, is_anomaly

    async def run(
        self,
        queue: "asyncio.Queue[SensorPacket]",
        max_messages: Optional[int] = None,
        stop_event: Optional[asyncio.Event] = None,
    ) -> int:
        """
        Emit packets onto `queue` until `max_messages` is reached or
        `stop_event` is set. Never blocks the event loop: `queue.put` is
        awaited (suspends only this task while the queue is full) and
        inter-arrival waits use `asyncio.sleep`, never `time.sleep`.
        """
        base_interval = 1.0 / self.config.sample_rate_hz
        emitted = 0
        while max_messages is None or emitted < max_messages:
            if stop_event is not None and stop_event.is_set():
                break

            values, is_anomaly = self._next_values()
            packet = SensorPacket(
                sequence_id=self._sequence_id,
                producer_id=self.producer_id,
                timestamp=time.time(),
                values=values,
                is_synthetic_anomaly=is_anomaly,
            )
            await queue.put(packet)
            self._sequence_id += 1
            emitted += 1

            jitter = self._rng.uniform(-self.config.jitter_frac, self.config.jitter_frac)
            await asyncio.sleep(max(0.0, base_interval * (1 + jitter)))

        logger.info("producer %s: emitted %d packets", self.producer_id, emitted)
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
