from __future__ import annotations

import asyncio

import numpy as np
import pytest

from analytics.base import AnomalyResult, BaseAnomalyDetector
from ingestion.stream_consumer import StreamConsumer, WindowMeta
from ingestion.stream_producer import (
    SensorPacket,
    SensorStreamProducer,
    StreamProducerConfig,
    run_producers,
)


class _MeanNormDetector(BaseAnomalyDetector):
    """Trivial concrete detector used only to exercise BaseAnomalyDetector."""

    def fit(self, normal_windows: np.ndarray) -> "_MeanNormDetector":
        self._is_fitted = True
        return self

    def _raw_score(self, window: np.ndarray) -> tuple[float, dict[str, float]]:
        score = float(np.mean(np.linalg.norm(window, axis=1)))
        return score, {"mean_norm": score}


# --------------------------------------------------------------------------- #
# BaseAnomalyDetector
# --------------------------------------------------------------------------- #
def test_base_detector_requires_fit_before_scoring():
    detector = _MeanNormDetector(name="test")
    with pytest.raises(RuntimeError):
        detector.score_window(np.zeros((5, 3)))


def test_base_detector_score_window_reports_latency_and_threshold():
    detector = _MeanNormDetector(name="test").fit(np.zeros((1, 5, 3)))
    detector.set_threshold(0.5)

    result = detector.score_window(np.ones((5, 3)))
    assert isinstance(result, AnomalyResult)
    assert result.latency_ms >= 0.0
    assert result.score > 0.5
    assert result.is_anomaly is True
    assert "mean_norm" in result.components


# --------------------------------------------------------------------------- #
# Stream producer
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_producer_emits_requested_message_count_with_correct_shape():
    queue: asyncio.Queue = asyncio.Queue()
    producer = SensorStreamProducer(
        "rig-0", StreamProducerConfig(n_features=6, sample_rate_hz=1_000_000)
    )

    emitted = await producer.run(queue, max_messages=25)

    assert emitted == 25
    assert queue.qsize() == 25
    packet: SensorPacket = queue.get_nowait()
    assert packet.values.shape == (6,)
    assert packet.producer_id == "rig-0"


@pytest.mark.asyncio
async def test_producer_injects_synthetic_anomalies_when_configured():
    queue: asyncio.Queue = asyncio.Queue()
    producer = SensorStreamProducer(
        "rig-0",
        StreamProducerConfig(n_features=4, sample_rate_hz=1_000_000, anomaly_probability=1.0),
    )

    await producer.run(queue, max_messages=10)

    packets = [queue.get_nowait() for _ in range(10)]
    assert all(p.is_synthetic_anomaly for p in packets)


@pytest.mark.asyncio
async def test_run_producers_fans_multiple_rigs_into_shared_queue():
    queue: asyncio.Queue = asyncio.Queue()
    producers = [
        SensorStreamProducer(f"rig-{i}", StreamProducerConfig(sample_rate_hz=1_000_000))
        for i in range(3)
    ]

    total = await run_producers(queue, producers, max_messages_per_producer=10)

    assert total == 30
    assert queue.qsize() == 30


# --------------------------------------------------------------------------- #
# Stream consumer
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_consumer_windows_and_dispatches_to_handler():
    queue: asyncio.Queue = asyncio.Queue()
    producer = SensorStreamProducer(
        "rig-0", StreamProducerConfig(n_features=5, sample_rate_hz=1_000_000)
    )
    await producer.run(queue, max_messages=12)

    seen: list[tuple[str, np.ndarray, WindowMeta]] = []

    def handler(producer_id: str, window: np.ndarray, meta: WindowMeta) -> None:
        seen.append((producer_id, window, meta))

    consumer = StreamConsumer(window_size=5, stride=1, handler=handler, max_workers=2)
    received = await consumer.run(queue, max_messages=12)

    assert received == 12
    # 12 packets, window_size=5, stride=1 -> 8 completed windows
    assert len(seen) == 8
    producer_id, window, meta = seen[0]
    assert producer_id == "rig-0"
    assert window.shape == (5, 5)
    assert meta.start_sequence_id == 0
    assert meta.end_sequence_id == 4
    assert meta.ingest_latency_ms >= 0.0


@pytest.mark.asyncio
async def test_consumer_respects_stride():
    queue: asyncio.Queue = asyncio.Queue()
    producer = SensorStreamProducer(
        "rig-0", StreamProducerConfig(n_features=3, sample_rate_hz=1_000_000)
    )
    await producer.run(queue, max_messages=20)

    seen = []
    consumer = StreamConsumer(
        window_size=5, stride=5, handler=lambda *a: seen.append(a), max_workers=2
    )
    await consumer.run(queue, max_messages=20)

    # 20 packets, window_size=5, stride=5 -> 4 non-overlapping windows
    assert len(seen) == 4


@pytest.mark.asyncio
async def test_consumer_flags_windows_containing_synthetic_anomalies():
    queue: asyncio.Queue = asyncio.Queue()
    producer = SensorStreamProducer(
        "rig-0",
        StreamProducerConfig(n_features=3, sample_rate_hz=1_000_000, anomaly_probability=1.0),
    )
    await producer.run(queue, max_messages=5)

    seen: list[WindowMeta] = []
    consumer = StreamConsumer(
        window_size=5, stride=1, handler=lambda pid, w, m: seen.append(m), max_workers=1
    )
    await consumer.run(queue, max_messages=5)

    assert len(seen) == 1
    assert seen[0].contains_synthetic_anomaly is True


@pytest.mark.asyncio
async def test_end_to_end_producer_to_consumer_via_detector():
    queue: asyncio.Queue = asyncio.Queue(maxsize=50)
    stop_event = asyncio.Event()
    producers = [
        SensorStreamProducer(f"rig-{i}", StreamProducerConfig(n_features=4, sample_rate_hz=1_000_000))
        for i in range(2)
    ]

    detector = _MeanNormDetector(name="e2e").fit(np.zeros((1, 10, 4)))
    detector.set_threshold(1e9)  # never trips; we only care that scoring runs
    results: list[AnomalyResult] = []

    def handler(producer_id: str, window: np.ndarray, meta: WindowMeta) -> None:
        results.append(detector.score_window(window))

    consumer = StreamConsumer(window_size=10, stride=10, handler=handler, max_workers=2)
    consumer_task = asyncio.create_task(consumer.run(queue, stop_event=stop_event))

    emitted = await run_producers(queue, producers, max_messages_per_producer=30)
    stop_event.set()
    received = await consumer_task

    assert emitted == 60
    assert received == 60
    # each rig: 30 packets / window_size 10, stride 10 -> 3 windows; 2 rigs -> 6
    assert len(results) == 6
    assert all(r.latency_ms >= 0.0 for r in results)
