from __future__ import annotations

from pathlib import Path

from benchmarks.visualize_results import (
    plot_classification_accuracy,
    plot_nominal_latency,
    plot_stress_latency,
    plot_stress_throughput_drop,
)


def _latency(p50: float, p95: float, p99: float, max_ms: float) -> dict:
    return {"count": 10, "mean_ms": p50, "p50_ms": p50, "p95_ms": p95, "p99_ms": p99,
            "max_ms": max_ms, "sla_ms": 20.0, "sla_violation_rate": 0.0}


def _make_sample_results() -> dict:
    engines = ["dmd_stl_fast_track", "lstm_autoencoder", "dense_autoencoder"]
    datasets = ["nasa_cmapss", "tcm5"]
    classification = []
    for dataset in datasets:
        for engine in engines:
            classification.append({
                "dataset": dataset,
                "engine_type": engine,
                "fit_time_sec": 0.1,
                "metrics": {"f1": 0.5, "auc_roc": 0.8, "precision": 0.6, "recall": 0.7},
                "latency": _latency(0.3, 0.5, 0.8, 1.0),
            })

    stress = []
    for dataset in datasets:
        for engine in ("dmd_stl_fast_track", "lstm_autoencoder"):
            for scenario, p99 in (("moderate", 5.0), ("extreme", 50.0)):
                stress.append({
                    "dataset": dataset,
                    "engine_type": engine,
                    "scenario": scenario,
                    "throughput_pkts_per_sec": 1000.0,
                    "drop_rate": 0.1,
                    "latency": _latency(1.0, 2.0, p99, p99 * 1.2),
                })

    return {"sla_ms": 20.0, "classification_and_latency": classification, "stress_tests": stress}


def test_all_four_figures_render_to_nonempty_files(tmp_path: Path):
    data = _make_sample_results()

    targets = {
        "classification_accuracy.png": plot_classification_accuracy,
        "nominal_latency.png": plot_nominal_latency,
        "stress_latency.png": plot_stress_latency,
        "stress_throughput_drop.png": plot_stress_throughput_drop,
    }
    for filename, plot_fn in targets.items():
        out_path = tmp_path / filename
        plot_fn(data, out_path)
        assert out_path.exists()
        assert out_path.stat().st_size > 1000  # a real rendered PNG, not an empty/broken file
