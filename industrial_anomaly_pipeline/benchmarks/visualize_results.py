"""
Regenerates the paper's figures from benchmarks/results/experiment_results.json.

Four figures, each answering one question the empirical section makes:
  1. classification_accuracy.png -- how accurate is each engine, per dataset?
  2. nominal_latency.png         -- how fast is each engine at normal throughput?
  3. stress_latency.png          -- what happens to that latency under a burst?
  4. stress_throughput_drop.png  -- how much load does the system shed under a burst?

Color follows engine identity consistently across every figure (never re-cycled):
DMD + STL fast-track = blue, LSTM Autoencoder = orange, Dense Autoencoder = aqua
-- the reference categorical palette's first three slots, chosen because they
validate all-pairs CVD-safe in both light and dark review (see the dataviz
skill's palette.md). A dashed reference line marks the 20ms SLA wherever
latency is plotted.

Usage:
    python benchmarks/visualize_results.py
    python benchmarks/visualize_results.py --results path/to/other_results.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = _PACKAGE_ROOT / "benchmarks" / "results" / "experiment_results.json"
FIGURES_DIR = _PACKAGE_ROOT / "paper" / "figures"

# Fixed categorical color assignment, held constant across every figure in this file.
ENGINE_COLOR = {
    "dmd_stl_fast_track": "#2a78d6",   # categorical slot 1 (blue)
    "lstm_autoencoder": "#eb6834",     # categorical slot 2 (orange)
    "dense_autoencoder": "#1baf7a",    # categorical slot 3 (aqua)
}
ENGINE_LABEL = {
    "dmd_stl_fast_track": "DMD + STL",
    "lstm_autoencoder": "LSTM AE",
    "dense_autoencoder": "Dense AE",
}
DATASET_LABEL = {"nasa_cmapss": "NASA C-MAPSS", "tcm5": "TCM5 (synthetic)"}

# Chart chrome, from the dataviz skill's reference palette (light chart surface).
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SLA_LINE = "#d03b3b"  # status-critical red; not used as a series color in this file
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "text.color": INK_PRIMARY,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_SECONDARY,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def _style_axis(ax, ylabel: str, log: bool = False) -> None:
    ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=10)
    if log:
        ax.set_yscale("log")
    ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
    ax.tick_params(length=0)


def _sla_line(ax, sla_ms: float):
    """A dashed SLA reference line, labeled via the legend rather than a
    floating in-plot text annotation -- a fixed x/y text position risks
    colliding with whichever bar's value label happens to land nearby (see
    marks-and-anatomy.md's collision guidance); the legend has no such risk
    regardless of bar heights."""
    return ax.axhline(
        sla_ms, color=SLA_LINE, linestyle="--", linewidth=1.5, zorder=4,
        label=f"{sla_ms:.0f}ms SLA",
    )


def _bar_labels(ax, bars, fmt: str) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            fmt.format(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=7.5, color=INK_SECONDARY,
            zorder=5,
            bbox=dict(boxstyle="round,pad=0.15", facecolor=SURFACE, edgecolor="none"),
        )


def load_results(path: Path) -> dict:
    return json.loads(path.read_text())


# --------------------------------------------------------------------------- #
# Figure 1: classification accuracy
# --------------------------------------------------------------------------- #
def plot_classification_accuracy(data: dict, out_path: Path) -> None:
    rows = data["classification_and_latency"]
    datasets = sorted({r["dataset"] for r in rows}, key=lambda d: list(DATASET_LABEL).index(d))
    metrics = ["f1", "auc_roc", "precision", "recall"]
    metric_labels = ["F1", "AUC-ROC", "Precision", "Recall"]
    engines = list(ENGINE_COLOR)

    fig, axes = plt.subplots(1, len(datasets), figsize=(11, 4.5), sharey=True)
    x = np.arange(len(metrics))
    bar_width = 0.24

    for ax, dataset in zip(axes, datasets):
        by_engine = {r["engine_type"]: r for r in rows if r["dataset"] == dataset}
        for i, engine in enumerate(engines):
            r = by_engine.get(engine)
            if r is None:
                continue
            values = [r["metrics"][m] for m in metrics]
            offset = (i - (len(engines) - 1) / 2) * bar_width
            bars = ax.bar(
                x + offset, values, width=bar_width * 0.9,
                color=ENGINE_COLOR[engine], label=ENGINE_LABEL[engine],
                zorder=3,
            )
            _bar_labels(ax, bars, "{:.2f}")
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, fontsize=9.5)
        ax.set_title(DATASET_LABEL[dataset], fontsize=11, color=INK_PRIMARY, pad=10)
        ax.set_ylim(0, 1.12)
        _style_axis(ax, "Score" if dataset == datasets[0] else "")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.04),
        ncol=len(engines), frameon=False, fontsize=10,
    )
    fig.suptitle(
        "Classification accuracy by engine and dataset\n"
        "(composite score, dynamic threshold, streamed in temporal order)",
        fontsize=12, color=INK_PRIMARY, y=1.14,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 2: nominal (non-burst) latency percentiles
# --------------------------------------------------------------------------- #
def plot_nominal_latency(data: dict, out_path: Path) -> None:
    rows = data["classification_and_latency"]
    sla_ms = data["sla_ms"]
    datasets = sorted({r["dataset"] for r in rows}, key=lambda d: list(DATASET_LABEL).index(d))
    percentiles = ["p50_ms", "p95_ms", "p99_ms"]
    percentile_labels = ["p50", "p95", "p99"]
    engines = list(ENGINE_COLOR)

    fig, axes = plt.subplots(1, len(datasets), figsize=(11, 4.5), sharey=True)
    x = np.arange(len(percentiles))
    bar_width = 0.24
    engine_handles: dict[str, object] = {}

    for ax, dataset in zip(axes, datasets):
        by_engine = {r["engine_type"]: r for r in rows if r["dataset"] == dataset}
        for i, engine in enumerate(engines):
            r = by_engine.get(engine)
            if r is None:
                continue
            values = [r["latency"][p] for p in percentiles]
            offset = (i - (len(engines) - 1) / 2) * bar_width
            container = ax.bar(
                x + offset, values, width=bar_width * 0.9,
                color=ENGINE_COLOR[engine], label=ENGINE_LABEL[engine],
                zorder=3,
            )
            engine_handles.setdefault(engine, container)
            _bar_labels(ax, container, "{:.2f}")
        sla_handle = _sla_line(ax, sla_ms)
        ax.set_xticks(x)
        ax.set_xticklabels(percentile_labels, fontsize=9.5)
        ax.set_title(DATASET_LABEL[dataset], fontsize=11, color=INK_PRIMARY, pad=10)
        _style_axis(ax, "Latency, ms (log scale)" if dataset == datasets[0] else "", log=True)
        ax.set_ylim(0.05, 40)

    handles = [engine_handles[e] for e in engines] + [sla_handle]
    labels = [ENGINE_LABEL[e] for e in engines] + [f"{sla_ms:.0f}ms SLA"]
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.04),
        ncol=len(handles), frameon=False, fontsize=10,
    )
    fig.suptitle(
        "Per-window latency at nominal throughput\n(microsecond-precision, log scale)",
        fontsize=12, color=INK_PRIMARY, y=1.14,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 3: burst stress test -- p99 latency, moderate vs. extreme
# --------------------------------------------------------------------------- #
def plot_stress_latency(data: dict, out_path: Path) -> None:
    rows = data["stress_tests"]
    sla_ms = data["sla_ms"]
    datasets = sorted({r["dataset"] for r in rows}, key=lambda d: list(DATASET_LABEL).index(d))
    scenarios = ["moderate", "extreme"]
    engines = [e for e in ENGINE_COLOR if any(r["engine_type"] == e for r in rows)]

    fig, axes = plt.subplots(1, len(datasets), figsize=(10, 4.8), sharey=True)
    x = np.arange(len(scenarios))
    bar_width = 0.32
    engine_handles: dict[str, object] = {}

    for ax, dataset in zip(axes, datasets):
        subset = {(r["engine_type"], r["scenario"]): r for r in rows if r["dataset"] == dataset}
        for i, engine in enumerate(engines):
            values = [subset[(engine, s)]["latency"]["p99_ms"] for s in scenarios]
            offset = (i - (len(engines) - 1) / 2) * bar_width
            bars = ax.bar(
                x + offset, values, width=bar_width * 0.9,
                color=ENGINE_COLOR[engine], label=ENGINE_LABEL[engine],
                zorder=3,
            )
            engine_handles.setdefault(engine, bars)
            _bar_labels(ax, bars, "{:.1f}")
        sla_handle = _sla_line(ax, sla_ms)
        ax.set_xticks(x)
        ax.set_xticklabels(["Moderate\n(~1,000 pkts/s)", "Extreme\n(4,300-7,400 pkts/s)"], fontsize=9)
        ax.set_title(DATASET_LABEL[dataset], fontsize=11, color=INK_PRIMARY, pad=10)
        _style_axis(ax, "p99 latency, ms (log scale)" if dataset == datasets[0] else "", log=True)
        ax.set_ylim(0.2, 400)

    handles = [engine_handles[e] for e in engines] + [sla_handle]
    labels = [ENGINE_LABEL[e] for e in engines] + [f"{sla_ms:.0f}ms SLA"]
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.06),
        ncol=len(handles), frameon=False, fontsize=10,
    )
    fig.suptitle(
        "Burst stress test: p99 latency under moderate vs. extreme load\n"
        "the fast-track engine holds the SLA margin the autoencoder loses",
        fontsize=12, color=INK_PRIMARY, y=1.17,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 4: burst stress test -- throughput and drop rate
# --------------------------------------------------------------------------- #
def plot_stress_throughput_drop(data: dict, out_path: Path) -> None:
    rows = data["stress_tests"]
    datasets = sorted({r["dataset"] for r in rows}, key=lambda d: list(DATASET_LABEL).index(d))
    scenarios = ["moderate", "extreme"]
    engines = [e for e in ENGINE_COLOR if any(r["engine_type"] == e for r in rows)]

    group_labels, group_keys = [], []
    for dataset in datasets:
        for scenario in scenarios:
            group_labels.append(f"{DATASET_LABEL[dataset].split()[0]}\n{scenario}")
            group_keys.append((dataset, scenario))

    fig, (ax_throughput, ax_drop) = plt.subplots(1, 2, figsize=(12, 4.8))
    x = np.arange(len(group_keys))
    bar_width = 0.32

    lookup = {(r["dataset"], r["scenario"], r["engine_type"]): r for r in rows}
    for i, engine in enumerate(engines):
        offset = (i - (len(engines) - 1) / 2) * bar_width
        throughput = [lookup[(d, s, engine)]["throughput_pkts_per_sec"] for d, s in group_keys]
        drop_rate = [lookup[(d, s, engine)]["drop_rate"] * 100 for d, s in group_keys]
        ax_throughput.bar(
            x + offset, throughput, width=bar_width * 0.9,
            color=ENGINE_COLOR[engine], label=ENGINE_LABEL[engine], zorder=3,
        )
        ax_drop.bar(
            x + offset, drop_rate, width=bar_width * 0.9,
            color=ENGINE_COLOR[engine], label=ENGINE_LABEL[engine], zorder=3,
        )

    for ax, ylabel, title in (
        (ax_throughput, "Throughput, packets/sec", "Achieved throughput"),
        (ax_drop, "Drop rate, %", "Packets shed (drop_new policy)"),
    ):
        ax.set_xticks(x)
        ax.set_xticklabels(group_labels, fontsize=9)
        ax.set_title(title, fontsize=11, color=INK_PRIMARY, pad=10)
        _style_axis(ax, ylabel)

    handles, labels = ax_throughput.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.05),
        ncol=len(engines), frameon=False, fontsize=10,
    )
    fig.suptitle(
        "Burst stress test: elasticity under load\n"
        "moderate load is fully absorbed; the extreme burst is shed, not crashed",
        fontsize=12, color=INK_PRIMARY, y=1.18,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=str(RESULTS_PATH))
    parser.add_argument("--out-dir", default=str(FIGURES_DIR))
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"error: {results_path} not found -- run benchmarks/run_experiments.py first", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_results(results_path)

    plot_classification_accuracy(data, out_dir / "classification_accuracy.png")
    plot_nominal_latency(data, out_dir / "nominal_latency.png")
    plot_stress_latency(data, out_dir / "stress_latency.png")
    plot_stress_throughput_drop(data, out_dir / "stress_throughput_drop.png")

    print(f"figures written to {out_dir}")


if __name__ == "__main__":
    main()
