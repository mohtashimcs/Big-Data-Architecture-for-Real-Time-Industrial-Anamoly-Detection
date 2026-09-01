# Scalable Big Data Architecture for Real-Time Industrial Anomaly Detection in Multivariate Time-Series

A modular, Strategy-pattern Python pipeline for real-time multivariate time-series (MTS)
anomaly detection, built to hold a **sub-20 ms per-window processing budget** on a single
CPU core with no GPU.

The project's premise is that the streaming system — not just the model — is the object
of study. Published MTS anomaly-detection results are almost always offline batch metrics
computed on a shuffled test set, decoupled from the architecture that would have to deliver
them in production. Here, latency, backpressure behaviour and numerical robustness are
measured, SLA-bound properties of the delivered system.

A non-blocking `asyncio` / thread-pool ingestion layer with explicit backpressure and
bounded in-flight concurrency feeds two interchangeable analytical engines behind one
`BaseAnomalyDetector` interface:

- **Engine 1 — mathematical fast-track:** a closed-form, rank-*r* truncated-SVD Dynamic
  Mode Decomposition (DMD) operator plus per-channel Seasonal-Trend (STL) residual energy.
  No gradient descent anywhere in `fit()`.
- **Engine 2 — deep reconstruction:** a PyTorch LSTM encoder–decoder or Dense
  flattened-window autoencoder, trained on normal-operation windows only.

Detections come from an evaluation engine that fuses a reference-centroid Euclidean-distance
term with each engine's own localised error through an epsilon-guarded harmonic mean, under a
dynamically recalibrating threshold.

---

## Results at a glance

Streamed in temporal order, no point adjustment, no threshold chosen with knowledge of the
labels. All figures below come from `industrial_anomaly_pipeline/benchmarks/results/experiment_results.json`.

| Dataset | Engine | F1 | AUC-ROC | Precision | Recall | p50 (ms) | p99 (ms) | SLA violations |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| NASA C-MAPSS | DMD + STL | 0.077 | 0.647 | 1.000 | 0.040 | 0.339 | 0.799 | 0 / 2,614 |
| NASA C-MAPSS | LSTM AE | 0.194 | 0.597 | 0.850 | 0.109 | 1.630 | 3.034 | 0 / 2,614 |
| NASA C-MAPSS | Dense AE | 0.194 | 0.597 | 0.850 | 0.109 | 0.306 | 0.659 | 0 / 2,614 |
| TCM5 (synthetic) | DMD + STL | 0.680 | 1.000 | 0.515 | 1.000 | 0.127 | 0.355 | 0 / 595 |
| TCM5 (synthetic) | LSTM AE | 0.716 | 0.993 | 0.566 | 0.976 | 1.528 | 2.381 | 0 / 595 |
| TCM5 (synthetic) | Dense AE | 0.717 | 0.997 | 0.570 | 0.964 | 0.268 | 0.618 | 0 / 595 |

**Zero SLA violations across all 6,418 scored windows** at nominal throughput. Initialisation
runs 0.035–0.169 s for the fast-track engine against 0.358–13.171 s for the autoencoders.

Under a deliberately adversarial burst (8 producers targeting 5,000 Hz each, far beyond
provisioned capacity) the system sheds load rather than failing, and the two engines diverge:

| Dataset | Engine | Throughput (pkt/s) | Dropped | p99 (ms) | SLA violations |
|---|---|---:|---:|---:|---:|
| C-MAPSS | DMD + STL | 5,620 | 0.0 % | 21.99 | 4 / 160 |
| C-MAPSS | LSTM AE | 4,745 | 64.6 % | 238.95 | 18 / 56 |
| TCM5 | DMD + STL | 6,012 | 0.0 % | 2.41 | 0 / 160 |
| TCM5 | LSTM AE | 4,178 | 59.7 % | 205.45 | 60 / 64 |

The fast-track engine absorbs the entire burst with zero drops; the LSTM autoencoder sheds
roughly two thirds of offered load *and* still breaches the SLA on most of what it does score.
That divergence, once the handler itself becomes the bottleneck, is the empirical case for
keeping a lightweight fallback engine available.

Latency and throughput figures are hardware-dependent and come from a single unrepeated run.
The relative ordering between engines is the reproducible part; the accuracy metrics are
deterministic given the fixed seeds.

---

## Repository layout

```
industrial_anomaly_pipeline/
  analytics/
    base.py                 # BaseAnomalyDetector Strategy interface; latency-instrumented scoring
    dmd_decomposer.py       # Engine 1: STL baseline + closed-form DMD operator fit
    autoencoder.py          # Engine 2: LSTM / Dense autoencoders, guarded inference, int8 hook
  evaluation/
    metrics_engine.py       # k-means++ centroids, harmonic composite, DynamicThresholder, metrics
  ingestion/
    stream_producer.py      # SensorStreamProducer, overflow policies, boundary sanitisation
    stream_consumer.py      # Windowing consumer, bounded in-flight dispatch
    benchmark_loader.py     # LoadedDataset schema; C-MAPSS, SMAP/MSL, MVTec, TCM5 loaders
  benchmarks/
    run_experiments.py      # Full harness: fit -> calibrate -> stream -> stress test -> results
    test_latency.py         # LatencyTracker plus six SLA regression tests
    visualize_results.py    # Regenerates every figure from the results file
    results/experiment_results.json
  tests/                    # 51 of the 57 regression tests (the other 6 are SLA tests above)
  paper/                    # Research paper sources, figures and generation scripts
  data/nasa_cmapss/         # C-MAPSS FD001-FD004 (ships with the repo)
  main.py                   # CLI runner wiring producers -> queue -> consumer -> detector
configs/config.yaml         # Retained from the earlier offline phase; see the note below
```

> **Note on `configs/config.yaml`.** It is a leftover from an earlier offline Phase 1/2
> pipeline whose `src/` code is no longer part of this repository. No module in the current
> pipeline reads it — `benchmarks/run_experiments.py` takes its window size, stride and
> engine hyperparameters from CLI flags and in-code defaults. It is kept for reference only.

---

## Getting started

```bash
pip install -r industrial_anomaly_pipeline/requirements.txt
```

The NASA C-MAPSS turbofan degradation benchmark ships with the repository under
`industrial_anomaly_pipeline/data/nasa_cmapss/`. The synthetic five-channel Tool Condition
Monitoring stream (TCM5) is generated from a fixed seed and needs no download, so the latency
and stress experiments are reproducible anywhere.

Two further sources are implemented in the loader but not bundled:

- **NASA SMAP / MSL** — supported against the standard `telemanom` per-channel layout, but
  licence-gated. Place the extracted contents under `data/smap_msl/` and the loader will pick
  them up; otherwise it raises with setup instructions rather than substituting other data.
- **MVTec-AD** — an image-patch visual-defect benchmark, not representable as an MTS window
  stream. The loader documents the mismatch and raises rather than reshaping pixels into a
  fake sensor vector.

### Run the full benchmark suite

```bash
cd industrial_anomaly_pipeline

python benchmarks/run_experiments.py            # full run, writes benchmarks/results/experiment_results.json
python benchmarks/run_experiments.py --quick    # smaller run for a fast smoke test
python benchmarks/run_experiments.py --skip-stress

python benchmarks/visualize_results.py          # regenerate every figure from the results file
```

For each (dataset, engine) pair the harness fits the engine on normal-operation windows,
calibrates the composite scoring node on a held-out normal split, then streams the labelled
test split through **in temporal order**, recording per-window latency and classification
outcome exactly as a deployed system would encounter them. A separate harness drives the real
`asyncio` pipeline (producers → bounded queue → consumer → engine) well beyond provisioned
capacity to measure throughput, drop rate and latency stability under load.

### Exercise the ingestion layer end to end

```bash
python main.py --n-producers 5 --n-features 20 --messages-per-producer 500
```

`main.py` wires the producers, queue and consumer to a minimal baseline detector purely to
demonstrate the ingestion path; the production engines live in `analytics/`.

---

## Testing

```bash
cd industrial_anomaly_pipeline
python -m pytest tests/ benchmarks/test_latency.py -v
```

57 test functions across five modules. Each maps to a specific behaviour or a specific
reliability finding, so a regression re-surfaces the exact defect it guards against:

| Module | Tests | Coverage |
|---|---:|---|
| `tests/test_ingestion.py` | 16 | Producer contracts, overflow policies, windowing and stride, bounded in-flight dispatch as real backpressure, handler-exception isolation, thread-safe aggregation, NaN/Inf sanitisation |
| `tests/test_analytics.py` | 17 | Both engines under the SLA, shape and channel guards, NaN imputation, zero-variance channels, no-autograd-graph inference, determinism, Dense vs LSTM window-length behaviour |
| `tests/test_evaluation.py` | 17 | Harmonic-mean singularities, threshold behaviour before sufficient history, the anomaly-cannot-drag-the-baseline rule, recovery from sustained drift, degenerate centroid inputs |
| `benchmarks/test_latency.py` | 6 | SLA assertions per engine, relative Dense/LSTM cost, end-to-end composite scoring, quantised-inference non-regression |
| `tests/test_visualize_results.py` | 1 | All four result figures render to non-empty files |

Several tests assert on **timing** rather than state, because the defects they guard against
are temporal — a state-only assertion would pass against both the correct and the defective
implementation.

---

## Design notes

**Why bound in-flight dispatch.** A `ThreadPoolExecutor`'s internal work queue has no size
limit. Submitting to it unconditionally as packets arrive lets queued-but-not-yet-run handler
calls — each holding a reference to its own window array — accumulate without bound whenever
handlers cannot keep pace. Nothing raises and nothing logs; the process grows until it is
killed. Bounding in-flight tasks with an `asyncio.Semaphore` converts that silent failure into
a visible one: a saturated pool suspends dispatch, which stops intake, which fills the queue,
which finally backpressures the producers — one coherent, testable chain.

**Why DMD for the fast-track engine.** It is a closed-form linear-algebra fit (an SVD plus a
few matrix products), so there is no training loop and initialisation is bounded by matrix
factorisation rather than an optimiser. Per-window inference is a single small matrix product,
which is what keeps p99 latency in the sub-millisecond range regardless of window length.

**Why the harmonic mean for fusion.** `2·d·e / (d + e + ε)` is dominated by the smaller of its
two inputs, so in principle a window scores low only if it is *both* close to a known-normal
operating regime and well predicted or reconstructed — neither signal alone can mask an
anomaly. The `ε` turns the otherwise-undefined `d = e = 0` case into a safe zero. See the known
limitation below for where this argument breaks down in practice.

**Why only non-flagged scores update the threshold baseline** — and why that needed an escape
hatch. Letting every score update the rolling baseline would allow a sustained genuine anomaly
to drag the boundary up until it stopped flagging itself. But excluding flagged scores creates
the opposite failure: if the stream's normal operating point drifts after calibration, every
post-drift normal score sits above the stale threshold, none qualifies to update the baseline,
and the threshold never recovers. After 50 consecutive flags the thresholder treats the run as
drift rather than one long anomaly and force-recalibrates. In a controlled before/after test
this alone raised the DMD engine's TCM5 F1 from 0.249 to 0.391.

**Window-level labelling policy.** A window inherits the anomalous label if *any* reading
inside it is anomalous. This is standard in the industrial CPS literature and favours recall on
partial-window anomalies, but it is a modelling choice that materially affects reported F1 — a
longer window mechanically raises the anomaly rate.

**C-MAPSS train-split correction.** C-MAPSS runs every unit to failure, so the raw training
split used as the "normal" baseline silently contained the same near-failure degradation the
test split labels anomalous. The loader trims the same trailing 30-cycle horizon from training
that defines an anomaly in test, restoring disjoint support between the two classes.

---

## Known limitations

**The composite score does not fuse its two terms as designed.** The harmonic mean is dominated
by its smaller input, which is only a *fusion* when both inputs occupy comparable ranges. They
do not. The autoencoders are trained on raw, unscaled sensor values — C-MAPSS channels span
roughly 10⁻³ to 9,065 — so their reconstruction error lands at 10⁵–10⁷ while the raw-space
centroid distance sits at about 7.7. In four of six (dataset, engine) configurations the
composite score correlates 1.0000 with one term and effectively ignores the other, which is why
the LSTM and Dense autoencoders return bit-identical metrics on C-MAPSS: neither model's output
reaches the decision. The correction — standardise both terms against their own
normal-validation distributions before fusing, and fit a MinMax scaler on normal data only in
the streaming path — is specified but **not yet implemented**.

**Recall is the binding constraint on the realistic benchmark.** At 4–11 % recall on C-MAPSS,
the system in its current configuration would miss most near-failure windows. It is decision
support with a human in the loop, not a protective function.

**Other bounds.** Timing comes from a single unrepeated run on one machine. External validity
rests on two sources, one of them synthetic. The stress test drives the real `asyncio` pipeline
but excludes network, serialisation and broker effects entirely, so the graceful-degradation
result is a claim about this pipeline's internal behaviour rather than an end-to-end deployment.

---

## Documents

- **Research paper** — `industrial_anomaly_pipeline/paper/` (Markdown, HTML, DOCX and PDF, plus
  the scripts that generate them).
- **Supporting material** — context and literature, project life cycle and tools, verification
  and traceability, results, professional/ethical/social/sustainability considerations, and a
  critical appraisal with the composite-score analysis above.

Every figure in both documents is regenerated from
`benchmarks/results/experiment_results.json` by `benchmarks/visualize_results.py`, so plots stay
consistent with a fresh experimental run rather than being hand-maintained.

---

## Key references

- Hundman et al., *Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic
  Thresholding*, KDD 2018.
- Malhotra et al., *Long Short Term Memory Networks for Anomaly Detection in Time Series*,
  ESANN 2015.
- Schmid, *Dynamic mode decomposition of numerical and experimental data*, JFM 2010.
- Cleveland et al., *STL: A Seasonal-Trend Decomposition Procedure Based on Loess*,
  J. Official Statistics 1990.
- Saxena et al., *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation*,
  PHM 2008 — the C-MAPSS benchmark.
- Kim et al., *Towards a Rigorous Evaluation of Time-Series Anomaly Detection*, AAAI 2022.
- Wu & Keogh, *Current Time Series Anomaly Detection Benchmarks are Flawed and are Creating the
  Illusion of Progress*, IEEE TKDE 2023.
