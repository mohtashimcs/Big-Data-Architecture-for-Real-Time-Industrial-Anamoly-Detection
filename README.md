# Scalable Big Data Architecture for Real-Time Industrial Anomaly Detection

COM748 Masters Research Project — Mohtashim Ali (B20081038)
Supervisor: Dr. Usman Butt

Complete implementation of all five project phases: offline modeling
(Phases 1–2), streaming architecture (Phase 3), SLA benchmarking
(Phase 4), and the written deliverables (Phase 5).

---

## Quick start

```bash
# 1. Install dependencies (use a venv if you prefer)
pip install -r requirements.txt

# 2. Verify everything works WITHOUT downloading any data
python src/selftest.py

# 3. Download datasets (see below), then run the real pipeline
python src/train_offline.py --dataset nasa      # Phases 1-2
python src/benchmark.py     --dataset nasa      # Phases 3-4
```

Run every command **from the project root** — config and data paths are
relative to it.

### `selftest.py` — run this first

Generates a synthetic multivariate sensor stream with injected anomalies and
exercises the entire pipeline (preprocessing → fast backend → scoring node →
streaming ingestion → SLA timing → stress test). It needs no Kaggle data, so
it isolates environment problems from data problems. If PyTorch is installed
it also trains a small autoencoder; otherwise it skips that path and says so.

Verified output on a clean environment:

```
[Phase 1] train=(716, 30, 6) val=(79, 30, 6) test=(295, 30, 6)  OK
[Phase 2] Fast backend fit in 0.06s (DMD operator (6, 6))  OK
[Phase 2] Fast backend offline: F1=0.876 AUCROC=1.000  OK
[Phase 3-4] fast_backend scored=295 mean=0.543ms p99=0.788ms SLA_violations=0.00%  OK
[Phase 4/Step 11] fast_backend produced=295 scored=10 dropped=285 throughput=31968 vec/s  OK
=== SELF-TEST PASSED ===
```

---

## Datasets

Download from Kaggle and place exactly as below:

```
data/nasa_cmapss/train_FD001.txt
data/nasa_cmapss/test_FD001.txt
data/nasa_cmapss/RUL_FD001.txt
data/swat/SWaT_Dataset_Normal_v1.csv
data/swat/SWaT_Dataset_Attack_v0.csv
```

- NASA C-MAPSS: <https://www.kaggle.com/datasets/bishals098/nasa-turbofan-engine-degradation-simulation>
- SWaT: <https://www.kaggle.com/datasets/vishala28/swat-dataset-secure-water-treatment-system>

If your download has different filenames or column headers, edit
`configs/config.yaml` rather than renaming code. Both loaders were tested
against synthetic files built to the documented formats (whitespace-separated
26-column C-MAPSS; SWaT CSV with `Timestamp` and `Normal/Attack` columns).

**SWaT is large (~500k rows at 1 Hz).** Three config knobs keep it tractable:
`swat.row_subsample` (default: every 5th row), `preprocessing.swat_window_stride`
(default 10), and `preprocessing.max_train_windows` / `max_test_windows`.
Without these, stride-1 windowing produces hundreds of thousands of windows
and exhausts memory.

---

## What each phase does

### Phase 1 — Environment & data preparation
| File | Role |
|---|---|
| `requirements.txt` | Step 1: NumPy/Pandas, PyTorch, statsmodels, scikit-learn |
| `src/data_loader.py` | Step 2: dataset-specific loaders → common schema; missing-value handling |
| `src/preprocessing.py` | Steps 2–3: MinMax scaling **fit on normal data only**, windowing, normal/test separation |

### Phase 2 — Core analytics
| File | Role |
|---|---|
| `src/models/autoencoder.py` | Step 4: LSTM/Dense autoencoder trained exclusively on normal windows, early stopping on a normal validation split |
| `src/models/fast_backend.py` | Step 5: VersaGuardian-style backend — STL residual baseline + closed-form DMD operator (single SVD, no gradient descent) |
| `src/models/scoring.py` | Step 6: scoring node fusing localized error with the harmonic mean of Euclidean distance to normal-regime centroids |
| `src/train_offline.py` | Orchestrates Phases 1–2; writes metrics and artifacts |

### Phase 3 — Streaming architecture
| File | Role |
|---|---|
| `src/streaming/ingestion.py` | Step 7: multi-threaded producer, bounded queue, blocking and drop-on-full overflow policies |
| `src/streaming/pipeline.py` | Step 8: consumer scoring one vector at a time; Step 9's timer wraps queue-exit → score-produced |

### Phase 4 — Benchmarking
| File | Role |
|---|---|
| `src/benchmark.py` | Steps 9–11: per-vector SLA latency, backend comparison, stress/elasticity burst test |

### Phase 5 — Written deliverables
| File | Role |
|---|---|
| `dissertation/dissertation_skeleton.docx` | Chapter skeleton aligned to the proposal, with `[TODO]` markers |
| `paper/paper.tex` + `paper_body.tex` | IEEE conference-format paper (compile on Overleaf) |
| `paper/paper_preview.pdf` | Locally-compiled preview of the paper |

---

## Where the numbers come from

| Output | File |
|---|---|
| Offline F1 / AUCROC, thresholds, train times | `outputs/<dataset>/metrics.json` |
| Per-window scores + predictions + labels (for ROC curves, plots) | `outputs/<dataset>/scores.csv` |
| Streaming latency, SLA violations, stress results | `outputs/<dataset>/benchmark_report.json` |
| Model artifacts for the streaming replay | `outputs/<dataset>/*.pt`, `*.pkl`, `model_meta.json` |

`benchmark.py` options: `--steady-hz` (production rate), `--max-vectors`
(cap per run, default 5000), `--stress-queue` (burst-test queue size).

---

## Design decisions worth defending in the viva

- **Scaler fit on normal data only.** Fitting on the full dataset would leak
  anomalous-region statistics into normalization and optimistically bias
  every reconstruction-error baseline — a subtle but real form of test-set
  leakage in unsupervised anomaly detection.
- **Harmonic mean, not arithmetic mean, for the distance term.** The harmonic
  mean is dominated by the smallest inputs, so a window scores low only if it
  sits close to *every* relevant normal operating regime. Under an arithmetic
  mean, proximity to one centroid could mask an anomalous vector.
- **Threshold calibrated from the normal validation split.** No anomaly labels
  are used anywhere in fitting or calibration, keeping the method genuinely
  unsupervised — which also addresses the proposal's identified data-sparsity
  risk.
- **Latency excludes queue wait time.** The 20 ms SLA targets the scoring
  engine's compute cost; queue delay is a property of the ingestion and
  back-pressure subsystem and is reported separately via queue high-watermark
  and drop rate.
- **Vectors scored one at a time, not batched.** Batching would amortize cost
  across vectors and understate true per-vector latency.
- **Window labeled anomalous if *any* reading in it is anomalous.** A
  conservative policy favouring recall; state it explicitly in Methodology
  since it affects reported F1.
- **Threads over processes.** The hot path is NumPy/PyTorch operations that
  release the GIL, so threads avoid the per-vector serialization cost that
  crossing process boundaries would add — the I/O bottleneck the proposal
  warns against.

---

## Known limitations (state these in the write-up)

1. **The autoencoder path has not been executed against real data here.** The
   code compiles and follows a standard LSTM encoder–decoder structure, but
   PyTorch could not be installed in the environment where this was built, so
   the training loop is unverified at runtime. The fast backend, scoring node,
   preprocessing, streaming pipeline, and both data loaders **were** executed
   and verified. Run `selftest.py` with PyTorch installed to close this gap.
2. **No benchmark numbers are included anywhere.** Every results table in the
   dissertation and paper is a placeholder. Fill them from your own runs — do
   not cite figures that were never measured.
3. **C-MAPSS anomaly labels are heuristic** (final N cycles before failure),
   unlike SWaT's native labels. This affects absolute F1 and cross-dataset
   comparability.
4. **Single-machine simulation**, not a distributed deployment. Latency
   figures will not transfer to production hardware without re-benchmarking.
5. **The fast backend is a good-faith reconstruction** of the STL+DMD approach
   described in the proposal's background research, not a validated
   VersaGuardian reference implementation.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `FileNotFoundError: Expected C-MAPSS file at ...` | Datasets not downloaded, or paths in `configs/config.yaml` don't match |
| `KeyError: Label column 'Normal/Attack' not found` | Your SWaT CSV uses different headers; update `swat.label_column` |
| `ModuleNotFoundError: No module named 'torch'` | `pip install torch` — required by `train_offline.py` and `benchmark.py` |
| MemoryError on SWaT | Increase `row_subsample` / `swat_window_stride`, or lower `max_train_windows` |
| `Missing artifacts in outputs/...` | Run `train_offline.py` before `benchmark.py` |
