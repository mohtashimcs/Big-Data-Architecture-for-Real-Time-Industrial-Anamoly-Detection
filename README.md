# IIoT Anomaly Detection — Phase 1 & Phase 2

## 1. Setup (Phase 1 / Step 1)

```bash
pip install -r requirements.txt --break-system-packages
```

## 2. Get the data (Phase 1 / Step 2)
```
data/
  nasa_cmapss/
    train_FD001.txt
    test_FD001.txt
    RUL_FD001.txt
  swat/
    SWaT_Dataset_Normal_v1.csv
    SWaT_Dataset_Attack_v0.csv
```

- NASA C-MAPSS: https://www.kaggle.com/datasets/bishals098/nasa-turbofan-engine-degradation-simulation
- SWaT: https://www.kaggle.com/datasets/vishala28/swat-dataset-secure-water-treatment-system

## 3. Run Phase 1 + Phase 2 end-to-end

```bash
# NASA C-MAPSS (engine degradation -> "near-failure window" = anomaly)
python src/train_offline.py --dataset nasa --config configs/config.yaml

# SWaT (native Normal/Attack labels)
python src/train_offline.py --dataset swat --config configs/config.yaml
```

This will, per the roadmap:

1. **(Step 2-3)** Load, clean (forward/back-fill + median fallback for
   missing values), MinMax-scale (fit on normal data only), and window the
   chosen dataset into normal-train / normal-val / labeled-test splits.
2. **(Step 4)** Train a PyTorch LSTM (or Dense) Autoencoder exclusively on
   normal windows, with early stopping on a held-out normal validation split.
3. **(Step 5)** Fit the fast mathematical backend: per-channel
   Seasonal-Trend decomposition (STL) residual baseline + a closed-form
   Dynamic Mode Decomposition (DMD) linear operator over snapshot windows —
   no gradient descent, so it initializes almost instantly.
4. **(Step 6)** Build the scoring node: combine a z-normalized reconstruction
   error and the z-normalized **harmonic mean of Euclidean distance** (each
   in-window vector vs. a set of normal reference centroids), threshold
   calibrated from the normal validation split's score distribution.
5. Print a quick offline F1 / AUCROC comparison between the two backends
   (a preview of the full Phase 4 benchmarking) and save all model
   artifacts to `outputs/<dataset>/` for the Phase 3 streaming pipeline to
   load.

## File map

```
configs/config.yaml          # all paths + hyperparameters in one place
src/data_loader.py            # Step 2: dataset-specific loaders -> common schema
src/preprocessing.py          # Step 2-3: MinMax scaling (fit on normal only) + windowing
src/models/autoencoder.py     # Step 4: PyTorch LSTM / Dense Autoencoder
src/models/fast_backend.py    # Step 5: STL baseline + DMD operator
src/models/scoring.py         # Step 6: reconstruction error + harmonic-mean-distance scoring node
src/train_offline.py          # orchestrates Steps 2-6 end to end, saves artifacts
```

## Design notes worth citing in your Methodology chapter

- **Why fit the scaler on normal data only**: prevents anomaly statistics
  from leaking into normalization, which would otherwise make the
  reconstruction-error baseline look artificially better than it would be
  in a true streaming deployment where you never see future anomalies in
  advance.
- **Why harmonic mean (not arithmetic mean) for the distance component**:
  the harmonic mean is dominated by the smallest input values, so a window's
  distance score only stays low if it's close to *every* reference
  operating-regime centroid that contributes meaningfully — a single nearby
  centroid can't mask a genuinely anomalous vector the way an arithmetic
  mean would. This makes the distance term stricter and regime-aware.
- **Why DMD for the fast backend**: it's a closed-form linear-algebra fit
  (SVD + matrix multiplies), so there's no training loop, which is what lets
  the fast backend be benchmarked head-to-head against the AE on
  initialization cost as well as inference latency in Phase 4.
- **Window-level vs point-level labels**: a window inherits the anomalous
  label if *any* reading inside it is anomalous — a conservative choice that
  favors recall on partial-window anomalies, standard in industrial CPS
  anomaly-detection literature; worth flagging as a labeling-policy decision
  in your Methodology chapter since it affects reported F1.
