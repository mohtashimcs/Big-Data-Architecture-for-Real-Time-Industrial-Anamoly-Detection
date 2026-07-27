"""
Engine 2: Deep Reconstruction Model (PyTorch LSTM / Dense Autoencoder).

Trained exclusively on normal operational data so it learns to reconstruct
standard sensor patterns with minimal error; anomalies (unseen patterns)
reconstruct poorly, which is the signal `_raw_score` exposes as
`reconstruction_error`.

Every inference path here runs under `torch.no_grad()` and returns plain
Python floats (never a live tensor/graph reference) -- the fix for the
"PyTorch tensor leakage / memory spike" failure mode a naive streaming
integration would otherwise hit: calling a module's forward pass without
`no_grad()` builds a full autograd graph (activations retained for a
backward pass that never happens) on every single window, which is wasted
compute and a growing memory footprint under sustained high-throughput
inference.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from analytics.base import BaseAnomalyDetector

logger = logging.getLogger(__name__)


class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features: int, hidden_dim: int, latent_dim: int,
                 num_layers: int, dropout: float) -> None:
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=n_features, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(
            input_size=hidden_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_layer = nn.Linear(hidden_dim, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, window, n_features); window length is not fixed at
        # construction time, so this architecture tolerates variable-length
        # windows across calls (unlike the flattened Dense variant below).
        batch, window, _ = x.shape
        _, (h_n, _) = self.encoder(x)
        latent = self.to_latent(h_n[-1])
        seed = self.from_latent(latent)
        seed_seq = seed.unsqueeze(1).repeat(1, window, 1)
        decoded, _ = self.decoder(seed_seq)
        return self.output_layer(decoded)


class DenseAutoencoder(nn.Module):
    def __init__(self, n_features: int, window_size: int, hidden_dim: int,
                 latent_dim: int, dropout: float) -> None:
        super().__init__()
        in_dim = n_features * window_size
        self.window_size = window_size
        self.n_features = n_features
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, in_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        flat = x.reshape(batch, -1)
        out = self.decoder(self.encoder(flat))
        return out.reshape(batch, self.window_size, self.n_features)


@dataclass
class AutoencoderTrainHistory:
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    train_time_sec: float = 0.0
    epochs_run: int = 0


class AutoencoderDetector(BaseAnomalyDetector):
    """BaseAnomalyDetector adapter around the LSTM/Dense reconstruction models."""

    def __init__(
        self,
        architecture: str = "lstm",
        hidden_dim: int = 64,
        latent_dim: int = 16,
        num_layers: int = 2,
        dropout: float = 0.2,
        epochs: int = 30,
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        early_stopping_patience: int = 5,
        val_split: float = 0.1,
        device: str | None = None,
        seed: int = 42,
        name: str | None = None,
    ) -> None:
        if architecture not in ("lstm", "dense"):
            raise ValueError(f"Unknown autoencoder architecture {architecture!r}, expected 'lstm' or 'dense'")
        super().__init__(name=name or f"{architecture}-autoencoder")
        self.architecture = architecture
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.early_stopping_patience = early_stopping_patience
        self.val_split = val_split
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed

        self.model: nn.Module | None = None
        self.history: AutoencoderTrainHistory | None = None
        self._n_features: int | None = None
        self._window_size: int | None = None
        self._channel_mean: np.ndarray | None = None
        self._quantized = False

    # ------------------------------------------------------------------- #
    # Training
    # ------------------------------------------------------------------- #
    def fit(self, normal_windows: np.ndarray) -> "AutoencoderDetector":
        normal_windows = self._validate_fit_input(normal_windows)
        self._n_features = normal_windows.shape[2]
        self._window_size = normal_windows.shape[1]
        self._channel_mean = normal_windows.reshape(-1, self._n_features).mean(axis=0)

        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(len(normal_windows))
        n_val = max(1, int(len(normal_windows) * self.val_split))
        val_windows = normal_windows[idx[:n_val]]
        train_windows = normal_windows[idx[n_val:]] if len(idx) > n_val else normal_windows

        torch.manual_seed(self.seed)
        self.model = self._build_model().to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        train_loader = DataLoader(
            TensorDataset(torch.tensor(train_windows, dtype=torch.float32)),
            batch_size=self.batch_size, shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(torch.tensor(val_windows, dtype=torch.float32)),
            batch_size=self.batch_size, shuffle=False,
        )

        history = AutoencoderTrainHistory()
        best_val, best_state, patience_ctr = float("inf"), None, 0
        start = time.perf_counter()

        for epoch in range(self.epochs):
            self.model.train()
            running = 0.0
            for (batch,) in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(batch), batch)
                loss.backward()
                optimizer.step()
                running += loss.item() * batch.size(0)
            train_loss = running / len(train_loader.dataset)

            val_loss = self._eval_loss(val_loader, criterion)
            history.train_losses.append(train_loss)
            history.val_losses.append(val_loss)
            history.epochs_run = epoch + 1
            logger.debug("%s: epoch %d/%d train_loss=%.6f val_loss=%.6f",
                         self.name, epoch + 1, self.epochs, train_loss, val_loss)

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= self.early_stopping_patience:
                    logger.info("%s: early stopping at epoch %d", self.name, epoch + 1)
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        history.train_time_sec = time.perf_counter() - start
        self.model.eval()  # inference mode from here on; dropout etc. disabled
        self.history = history
        self._is_fitted = True
        logger.info("%s: fit in %.1fs (%d epochs), final val_loss=%.6f",
                    self.name, history.train_time_sec, history.epochs_run, best_val)
        return self

    def _eval_loss(self, loader: DataLoader, criterion: nn.Module) -> float:
        self.model.eval()
        running = 0.0
        with torch.no_grad():  # validation is inference-only: no graph needed
            for (batch,) in loader:
                batch = batch.to(self.device)
                running += criterion(self.model(batch), batch).item() * batch.size(0)
        return running / len(loader.dataset)

    def _build_model(self) -> nn.Module:
        if self.architecture == "lstm":
            return LSTMAutoencoder(
                n_features=self._n_features, hidden_dim=self.hidden_dim,
                latent_dim=self.latent_dim, num_layers=self.num_layers, dropout=self.dropout,
            )
        return DenseAutoencoder(
            n_features=self._n_features, window_size=self._window_size,
            hidden_dim=self.hidden_dim, latent_dim=self.latent_dim, dropout=self.dropout,
        )

    def _validate_fit_input(self, normal_windows: np.ndarray) -> np.ndarray:
        normal_windows = np.asarray(normal_windows, dtype=np.float32)
        if normal_windows.ndim != 3:
            raise ValueError(
                f"{self.name}: fit() expects (n_windows, window_size, n_features), "
                f"got shape {normal_windows.shape}"
            )
        if np.isnan(normal_windows).any():
            flat = normal_windows.reshape(-1, normal_windows.shape[-1])
            col_mean = np.nanmean(flat, axis=0)
            nan_mask = np.isnan(normal_windows)
            normal_windows = np.where(nan_mask, col_mean, normal_windows).astype(np.float32)
            logger.warning(
                "%s: fit() input contained NaNs; imputed %d cell(s) with per-channel means",
                self.name, int(nan_mask.sum()),
            )
        return normal_windows

    # ------------------------------------------------------------------- #
    # Inference (hot path)
    # ------------------------------------------------------------------- #
    def _prepare_window(self, window: np.ndarray) -> np.ndarray:
        window = np.asarray(window, dtype=np.float32)
        if window.ndim != 2:
            raise ValueError(
                f"{self.name}: expected a 2D window (window_size, n_features), got shape {window.shape}"
            )
        if window.shape[1] != self._n_features:
            raise ValueError(
                f"{self.name}: window has {window.shape[1]} sensor channel(s) but this "
                f"engine was fit on {self._n_features}; upstream ingestion must emit a "
                "fixed sensor schema, or realign/impute channels before scoring."
            )
        if self.architecture == "dense" and window.shape[0] != self._window_size:
            # The Dense AE flattens (window_size * n_features) at fit time, so unlike
            # the LSTM variant it cannot tolerate a variable window length at inference.
            raise ValueError(
                f"{self.name}: Dense autoencoder was fit on window_size={self._window_size}, "
                f"got {window.shape[0]}; use the 'lstm' architecture if window length "
                "varies across the stream, or pad/truncate upstream."
            )
        if np.isnan(window).any():
            window = window.copy()
            rows, cols = np.where(np.isnan(window))
            window[rows, cols] = self._channel_mean[cols]
            logger.debug(
                "%s: imputed %d missing-channel value(s) in-window using fitted means",
                self.name, len(rows),
            )
        return window

    def _raw_score(self, window: np.ndarray) -> tuple[float, dict[str, float]]:
        window = self._prepare_window(window)
        self.model.eval()  # idempotent guard: never score with dropout/BN in train mode
        with torch.no_grad():  # critical: no autograd graph retained across the streaming hot path
            x = torch.from_numpy(window).unsqueeze(0).to(self.device)
            recon = self.model(x)
            mse = torch.mean((recon - x) ** 2).item()
        return mse, {"reconstruction_error": mse}

    # ------------------------------------------------------------------- #
    # Optional lightweight-inference hooks
    # ------------------------------------------------------------------- #
    def quantize_dynamic(self) -> "AutoencoderDetector":
        """
        Post-training dynamic quantization (Linear/LSTM weights -> int8) for
        lower-latency, lower-memory CPU inference -- no calibration data
        needed, safe to call any time after fit(). No-op if already applied.
        """
        if not self._is_fitted:
            raise RuntimeError(f"{self.name} must be fit() before quantization")
        if self._quantized:
            return self
        import torch.quantization as tq

        self.model = tq.quantize_dynamic(
            self.model.to("cpu"), {nn.Linear, nn.LSTM}, dtype=torch.qint8
        )
        self.device = "cpu"
        self._quantized = True
        logger.info("%s: applied dynamic int8 quantization", self.name)
        return self

    def get_params(self) -> dict:
        params = super().get_params()
        params.update(
            architecture=self.architecture,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            num_layers=self.num_layers,
            n_features=self._n_features,
            window_size=self._window_size,
            quantized=self._quantized,
            train_time_sec=self.history.train_time_sec if self.history else None,
            epochs_run=self.history.epochs_run if self.history else None,
        )
        return params
