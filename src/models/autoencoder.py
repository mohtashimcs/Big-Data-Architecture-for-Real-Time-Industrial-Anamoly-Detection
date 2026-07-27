"""
Phase 2 / Step 4: reconstruction-based Autoencoder in PyTorch.

Trained exclusively on normal operational data so it learns to reconstruct
standard sensor patterns with minimal error; anomalous windows fall outside
the learned normal manifold and reconstruct poorly, and that reconstruction
error is the raw anomaly signal consumed by the Step 6 scoring node.

Two architectures (configs/config.yaml -> autoencoder.type):
  lstm  : LSTM encoder-decoder, models temporal structure within the window
  dense : flattened-window dense AE, cheaper, useful as an ablation baseline
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features: int, hidden_dim: int, latent_dim: int,
                 num_layers: int, dropout: float):
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=n_features, hidden_size=hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(
            input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_layer = nn.Linear(hidden_dim, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, window, n_features)
        _, window, _ = x.shape
        _, (h_n, _) = self.encoder(x)
        latent = self.to_latent(h_n[-1])                  # (batch, latent_dim)
        seed = self.from_latent(latent)                    # (batch, hidden_dim)
        seed_seq = seed.unsqueeze(1).repeat(1, window, 1)   # broadcast across time
        decoded, _ = self.decoder(seed_seq)
        return self.output_layer(decoded)


class DenseAutoencoder(nn.Module):
    def __init__(self, n_features: int, window_size: int, hidden_dim: int,
                 latent_dim: int, dropout: float):
        super().__init__()
        self.window_size, self.n_features = window_size, n_features
        in_dim = n_features * window_size
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
        z = self.encoder(x.reshape(batch, -1))
        return self.decoder(z).reshape(batch, self.window_size, self.n_features)


def build_autoencoder(n_features: int, window_size: int, cfg: dict) -> nn.Module:
    ae = cfg["autoencoder"]
    if ae["type"] == "lstm":
        return LSTMAutoencoder(n_features, ae["hidden_dim"], ae["latent_dim"],
                                ae["num_layers"], ae["dropout"])
    if ae["type"] == "dense":
        return DenseAutoencoder(n_features, window_size, ae["hidden_dim"],
                                 ae["latent_dim"], ae["dropout"])
    raise ValueError(f"Unknown autoencoder type '{ae['type']}' (expected lstm|dense)")


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TrainResult:
    model: nn.Module
    train_losses: list
    val_losses: list
    train_time_sec: float


def train_autoencoder(train_windows: np.ndarray, val_windows: np.ndarray,
                       cfg: dict, device: str | None = None) -> TrainResult:
    """Step 4: train on normal data only, early-stopping on normal val MSE."""
    ae = cfg["autoencoder"]
    device = device or default_device()
    window_size, n_features = train_windows.shape[1], train_windows.shape[2]

    model = build_autoencoder(n_features, window_size, cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=ae["learning_rate"])
    criterion = nn.MSELoss()

    train_ds = TensorDataset(torch.tensor(train_windows, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(val_windows, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=ae["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=ae["batch_size"], shuffle=False)

    best_val, best_state, patience = float("inf"), None, 0
    train_losses, val_losses = [], []
    start = time.perf_counter()

    for epoch in range(ae["epochs"]):
        model.train()
        running = 0.0
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch), batch)
            loss.backward()
            optimizer.step()
            running += loss.item() * batch.size(0)
        train_loss = running / len(train_ds)

        model.eval()
        running = 0.0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                running += criterion(model(batch), batch).item() * batch.size(0)
        val_loss = running / len(val_ds)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"[AE] epoch {epoch+1}/{ae['epochs']} "
              f"train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        if val_loss < best_val - 1e-9:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= ae["early_stopping_patience"]:
                print(f"[AE] early stopping at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return TrainResult(model, train_losses, val_losses, time.perf_counter() - start)


def reconstruction_error(model: nn.Module, windows: np.ndarray,
                          device: str | None = None, batch_size: int = 256) -> np.ndarray:
    """Per-window reconstruction MSE — the AE's raw anomaly signal (Step 6)."""
    device = device or default_device()
    model = model.to(device).eval()
    errors = []
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            batch = torch.tensor(windows[start:start + batch_size],
                                  dtype=torch.float32).to(device)
            recon = model(batch)
            errors.append(torch.mean((recon - batch) ** 2, dim=(1, 2)).cpu().numpy())
    return np.concatenate(errors) if errors else np.array([])
