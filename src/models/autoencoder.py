"""
 reconstruction-based Autoencoder, trained exclusively on
normal operational data so it learns to reconstruct standard sensor
patterns with minimal error. Anomalies (unseen patterns) reconstruct poorly,
which is the signal the scoring node consumes.

Supports two architectures (configs/config.yaml -> autoencoder.type):
  - "lstm"  : LSTM encoder-decoder, good for temporal/sequential structure
  - "dense" : flattened-window dense AE, cheaper, useful as a fast baseline
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
        # x: (batch, window, n_features)
        batch, window, _ = x.shape
        _, (h_n, _) = self.encoder(x)
        latent = self.to_latent(h_n[-1])                  # (batch, latent_dim)
        seed = self.from_latent(latent)                    # (batch, hidden_dim)
        seed_seq = seed.unsqueeze(1).repeat(1, window, 1)   # repeat across time
        decoded, _ = self.decoder(seed_seq)
        return self.output_layer(decoded)


class DenseAutoencoder(nn.Module):
    def __init__(self, n_features: int, window_size: int, hidden_dim: int,
                 latent_dim: int, dropout: float):
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
        z = self.encoder(flat)
        out = self.decoder(z)
        return out.reshape(batch, self.window_size, self.n_features)


def build_autoencoder(n_features: int, window_size: int, cfg: dict) -> nn.Module:
    ae_cfg = cfg["autoencoder"]
    if ae_cfg["type"] == "lstm":
        return LSTMAutoencoder(
            n_features=n_features, hidden_dim=ae_cfg["hidden_dim"],
            latent_dim=ae_cfg["latent_dim"], num_layers=ae_cfg["num_layers"],
            dropout=ae_cfg["dropout"],
        )
    elif ae_cfg["type"] == "dense":
        return DenseAutoencoder(
            n_features=n_features, window_size=window_size,
            hidden_dim=ae_cfg["hidden_dim"], latent_dim=ae_cfg["latent_dim"],
            dropout=ae_cfg["dropout"],
        )
    raise ValueError(f"Unknown autoencoder type {ae_cfg['type']}")


@dataclass
class TrainResult:
    model: nn.Module
    train_losses: list
    val_losses: list
    train_time_sec: float


def train_autoencoder(train_windows: np.ndarray, val_windows: np.ndarray,
                       cfg: dict, device: str = None) -> TrainResult:
    """
    train the AE exclusively on normal data with early stopping on
    a held-out normal validation split (reconstruction MSE).
    """
    ae_cfg = cfg["autoencoder"]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    n_features = train_windows.shape[2]
    window_size = train_windows.shape[1]
    model = build_autoencoder(n_features, window_size, cfg).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=ae_cfg["learning_rate"])
    criterion = nn.MSELoss()

    train_ds = TensorDataset(torch.tensor(train_windows, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(val_windows, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=ae_cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=ae_cfg["batch_size"], shuffle=False)

    best_val = float("inf")
    best_state = None
    patience_ctr = 0
    train_losses, val_losses = [], []

    start = time.perf_counter()
    for epoch in range(ae_cfg["epochs"]):
        model.train()
        running = 0.0
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            running += loss.item() * batch.size(0)
        train_loss = running / len(train_ds)

        model.eval()
        running = 0.0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                recon = model(batch)
                running += criterion(recon, batch).item() * batch.size(0)
        val_loss = running / len(val_ds)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"[AE] epoch {epoch+1}/{ae_cfg['epochs']} "
              f"train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= ae_cfg["early_stopping_patience"]:
                print(f"[AE] early stopping at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    elapsed = time.perf_counter() - start

    return TrainResult(model=model, train_losses=train_losses,
                        val_losses=val_losses, train_time_sec=elapsed)


def reconstruction_error(model: nn.Module, windows: np.ndarray,
                          device: str = None, batch_size: int = 256) -> np.ndarray:
    """
    Per-window reconstruction error (mean squared error across the window),
    used as one half of the scoring node.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    errors = []
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            batch = torch.tensor(windows[start:start + batch_size], dtype=torch.float32).to(device)
            recon = model(batch)
            mse = torch.mean((recon - batch) ** 2, dim=(1, 2))
            errors.append(mse.cpu().numpy())
    return np.concatenate(errors)
