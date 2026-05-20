from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch
from torch import nn

from tankx.ml.models.base import TargetKind
from tankx.ml.windows import make_sequences

DeepKind = Literal['lstm', 'gru', 'transformer']

def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

def _to_tensor(arr: np.ndarray, *, dtype: torch.dtype=torch.float32) -> torch.Tensor:
    return torch.as_tensor(np.ascontiguousarray(arr), dtype=dtype)

class _RecurrentNet(nn.Module):

    def __init__(self, n_features: int, hidden_size: int, num_layers: int, kind: Literal['lstm', 'gru'], dropout: float) -> None:
        super().__init__()
        rnn_cls = nn.LSTM if kind == 'lstm' else nn.GRU
        self.rnn = rnn_cls(input_size=n_features, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout if num_layers > 1 else 0.0, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)

class _TransformerNet(nn.Module):

    def __init__(self, n_features: int, d_model: int, nhead: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=2 * d_model, dropout=dropout, batch_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.encoder(h)
        return self.head(h[:, -1, :]).squeeze(-1)

@dataclass
class _TorchPredictorBase:
    seq_len: int = 32
    hidden_size: int = 32
    num_layers: int = 1
    dropout: float = 0.2
    epochs: int = 10
    batch_size: int = 256
    lr: float = 0.001
    weight_decay: float = 0.0001
    val_fraction: float = 0.1
    patience: int = 3
    seed: int = 0
    device: str = 'cpu'
    target_kind: TargetKind = 'direction'
    name: str = 'TorchModel'
    history: list[dict[str, float]] = field(default_factory=list)
    _model: nn.Module | None = None
    _n_features: int = 0

    def _build_model(self, n_features: int) -> nn.Module:
        raise NotImplementedError

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        _set_seed(self.seed)
        self.history.clear()
        if len(X) < self.seq_len + 5:
            self._model = None
            self._n_features = X.shape[1]
            return
        n = len(X)
        n_val = max(self.seq_len + 1, int(self.val_fraction * n))
        X_train_raw = X[:n - n_val]
        y_train_raw = y[:n - n_val]
        X_val_raw = X[n - n_val:]
        y_val_raw = y[n - n_val:]
        X_train, y_train = make_sequences(X_train_raw, y_train_raw, self.seq_len)
        X_val, y_val = make_sequences(X_val_raw, y_val_raw, self.seq_len)
        device = torch.device(self.device)
        model = self._build_model(X.shape[1]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.target_kind == 'direction':
            loss_fn: nn.Module = nn.BCEWithLogitsLoss()
        else:
            loss_fn = nn.MSELoss()
        Xt = _to_tensor(X_train).to(device)
        yt = _to_tensor(y_train).to(device)
        Xv = _to_tensor(X_val).to(device)
        yv = _to_tensor(y_val).to(device)
        best_val = float('inf')
        best_state: dict[str, torch.Tensor] | None = None
        patience_left = self.patience
        n_samples = Xt.shape[0]
        rng = torch.Generator(device='cpu').manual_seed(self.seed)
        for epoch in range(self.epochs):
            model.train()
            perm = torch.randperm(n_samples, generator=rng)
            train_loss_sum = 0.0
            n_batches = 0
            for start in range(0, n_samples, self.batch_size):
                idx = perm[start:start + self.batch_size]
                xb = Xt[idx]
                yb = yt[idx]
                optimizer.zero_grad()
                logits = model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                optimizer.step()
                train_loss_sum += float(loss.detach().item())
                n_batches += 1
            train_loss = train_loss_sum / max(1, n_batches)
            model.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(model(Xv), yv).item())
            self.history.append({'epoch': float(epoch), 'train_loss': train_loss, 'val_loss': val_loss})
            if val_loss + 1e-08 < best_val:
                best_val = val_loss
                best_state = {k: v.clone().detach() for k, v in model.state_dict().items()}
                patience_left = self.patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        self._model = model
        self._n_features = X.shape[1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            n = len(X)
            return np.full(n, 0.5 if self.target_kind == 'direction' else 0.0)
        if X.shape[1] != self._n_features:
            raise ValueError(f'Feature dim mismatch: trained on {self._n_features}, got {X.shape[1]}')
        device = torch.device(self.device)
        model = self._model
        model.eval()
        n = len(X)
        if n < self.seq_len:
            return np.full(n, 0.5 if self.target_kind == 'direction' else 0.0)
        X_seq, _ = make_sequences(X, np.zeros(n), self.seq_len)
        with torch.no_grad():
            logits = model(_to_tensor(X_seq).to(device)).cpu().numpy()
        if self.target_kind == 'direction':
            preds_seq = 1.0 / (1.0 + np.exp(-logits))
            neutral_fill = 0.5
        else:
            preds_seq = logits
            neutral_fill = 0.0
        out = np.full(n, neutral_fill, dtype=np.float64)
        out[self.seq_len - 1:] = preds_seq
        return out

@dataclass
class LSTMPredictor(_TorchPredictorBase):
    name: str = 'LSTM'

    def _build_model(self, n_features: int) -> nn.Module:
        return _RecurrentNet(n_features, self.hidden_size, self.num_layers, 'lstm', self.dropout)

@dataclass
class GRUPredictor(_TorchPredictorBase):
    name: str = 'GRU'

    def _build_model(self, n_features: int) -> nn.Module:
        return _RecurrentNet(n_features, self.hidden_size, self.num_layers, 'gru', self.dropout)

@dataclass
class TransformerPredictor(_TorchPredictorBase):
    name: str = 'Transformer'
    nhead: int = 4
    d_model: int = 32
    num_layers: int = 2

    def _build_model(self, n_features: int) -> nn.Module:
        return _TransformerNet(n_features, self.d_model, self.nhead, self.num_layers, self.dropout)
__all__ = ['DeepKind', 'GRUPredictor', 'LSTMPredictor', 'TransformerPredictor']
