from __future__ import annotations

import numpy as np


def make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    if X.ndim != 2:
        raise ValueError(f'X must be 2-D, got shape {X.shape}')
    if y.ndim != 1:
        raise ValueError(f'y must be 1-D, got shape {y.shape}')
    if len(X) != len(y):
        raise ValueError(f'X len {len(X)} != y len {len(y)}')
    if seq_len < 1:
        raise ValueError('seq_len must be >= 1')
    n_samples = len(X)
    if n_samples < seq_len:
        raise ValueError(f'Need at least seq_len={seq_len} rows, got {n_samples}')
    n_out = n_samples - seq_len + 1
    n_feats = X.shape[1]
    out_X = np.empty((n_out, seq_len, n_feats), dtype=X.dtype)
    for i in range(n_out):
        out_X[i] = X[i:i + seq_len]
    out_y = y[seq_len - 1:]
    return (out_X, out_y)
__all__ = ['make_sequences']
