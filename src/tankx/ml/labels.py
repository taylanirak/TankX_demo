from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TripleBarrierConfig:
    horizon_bars: int = 12
    take_profit_sigma: float = 1.0
    stop_loss_sigma: float = 1.0
    sigma_lookback: int = 100

def triple_barrier_labels(df: pd.DataFrame, config: TripleBarrierConfig=TripleBarrierConfig()) -> pd.Series:
    if 'close' not in df.columns:
        raise KeyError("triple_barrier_labels requires a 'close' column")
    close = np.asarray(df['close'], dtype=np.float64)
    high = np.asarray(df['high'], dtype=np.float64) if 'high' in df.columns else close
    low = np.asarray(df['low'], dtype=np.float64) if 'low' in df.columns else close
    n = len(close)
    log_ret = np.zeros(n, dtype=np.float64)
    log_ret[1:] = np.log(close[1:] / close[:-1])
    sigma = pd.Series(log_ret).rolling(window=config.sigma_lookback, min_periods=config.sigma_lookback).std().to_numpy()
    labels = np.full(n, np.nan, dtype=np.float64)
    h = config.horizon_bars
    k_tp = config.take_profit_sigma
    k_sl = config.stop_loss_sigma
    for t in range(n - h):
        s = sigma[t]
        if not np.isfinite(s) or s <= 0:
            continue
        entry_price = close[t]
        tp_price = entry_price * np.exp(k_tp * s)
        sl_price = entry_price * np.exp(-k_sl * s)
        label = 0.0
        for j in range(1, h + 1):
            if high[t + j] >= tp_price:
                label = 1.0
                break
            if low[t + j] <= sl_price:
                label = -1.0
                break
        labels[t] = label
    return pd.Series(labels, index=df.index, name='y_triple_barrier')

def label_distribution(labels: pd.Series) -> dict[str, float]:
    s = labels.dropna()
    n = max(1, len(s))
    return {'tp_frac': float((s == 1.0).sum()) / n, 'sl_frac': float((s == -1.0).sum()) / n, 'timeout_frac': float((s == 0.0).sum()) / n, 'n_labels': float(n)}
__all__ = ['TripleBarrierConfig', 'label_distribution', 'triple_barrier_labels']
