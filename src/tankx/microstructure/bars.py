from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


def time_bars(trades: pd.DataFrame, interval: str='1s') -> pd.DataFrame:
    g = trades.set_index('ts').groupby(pd.Grouper(freq=interval))
    out = g.agg(open=('price', 'first'), high=('price', 'max'), low=('price', 'min'), close=('price', 'last'), volume=('qty', 'sum'), trades=('price', 'count')).dropna(subset=['open'])
    out.index = pd.DatetimeIndex(out.index, name='timestamp')
    if out.index.tz is None:
        out.index = out.index.tz_localize('UTC')
    return out

def _accumulating_bars(trades: pd.DataFrame, threshold: float, *, mode: Literal['volume', 'dollar']) -> pd.DataFrame:
    if threshold <= 0:
        raise ValueError('threshold must be positive')
    if mode not in {'volume', 'dollar'}:
        raise ValueError(f'unknown mode {mode!r}')
    price = trades['price'].to_numpy(dtype=np.float64)
    qty = trades['qty'].to_numpy(dtype=np.float64)
    ts = trades['ts'].to_numpy()
    weight = qty if mode == 'volume' else price * qty
    cum = np.cumsum(weight)
    bar_idx = (cum // threshold).astype(np.int64)
    df = pd.DataFrame({'ts': ts, 'price': price, 'qty': qty, 'bar_idx': bar_idx})
    df['dollar'] = df['price'] * df['qty']
    g = df.groupby('bar_idx')
    out = g.agg(timestamp=('ts', 'last'), open=('price', 'first'), high=('price', 'max'), low=('price', 'min'), close=('price', 'last'), volume=('qty', 'sum'), trades=('price', 'count'), dollar_volume=('dollar', 'sum'))
    out = out.set_index('timestamp')
    out.index = pd.DatetimeIndex(out.index, name='timestamp')
    if out.index.tz is None:
        out.index = out.index.tz_localize('UTC')
    return out

def volume_bars(trades: pd.DataFrame, threshold_qty: float) -> pd.DataFrame:
    return _accumulating_bars(trades, threshold_qty, mode='volume')

def dollar_bars(trades: pd.DataFrame, threshold_usd: float) -> pd.DataFrame:
    return _accumulating_bars(trades, threshold_usd, mode='dollar')
__all__ = ['dollar_bars', 'time_bars', 'volume_bars']
