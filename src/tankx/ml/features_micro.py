from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tankx.config import TRADES_ROOT
from tankx.data.trades import _daily_parquet_path
from tankx.microstructure.features import order_flow_imbalance


def _cached_trade_paths(symbol: str, root: Path=TRADES_ROOT) -> list[Path]:
    slug = symbol.replace('/', '').upper()
    import datetime as dt
    sentinel = _daily_parquet_path(symbol, dt.date(2000, 1, 1), root=root)
    parent = sentinel.parent
    if not parent.exists():
        return []
    return sorted(parent.glob(f'{slug}-aggTrades-*.parquet'))

def load_cached_trades(symbol: str, root: Path=TRADES_ROOT) -> pd.DataFrame:
    paths = _cached_trade_paths(symbol, root)
    if not paths:
        return pd.DataFrame()
    frames = [pd.read_parquet(p) for p in paths]
    out = pd.concat(frames, ignore_index=True).sort_values('ts', kind='mergesort')
    return out.reset_index(drop=True)

def build_micro_features(bars: pd.DataFrame, trades: pd.DataFrame, *, ofi_window: str='5min') -> pd.DataFrame:
    if trades is None or trades.empty:
        return pd.DataFrame(index=bars.index)
    if 'ts' not in trades.columns:
        raise KeyError("trades frame must have a 'ts' column")
    idx = pd.DatetimeIndex(bars.index)
    if idx.tz is None:
        raise ValueError('bars frame must be tz-aware')
    bar_delta = idx[1] - idx[0]
    bar_freq = f'{int(bar_delta.total_seconds())}s'
    trade_ts = pd.DatetimeIndex(trades['ts'])
    mask = (trade_ts >= idx.min() - pd.Timedelta(days=1)) & (trade_ts <= idx.max() + pd.Timedelta(days=1))
    sub = trades.loc[mask].copy()
    if sub.empty:
        return pd.DataFrame(index=bars.index)
    sign = np.where(sub['is_buyer_maker'].to_numpy(), -1.0, +1.0)
    sub['signed_vol'] = sign * sub['qty'].to_numpy(dtype=np.float64)
    ofi = order_flow_imbalance(sub, window=ofi_window)
    ofi_dedup = ofi.groupby(level=0).last()
    ofi_bar = ofi_dedup.resample(bar_freq).last().ffill()
    sub_indexed = sub.set_index(pd.DatetimeIndex(sub['ts']))
    signed_vol_bar = sub_indexed['signed_vol'].resample(bar_freq).sum()
    trade_count_bar = sub_indexed['price'].resample(bar_freq).count().astype(np.float64)
    shifted = pd.DataFrame({'ofi': ofi_bar, 'signed_vol': signed_vol_bar, 'trade_count': trade_count_bar}).shift(1)
    return shifted.reindex(bars.index, method='ffill')
__all__ = ['build_micro_features', 'load_cached_trades']
