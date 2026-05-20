from __future__ import annotations

import numpy as np
import pandas as pd


def _timeframe_to_pandas_freq(tf: str) -> str:
    if tf.endswith('m'):
        return f'{int(tf[:-1])}min'
    if tf.endswith('h'):
        return f'{int(tf[:-1])}h'
    if tf.endswith('d'):
        return f'{int(tf[:-1])}D'
    raise ValueError(f'Unknown timeframe: {tf!r}')

def _timeframe_to_timedelta(tf: str) -> pd.Timedelta:
    if tf.endswith('m'):
        return pd.Timedelta(minutes=int(tf[:-1]))
    if tf.endswith('h'):
        return pd.Timedelta(hours=int(tf[:-1]))
    if tf.endswith('d'):
        return pd.Timedelta(days=int(tf[:-1]))
    raise ValueError(f'Unknown timeframe: {tf!r}')

def build_mtf_features(df: pd.DataFrame, timeframes: tuple[str, ...]=('1h', '4h')) -> pd.DataFrame:
    if 'close' not in df.columns:
        raise KeyError("build_mtf_features requires a 'close' column")
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is None:
        raise ValueError('Input frame must be tz-aware')
    out_cols: dict[str, pd.Series] = {}
    close_native = df['close']
    for tf in timeframes:
        freq = _timeframe_to_pandas_freq(tf)
        delta = _timeframe_to_timedelta(tf)
        agg_close = close_native.resample(freq).last().dropna()
        agg_ret = pd.Series(np.log(agg_close.to_numpy()[1:] / agg_close.to_numpy()[:-1]), index=agg_close.index[1:], name=f'ret_{tf}')
        agg_vol = agg_ret.rolling(window=20, min_periods=20).std()
        agg_ret_shifted = agg_ret.copy()
        agg_ret_shifted.index = agg_ret_shifted.index + delta
        agg_vol_shifted = agg_vol.copy()
        agg_vol_shifted.index = agg_vol_shifted.index + delta
        out_cols[f'ret_{tf}_lag1'] = agg_ret_shifted.reindex(idx, method='ffill')
        out_cols[f'vol_{tf}_lag1'] = agg_vol_shifted.reindex(idx, method='ffill')
    return pd.DataFrame(out_cols, index=df.index)
__all__ = ['build_mtf_features']
