from __future__ import annotations

import math

import numpy as np
import pandas as pd


def vol_target_positions(positions: pd.Series, df: pd.DataFrame, *, daily_target: float=0.01, lookback: int=288, bars_per_day: int=288, max_leverage: float=1.0) -> pd.Series:
    if 'close' not in df.columns:
        raise KeyError("df must have a 'close' column")
    if not positions.index.equals(df.index):
        raise ValueError('positions.index must match df.index')
    if daily_target <= 0:
        raise ValueError('daily_target must be > 0')
    if lookback < 5:
        raise ValueError('lookback must be >= 5')
    close = np.asarray(df['close'], dtype=np.float64)
    log_ret = np.zeros(len(close), dtype=np.float64)
    log_ret[1:] = np.log(close[1:] / close[:-1])
    bar_std = pd.Series(log_ret).rolling(window=lookback, min_periods=lookback).std()
    daily_vol = bar_std * math.sqrt(bars_per_day)
    daily_vol = daily_vol.shift(1)
    raw = positions.to_numpy(dtype=np.float64)
    target_arr = np.where(daily_vol.to_numpy() > 0, np.minimum(max_leverage, daily_target / daily_vol.to_numpy()), 0.0)
    scaled = raw * np.nan_to_num(target_arr, nan=0.0)
    return pd.Series(scaled, index=positions.index, name='position')
__all__ = ['vol_target_positions']
