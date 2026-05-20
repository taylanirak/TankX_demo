from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureConfig:
    n_lag_returns: int = 10
    rolling_windows: tuple[int, ...] = (5, 20, 60)
    include_volume_z: bool = True
    include_tod_cyclic: bool = True
    volume_z_window: int = 60
    include_multi_timeframe: bool = False
    mtf_timeframes: tuple[str, ...] = ('1h', '4h')

def build_features(df: pd.DataFrame, config: FeatureConfig=FeatureConfig()) -> pd.DataFrame:
    if 'close' not in df.columns or 'volume' not in df.columns:
        raise KeyError("Input frame must have 'close' and 'volume' columns")
    feats: dict[str, pd.Series] = {}
    close = df['close']
    log_ret_values = np.log(np.asarray(close, dtype=np.float64) / np.asarray(close.shift(1), dtype=np.float64))
    log_ret = pd.Series(log_ret_values, index=df.index, name='log_ret')
    for k in range(1, config.n_lag_returns + 1):
        feats[f'ret_lag{k}'] = log_ret.shift(k - 1)
    for w in config.rolling_windows:
        feats[f'ret_mean_{w}'] = log_ret.rolling(window=w).mean()
        feats[f'ret_std_{w}'] = log_ret.rolling(window=w).std()
    if config.include_volume_z:
        vol = df['volume']
        mu = vol.rolling(window=config.volume_z_window).mean()
        sigma = vol.rolling(window=config.volume_z_window).std()
        z = (vol - mu) / sigma
        feats['volume_z'] = z.where(sigma > 0, 0.0)
    if config.include_tod_cyclic:
        idx = pd.DatetimeIndex(df.index)
        hours = idx.hour + idx.minute / 60.0
        feats['tod_sin'] = pd.Series(np.sin(2.0 * np.pi * hours / 24.0), index=df.index)
        feats['tod_cos'] = pd.Series(np.cos(2.0 * np.pi * hours / 24.0), index=df.index)
    out = pd.DataFrame(feats, index=df.index).replace([np.inf, -np.inf], np.nan)
    if config.include_multi_timeframe:
        from tankx.ml.features_mtf import build_mtf_features
        mtf = build_mtf_features(df, timeframes=config.mtf_timeframes)
        out = pd.concat([out, mtf], axis=1)
    return out

def feature_column_names(config: FeatureConfig=FeatureConfig()) -> list[str]:
    cols = [f'ret_lag{k}' for k in range(1, config.n_lag_returns + 1)]
    for w in config.rolling_windows:
        cols.append(f'ret_mean_{w}')
        cols.append(f'ret_std_{w}')
    if config.include_volume_z:
        cols.append('volume_z')
    if config.include_tod_cyclic:
        cols.append('tod_sin')
        cols.append('tod_cos')
    if config.include_multi_timeframe:
        for tf in config.mtf_timeframes:
            cols.append(f'ret_{tf}_lag1')
            cols.append(f'vol_{tf}_lag1')
    return cols
__all__ = ['FeatureConfig', 'build_features', 'feature_column_names']
