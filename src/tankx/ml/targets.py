from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

TargetKind = Literal['direction', 'return', 'triple_barrier']

def build_return_target(df: pd.DataFrame) -> pd.Series:
    close = np.asarray(df['close'], dtype=np.float64)
    fwd_values = np.log(np.concatenate([close[1:], [np.nan]]) / close)
    return pd.Series(fwd_values, index=df.index, name='y_return')

def build_direction_target(df: pd.DataFrame, *, neutral_band: float=0.0) -> pd.Series:
    fwd = build_return_target(df)
    if neutral_band <= 0:
        return (fwd > 0).astype('float64').mask(fwd.isna()).rename('y_direction')
    out = pd.Series(np.nan, index=df.index, name='y_direction')
    out[fwd > neutral_band] = 1.0
    out[fwd < -neutral_band] = 0.0
    return out

def align_features_and_target(features: pd.DataFrame, target: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    if not features.index.equals(target.index):
        raise ValueError('features and target must share the same index')
    mask = features.notna().all(axis=1) & target.notna()
    return (features.loc[mask], target.loc[mask])
__all__ = ['TargetKind', 'align_features_and_target', 'build_direction_target', 'build_return_target']
