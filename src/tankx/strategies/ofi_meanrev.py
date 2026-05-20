from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OFIMeanReversion:
    z_threshold: float = 2.0
    holding_bars: int = 5
    lookback_bars: int = 60
    name: str = 'OFI Mean Reversion'
    params: dict[str, Any] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        if self.z_threshold <= 0:
            raise ValueError('z_threshold must be positive')
        if self.holding_bars <= 0:
            raise ValueError('holding_bars must be positive')
        if self.lookback_bars < 10:
            raise ValueError('lookback_bars must be >= 10')
        object.__setattr__(self, 'params', {'z_threshold': self.z_threshold, 'holding_bars': self.holding_bars, 'lookback_bars': self.lookback_bars})

    def generate_positions(self, df: pd.DataFrame) -> pd.Series:
        if 'ofi' not in df.columns:
            raise KeyError("OFIMeanReversion requires an 'ofi' column on the bar frame")
        ofi = df['ofi']
        mean = ofi.shift(1).rolling(self.lookback_bars).mean()
        std = ofi.shift(1).rolling(self.lookback_bars).std()
        z = (ofi - mean) / std
        target = np.where(z > self.z_threshold, -1.0, np.where(z < -self.z_threshold, +1.0, 0.0))
        n = len(df)
        position = np.zeros(n, dtype=np.float64)
        remaining = 0
        current = 0.0
        z_arr = z.to_numpy(dtype=np.float64)
        for i in range(n):
            if np.isnan(z_arr[i]):
                position[i] = 0.0
                continue
            if remaining > 0:
                position[i] = current
                remaining -= 1
                continue
            if target[i] != 0.0:
                current = float(target[i])
                remaining = self.holding_bars - 1
                position[i] = current
            else:
                current = 0.0
                position[i] = 0.0
        return pd.Series(position, index=df.index, name='position')

def build_ofi_bar_frame(trades: pd.DataFrame, bar: str='1min', ofi_window: str='30s') -> pd.DataFrame:
    from tankx.microstructure.bars import time_bars
    from tankx.microstructure.features import order_flow_imbalance
    bars = time_bars(trades, interval=bar)
    ofi = order_flow_imbalance(trades, window=ofi_window)
    ofi = ofi.groupby(level=0).last()
    bars['ofi'] = ofi.reindex(bars.index, method='ffill')
    bars['ofi'] = bars['ofi'].fillna(0.0)
    return bars
__all__ = ['OFIMeanReversion', 'build_ofi_bar_frame']
