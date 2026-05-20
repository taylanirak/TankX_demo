from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BollingerMeanReversion:
    window: int = 20
    num_std: float = 2.0
    name: str = 'Bollinger Mean Reversion'
    params: dict[str, Any] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        if self.window <= 1:
            raise ValueError('window must be > 1')
        if self.num_std <= 0:
            raise ValueError('num_std must be > 0')
        object.__setattr__(self, 'params', {'window': self.window, 'num_std': self.num_std})

    def generate_positions(self, df: pd.DataFrame) -> pd.Series:
        close = df['close'].to_numpy()
        mu = df['close'].rolling(self.window).mean().to_numpy()
        sigma = df['close'].rolling(self.window).std().to_numpy()
        lower = mu - self.num_std * sigma
        n = len(close)
        position = np.full(n, np.nan, dtype=np.float64)
        in_trade = False
        for i in range(n):
            if np.isnan(mu[i]):
                continue
            if not in_trade:
                if close[i] < lower[i]:
                    in_trade = True
                    position[i] = 1.0
                else:
                    position[i] = 0.0
            elif close[i] > mu[i]:
                in_trade = False
                position[i] = 0.0
            else:
                position[i] = 1.0
        return pd.Series(position, index=df.index, name='position')
__all__ = ['BollingerMeanReversion']
