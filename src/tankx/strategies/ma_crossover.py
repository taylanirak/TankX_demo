from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class MACrossover:
    short_window: int = 20
    long_window: int = 100
    name: str = 'MA Crossover'
    params: dict[str, Any] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        if self.short_window <= 0 or self.long_window <= 0:
            raise ValueError('Windows must be positive')
        if self.short_window >= self.long_window:
            raise ValueError(f'short_window ({self.short_window}) must be < long_window ({self.long_window})')
        object.__setattr__(self, 'params', {'short_window': self.short_window, 'long_window': self.long_window})

    def generate_positions(self, df: pd.DataFrame) -> pd.Series:
        close = df['close']
        sma_short = close.rolling(self.short_window).mean()
        sma_long = close.rolling(self.long_window).mean()
        position = (sma_short > sma_long).astype(float)
        position = position.where(sma_long.notna())
        position.name = 'position'
        return position
__all__ = ['MACrossover']
