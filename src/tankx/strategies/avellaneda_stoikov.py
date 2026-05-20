from __future__ import annotations

import math
from dataclasses import dataclass

from tankx.backtest.engine_event import MMState, Quotes


@dataclass(frozen=True)
class AvellanedaStoikov:
    name: str = 'Avellaneda-Stoikov'
    gamma: float = 0.01
    sigma: float = 5.0
    k: float = 1.5

    def __post_init__(self) -> None:
        if self.gamma <= 0:
            raise ValueError('gamma must be > 0')
        if self.sigma <= 0:
            raise ValueError('sigma must be > 0')
        if self.k <= 0:
            raise ValueError('k must be > 0')

    def quotes(self, state: MMState) -> Quotes:
        t_left = max(1e-09, state.horizon_left_seconds)
        reservation = state.mid - state.inventory * self.gamma * self.sigma ** 2 * t_left
        half_spread = self.gamma * self.sigma ** 2 * t_left + 2.0 / self.gamma * math.log(1.0 + self.gamma / self.k)
        return Quotes(bid=reservation - half_spread, ask=reservation + half_spread)
__all__ = ['AvellanedaStoikov']
