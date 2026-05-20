from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

Side = Literal['bid', 'ask']

@dataclass(frozen=True)
class PoissonFillModel:
    A: float = 0.5
    k: float = 1.5

    def fill_probability(self, delta: float, dt_seconds: float) -> float:
        if delta < 0:
            return 1.0
        intensity = self.A * math.exp(-self.k * delta)
        return 1.0 - math.exp(-intensity * dt_seconds)

    def draw_fills(self, bid_delta: float, ask_delta: float, dt_seconds: float, rng: np.random.Generator) -> tuple[bool, bool]:
        p_bid = self.fill_probability(bid_delta, dt_seconds)
        p_ask = self.fill_probability(ask_delta, dt_seconds)
        return (rng.random() < p_bid, rng.random() < p_ask)
__all__ = ['PoissonFillModel', 'Side']
