from __future__ import annotations

from dataclasses import dataclass


@dataclass
class QueuePosition:
    price: float
    side: str
    shares_ahead: float
    last_level_size: float

    def on_trade_at_level(self, trade_qty: float) -> None:
        self.shares_ahead = max(0.0, self.shares_ahead - trade_qty)

    def on_level_size_change(self, new_level_size: float) -> None:
        if new_level_size >= self.last_level_size:
            self.last_level_size = new_level_size
            return
        decrease = self.last_level_size - new_level_size
        if self.last_level_size <= 0:
            self.shares_ahead = 0.0
        else:
            fraction_ahead = self.shares_ahead / self.last_level_size
            self.shares_ahead = max(0.0, self.shares_ahead - decrease * fraction_ahead)
        self.last_level_size = new_level_size

    def is_at_front(self) -> bool:
        return self.shares_ahead <= 0.0
__all__ = ['QueuePosition']
