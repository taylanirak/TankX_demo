from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from sortedcontainers import SortedDict

Side = Literal['bid', 'ask']

@dataclass(frozen=True)
class BBOSnapshot:
    bid_price: float
    bid_size: float
    ask_price: float
    ask_size: float

    @property
    def mid(self) -> float:
        return 0.5 * (self.bid_price + self.ask_price)

    @property
    def spread(self) -> float:
        return self.ask_price - self.bid_price

    @property
    def microprice(self) -> float:
        total = self.bid_size + self.ask_size
        if total <= 0:
            return self.mid
        return (self.bid_price * self.ask_size + self.ask_price * self.bid_size) / total

    @property
    def imbalance(self) -> float:
        total = self.bid_size + self.ask_size
        if total <= 0:
            return 0.0
        return (self.bid_size - self.ask_size) / total

class LimitOrderBook:

    def __init__(self) -> None:
        self._bids: SortedDict = SortedDict()
        self._asks: SortedDict = SortedDict()

    def apply_update(self, side: Side, price: float, size: float) -> None:
        if price <= 0:
            raise ValueError(f'price must be > 0, got {price}')
        if size < 0:
            raise ValueError(f'size must be >= 0, got {size}')
        bucket = self._bids if side == 'bid' else self._asks
        key = -price if side == 'bid' else price
        if size == 0:
            bucket.pop(key, None)
        else:
            bucket[key] = size

    def apply_snapshot(self, side: Side, levels: list[tuple[float, float]]) -> None:
        bucket = self._bids if side == 'bid' else self._asks
        bucket.clear()
        for price, size in levels:
            if size <= 0:
                continue
            key = -price if side == 'bid' else price
            bucket[key] = size

    def best_bid(self) -> tuple[float, float] | None:
        if not self._bids:
            return None
        neg_price, size = self._bids.peekitem(0)
        return (-neg_price, size)

    def best_ask(self) -> tuple[float, float] | None:
        if not self._asks:
            return None
        return self._asks.peekitem(0)

    def bbo(self) -> BBOSnapshot | None:
        b = self.best_bid()
        a = self.best_ask()
        if b is None or a is None:
            return None
        return BBOSnapshot(bid_price=b[0], bid_size=b[1], ask_price=a[0], ask_size=a[1])

    def depth_n(self, n: int) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        bid_pairs: list[tuple[float, float]] = []
        for i, (neg_price, size) in enumerate(self._bids.items()):
            if i >= n:
                break
            bid_pairs.append((-neg_price, size))
        ask_pairs: list[tuple[float, float]] = []
        for i, (price, size) in enumerate(self._asks.items()):
            if i >= n:
                break
            ask_pairs.append((price, size))
        return (bid_pairs, ask_pairs)

    def imbalance(self, depth: int=5) -> float:
        bids, asks = self.depth_n(depth)
        b = sum((s for _, s in bids))
        a = sum((s for _, s in asks))
        if b + a <= 0:
            return 0.0
        return (b - a) / (b + a)

    def validate(self) -> None:
        if self._bids and self._asks:
            best_bid_price = -self._bids.peekitem(0)[0]
            best_ask_price = self._asks.peekitem(0)[0]
            if best_bid_price >= best_ask_price:
                raise ValueError(f'Crossed book: best_bid={best_bid_price} >= best_ask={best_ask_price}')
        for neg_price, size in self._bids.items():
            if -neg_price <= 0:
                raise ValueError(f'Non-positive bid price: {-neg_price}')
            if size <= 0:
                raise ValueError(f'Non-positive bid size at {-neg_price}: {size}')
        for price, size in self._asks.items():
            if price <= 0:
                raise ValueError(f'Non-positive ask price: {price}')
            if size <= 0:
                raise ValueError(f'Non-positive ask size at {price}: {size}')

    def __len__(self) -> int:
        return len(self._bids) + len(self._asks)

    def is_empty(self) -> bool:
        return len(self._bids) == 0 and len(self._asks) == 0
__all__ = ['BBOSnapshot', 'LimitOrderBook', 'Side']
