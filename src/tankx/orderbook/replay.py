from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import pandas as pd

from tankx.orderbook.lob import BBOSnapshot, LimitOrderBook


@dataclass(frozen=True)
class BookSnapshot:
    ts: pd.Timestamp
    mid: float
    bbo: BBOSnapshot
    book: LimitOrderBook

def _snapshot_to_levels(rows: pd.DataFrame, mid: float) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    rows = rows.sort_values('percentage')
    bid_rows = rows[rows['percentage'] < 0]
    ask_rows = rows[rows['percentage'] > 0]
    bid_levels: list[tuple[float, float]] = []
    prev_cum = 0.0
    for _, row in bid_rows.iloc[::-1].iterrows():
        cum = float(row['depth'])
        inc = max(0.0, cum - prev_cum)
        if inc > 0:
            price = mid * (1.0 + float(row['percentage']) / 100.0)
            bid_levels.append((price, inc))
        prev_cum = cum
    ask_levels: list[tuple[float, float]] = []
    prev_cum = 0.0
    for _, row in ask_rows.iterrows():
        cum = float(row['depth'])
        inc = max(0.0, cum - prev_cum)
        if inc > 0:
            price = mid * (1.0 + float(row['percentage']) / 100.0)
            ask_levels.append((price, inc))
        prev_cum = cum
    return (bid_levels, ask_levels)

def replay_book_depth(book_depth: pd.DataFrame, reference_price: pd.Series) -> Iterator[BookSnapshot]:
    idx = pd.DatetimeIndex(reference_price.index)
    if idx.tz is None:
        raise ValueError('reference_price must be tz-aware')
    book = LimitOrderBook()
    grouped = book_depth.groupby('ts', sort=True)
    ref_idx = idx
    ref_values = reference_price.to_numpy()
    for ts, rows in grouped:
        pos = ref_idx.searchsorted(ts, side='right') - 1
        if pos < 0:
            continue
        mid_proxy = float(ref_values[pos])
        if mid_proxy <= 0:
            continue
        bid_levels, ask_levels = _snapshot_to_levels(rows, mid_proxy)
        book.apply_snapshot('bid', bid_levels)
        book.apply_snapshot('ask', ask_levels)
        bbo = book.bbo()
        if bbo is None:
            continue
        ts_typed = ts if isinstance(ts, pd.Timestamp) else pd.Timestamp(ts)  # type: ignore[arg-type]
        yield BookSnapshot(ts=ts_typed, mid=bbo.mid, bbo=bbo, book=book)

def collect_top_of_book_series(snapshots: Iterable[BookSnapshot]) -> pd.DataFrame:
    rows = []
    for snap in snapshots:
        bbo = snap.bbo
        rows.append({'ts': snap.ts, 'bid_price': bbo.bid_price, 'bid_size': bbo.bid_size, 'ask_price': bbo.ask_price, 'ask_size': bbo.ask_size, 'mid': bbo.mid, 'spread': bbo.spread, 'microprice': bbo.microprice, 'imbalance': bbo.imbalance})
    if not rows:
        return pd.DataFrame(columns=['ts', 'bid_price', 'bid_size', 'ask_price', 'ask_size', 'mid', 'spread', 'microprice', 'imbalance'])
    return pd.DataFrame(rows).set_index('ts')
__all__ = ['BookSnapshot', 'collect_top_of_book_series', 'replay_book_depth']
