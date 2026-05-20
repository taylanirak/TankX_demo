from __future__ import annotations

import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from tankx.orderbook.lob import LimitOrderBook
from tankx.orderbook.queue import QueuePosition
from tankx.orderbook.replay import collect_top_of_book_series, replay_book_depth


def test_empty_book_has_no_bbo() -> None:
    lob = LimitOrderBook()
    assert lob.bbo() is None
    assert lob.best_bid() is None
    assert lob.best_ask() is None
    assert lob.is_empty()

def test_apply_update_basic() -> None:
    lob = LimitOrderBook()
    lob.apply_update('bid', 100.0, 1.0)
    lob.apply_update('bid', 99.0, 2.0)
    lob.apply_update('ask', 101.0, 3.0)
    lob.apply_update('ask', 102.0, 4.0)
    assert lob.best_bid() == (100.0, 1.0)
    assert lob.best_ask() == (101.0, 3.0)
    bbo = lob.bbo()
    assert bbo is not None
    assert bbo.spread == pytest.approx(1.0)

def test_apply_update_zero_size_removes_level() -> None:
    lob = LimitOrderBook()
    lob.apply_update('bid', 100.0, 1.0)
    lob.apply_update('bid', 100.0, 0.0)
    assert lob.best_bid() is None

def test_apply_update_rejects_non_positive_price() -> None:
    lob = LimitOrderBook()
    with pytest.raises(ValueError, match='price'):
        lob.apply_update('bid', 0.0, 1.0)

def test_apply_update_rejects_negative_size() -> None:
    lob = LimitOrderBook()
    with pytest.raises(ValueError, match='size'):
        lob.apply_update('bid', 100.0, -1.0)

def test_snapshot_replaces_side() -> None:
    lob = LimitOrderBook()
    lob.apply_update('bid', 99.0, 5.0)
    lob.apply_snapshot('bid', [(100.0, 1.0), (98.0, 2.0)])
    assert lob.best_bid() == (100.0, 1.0)
    bids, _ = lob.depth_n(5)
    assert 99.0 not in [p for p, _ in bids]

def test_microprice_leans_toward_thinner_side() -> None:
    lob = LimitOrderBook()
    lob.apply_update('bid', 100.0, 1.0)
    lob.apply_update('ask', 101.0, 100.0)
    bbo = lob.bbo()
    assert bbo is not None
    assert bbo.microprice < bbo.mid

def test_imbalance() -> None:
    lob = LimitOrderBook()
    for i, size in enumerate([10.0, 5.0, 3.0]):
        lob.apply_update('bid', 100.0 - i, size)
    for i, size in enumerate([1.0, 1.0, 1.0]):
        lob.apply_update('ask', 101.0 + i, size)
    imb = lob.imbalance(depth=3)
    assert imb == pytest.approx((18 - 3) / (18 + 3))

def test_validate_detects_crossed_book() -> None:
    lob = LimitOrderBook()
    lob.apply_update('bid', 100.0, 1.0)
    lob.apply_update('ask', 99.0, 1.0)
    with pytest.raises(ValueError, match='Crossed'):
        lob.validate()

def test_validate_passes_clean_book() -> None:
    lob = LimitOrderBook()
    lob.apply_update('bid', 99.0, 1.0)
    lob.apply_update('ask', 101.0, 1.0)
    lob.validate()

@given(ops=st.lists(st.tuples(st.sampled_from(['bid', 'ask']), st.floats(min_value=0.01, max_value=1000.0), st.floats(min_value=0.0, max_value=10.0)), min_size=0, max_size=50))
@settings(max_examples=80, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_lob_invariants_under_random_ops(ops: list[tuple[str, float, float]]) -> None:
    lob = LimitOrderBook()
    for side, price, size in ops:
        lob.apply_update(side, price, size)
        b = lob.best_bid()
        a = lob.best_ask()
        if b is not None:
            assert b[1] > 0
            assert b[0] > 0
        if a is not None:
            assert a[1] > 0
            assert a[0] > 0

def test_queue_position_decrements_on_trades() -> None:
    q = QueuePosition(price=100.0, side='bid', shares_ahead=10.0, last_level_size=15.0)
    q.on_trade_at_level(3.0)
    assert q.shares_ahead == 7.0
    q.on_trade_at_level(20.0)
    assert q.shares_ahead == 0.0
    assert q.is_at_front()

def test_queue_position_decays_on_cancellations() -> None:
    q = QueuePosition(price=100.0, side='bid', shares_ahead=10.0, last_level_size=20.0)
    q.on_level_size_change(15.0)
    assert q.shares_ahead == pytest.approx(7.5)
    assert q.last_level_size == 15.0

def test_queue_position_level_grows_is_noop() -> None:
    q = QueuePosition(price=100.0, side='bid', shares_ahead=5.0, last_level_size=10.0)
    q.on_level_size_change(20.0)
    assert q.shares_ahead == 5.0
    assert q.last_level_size == 20.0

def test_replay_yields_snapshots() -> None:
    rows = [{'ts': pd.Timestamp('2026-05-15 00:00:01', tz='UTC'), 'percentage': -1.0, 'depth': 5.0, 'notional': 5 * 50000.0}, {'ts': pd.Timestamp('2026-05-15 00:00:01', tz='UTC'), 'percentage': -0.2, 'depth': 2.0, 'notional': 2 * 50000.0}, {'ts': pd.Timestamp('2026-05-15 00:00:01', tz='UTC'), 'percentage': 0.2, 'depth': 2.5, 'notional': 2.5 * 50000.0}, {'ts': pd.Timestamp('2026-05-15 00:00:01', tz='UTC'), 'percentage': 1.0, 'depth': 6.0, 'notional': 6 * 50000.0}]
    book_depth = pd.DataFrame(rows)
    ref = pd.Series([50000.0], index=pd.DatetimeIndex([pd.Timestamp('2026-05-15 00:00:00', tz='UTC')]))
    snaps = list(replay_book_depth(book_depth, ref))
    assert len(snaps) == 1
    snap = snaps[0]
    assert snap.bbo.bid_price < snap.bbo.ask_price
    assert snap.bbo.spread > 0

def test_collect_top_of_book_returns_aligned_frame() -> None:
    rows = []
    for hour in range(3):
        ts = pd.Timestamp('2026-05-15', tz='UTC') + pd.Timedelta(hours=hour)
        for pct, depth in [(-1.0, 5.0), (-0.2, 2.0), (0.2, 2.5), (1.0, 6.0)]:
            rows.append({'ts': ts, 'percentage': pct, 'depth': depth, 'notional': 0.0})
    bd = pd.DataFrame(rows)
    ref = pd.Series([50000.0, 50500.0, 50100.0], index=pd.DatetimeIndex([pd.Timestamp('2026-05-15', tz='UTC') + pd.Timedelta(hours=h) for h in range(3)]))
    snaps = list(replay_book_depth(bd, ref))
    out = collect_top_of_book_series(snaps)
    assert len(out) == 3
    assert {'bid_price', 'ask_price', 'spread', 'microprice', 'imbalance'} <= set(out.columns)
    assert (out['spread'] > 0).all()
    assert out['imbalance'].between(-1.0, 1.0).all()
