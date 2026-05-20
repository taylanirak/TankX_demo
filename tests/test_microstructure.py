from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tankx.microstructure.bars import dollar_bars, time_bars, volume_bars
from tankx.microstructure.features import (
    effective_spread,
    kyle_lambda,
    microprice_from_book,
    order_flow_imbalance,
    signed_volume,
    trade_size_distribution,
)


def _trades(n: int=1000, *, alternating: bool=False, seed: int=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed=seed)
    ts = pd.date_range('2026-05-15', periods=n, freq='100ms', tz='UTC')
    price = 50000.0 + np.cumsum(rng.normal(0, 0.5, n))
    qty = np.abs(rng.normal(0.05, 0.02, n))
    if alternating:
        is_buyer_maker = np.array([i % 2 == 0 for i in range(n)])
    else:
        is_buyer_maker = rng.random(n) < 0.5
    return pd.DataFrame({'agg_id': np.arange(n, dtype=np.int64), 'price': price, 'qty': qty, 'first_id': np.arange(n, dtype=np.int64), 'last_id': np.arange(n, dtype=np.int64), 'ts': ts, 'is_buyer_maker': is_buyer_maker})

def test_signed_volume_all_buys() -> None:
    df = _trades(100, seed=0)
    df['is_buyer_maker'] = False
    sv = signed_volume(df)
    assert (sv > 0).all()
    assert sv.sum() == pytest.approx(df['qty'].sum(), rel=1e-12)

def test_signed_volume_all_sells() -> None:
    df = _trades(100, seed=0)
    df['is_buyer_maker'] = True
    sv = signed_volume(df)
    assert (sv < 0).all()
    assert sv.sum() == pytest.approx(-df['qty'].sum(), rel=1e-12)

def test_ofi_in_unit_interval() -> None:
    df = _trades(2000)
    ofi = order_flow_imbalance(df, window='5s').dropna()
    assert ofi.between(-1.0, 1.0).all()

def test_ofi_all_buys_is_plus_one() -> None:
    df = _trades(500)
    df['is_buyer_maker'] = False
    ofi = order_flow_imbalance(df, window='5s').dropna()
    assert np.allclose(ofi.to_numpy(), 1.0)

def test_ofi_all_sells_is_minus_one() -> None:
    df = _trades(500)
    df['is_buyer_maker'] = True
    ofi = order_flow_imbalance(df, window='5s').dropna()
    assert np.allclose(ofi.to_numpy(), -1.0)

def test_effective_spread_zero_when_price_equals_mid() -> None:
    df = _trades(50)
    mid = pd.Series(df['price'].to_numpy(), index=pd.DatetimeIndex(df['ts']))
    eff = effective_spread(df, mid)
    assert np.allclose(eff.to_numpy(), 0.0)

def test_effective_spread_positive_when_price_deviates() -> None:
    df = _trades(50)
    mid = pd.Series(df['price'].to_numpy() - 1.0, index=pd.DatetimeIndex(df['ts']))
    eff = effective_spread(df, mid)
    assert (eff > 0).all()

def test_microprice_collapses_to_mid_for_symmetric_book() -> None:
    bid = pd.Series([100.0, 101.0])
    ask = pd.Series([100.5, 101.5])
    bid_size = pd.Series([10.0, 10.0])
    ask_size = pd.Series([10.0, 10.0])
    mp = microprice_from_book(bid, ask, bid_size, ask_size)
    expected = (bid + ask) / 2
    pd.testing.assert_series_equal(mp.rename(None), expected.rename(None))

def test_microprice_leans_toward_thicker_side() -> None:
    bid = pd.Series([100.0])
    ask = pd.Series([101.0])
    bid_size = pd.Series([100.0])
    ask_size = pd.Series([1.0])
    mp = microprice_from_book(bid, ask, bid_size, ask_size).iloc[0]
    assert mp > 100.5

def test_kyle_lambda_finite_and_nonneg_on_synthetic() -> None:
    df = _trades(5000, seed=42)
    lam, frame = kyle_lambda(df, bar='1s')
    assert np.isfinite(lam)
    assert lam >= 0
    assert 'delta_p_abs' in frame.columns

def test_trade_size_distribution_quantiles_monotone() -> None:
    df = _trades(5000)
    d = trade_size_distribution(df)
    assert d.p50 <= d.p90 <= d.p99 <= d.p999

def test_trade_size_distribution_too_few_samples_returns_nans() -> None:
    df = _trades(10)
    d = trade_size_distribution(df)
    assert np.isnan(d.p99)

def test_time_bars_have_correct_ohlcv_relations() -> None:
    df = _trades(2000)
    bars = time_bars(df, interval='1s')
    assert (bars['high'] >= bars['low']).all()
    assert (bars['high'] >= bars['open']).all()
    assert (bars['high'] >= bars['close']).all()
    assert (bars['low'] <= bars['open']).all()
    assert (bars['low'] <= bars['close']).all()
    assert (bars['volume'] > 0).all()

def test_volume_bars_conserve_total_volume() -> None:
    df = _trades(1000)
    bars = volume_bars(df, threshold_qty=df['qty'].mean() * 10)
    assert bars['volume'].sum() == pytest.approx(df['qty'].sum(), rel=1e-10)

def test_dollar_bars_conserve_total_dollar_volume() -> None:
    df = _trades(1000)
    target = (df['price'] * df['qty']).sum() / 10.0
    bars = dollar_bars(df, threshold_usd=target)
    assert bars['dollar_volume'].sum() == pytest.approx((df['price'] * df['qty']).sum(), rel=1e-10)

def test_bars_reject_non_positive_threshold() -> None:
    df = _trades(100)
    with pytest.raises(ValueError, match='positive'):
        volume_bars(df, threshold_qty=0)
