from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tankx.ml.features_micro import build_micro_features, load_cached_trades


def _synthetic_bars(n_minutes: int=120) -> pd.DataFrame:
    n = n_minutes // 5
    idx = pd.date_range('2026-05-15', periods=n, freq='5min', tz='UTC', name='timestamp')
    rng = np.random.default_rng(seed=0)
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    return pd.DataFrame({'open': close, 'high': close + 0.1, 'low': close - 0.1, 'close': close, 'volume': np.full(n, 100.0)}, index=idx)

def _synthetic_trades(n_trades: int=5000) -> pd.DataFrame:
    rng = np.random.default_rng(seed=1)
    ts = pd.date_range('2026-05-15', periods=n_trades, freq='100ms', tz='UTC')
    return pd.DataFrame({'agg_id': np.arange(n_trades), 'price': 100.0 + rng.normal(0, 0.1, n_trades), 'qty': np.abs(rng.normal(0.1, 0.05, n_trades)), 'first_id': np.arange(n_trades), 'last_id': np.arange(n_trades), 'ts': ts, 'is_buyer_maker': rng.random(n_trades) < 0.5})

def test_returns_empty_when_trades_empty() -> None:
    bars = _synthetic_bars()
    out = build_micro_features(bars, pd.DataFrame())
    assert out.empty
    assert out.index.equals(bars.index)

def test_produces_expected_columns() -> None:
    bars = _synthetic_bars(n_minutes=60)
    trades = _synthetic_trades(n_trades=2000)
    out = build_micro_features(bars, trades, ofi_window='30s')
    assert {'ofi', 'signed_vol', 'trade_count'} == set(out.columns)
    assert out.index.equals(bars.index)

def test_first_bar_feature_is_nan_due_to_shift() -> None:
    bars = _synthetic_bars(n_minutes=60)
    trades = _synthetic_trades(n_trades=2000)
    out = build_micro_features(bars, trades)
    assert out.iloc[0].isna().all()

def test_all_buys_yields_positive_signed_vol() -> None:
    bars = _synthetic_bars(n_minutes=60)
    trades = _synthetic_trades(n_trades=2000)
    trades['is_buyer_maker'] = False
    out = build_micro_features(bars, trades, ofi_window='30s').dropna()
    assert (out['signed_vol'] > 0).all()
    assert (out['ofi'] > 0.99).all()

def test_rejects_tz_naive_bars() -> None:
    n = 5
    bars = pd.DataFrame({'close': np.arange(n, dtype=float), 'volume': np.full(n, 100.0)}, index=pd.DatetimeIndex([f'2026-01-0{i + 1}' for i in range(n)]))
    trades = _synthetic_trades(n_trades=10)
    with pytest.raises(ValueError, match='tz-aware'):
        build_micro_features(bars, trades)

def test_rejects_missing_ts_column() -> None:
    bars = _synthetic_bars(n_minutes=30)
    trades = pd.DataFrame({'price': [1.0, 2.0]})
    with pytest.raises(KeyError, match='ts'):
        build_micro_features(bars, trades)

def test_load_cached_trades_returns_empty_when_no_cache(tmp_path: pytest.TempPathFactory) -> None:
    out = load_cached_trades('BTC/USDT', root=tmp_path)
    assert out.empty
