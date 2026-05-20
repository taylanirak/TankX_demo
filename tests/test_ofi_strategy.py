from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tankx.strategies.ofi_meanrev import OFIMeanReversion, build_ofi_bar_frame


def _bars_with_ofi(ofi_values: np.ndarray) -> pd.DataFrame:
    n = len(ofi_values)
    idx = pd.date_range('2026-05-15', periods=n, freq='1min', tz='UTC', name='timestamp')
    close = np.linspace(50000, 51000, n)
    return pd.DataFrame({'open': close, 'high': close + 1, 'low': close - 1, 'close': close, 'volume': np.full(n, 10.0), 'ofi': ofi_values}, index=idx)

def test_strategy_warmup_returns_zero() -> None:
    df = _bars_with_ofi(np.full(100, 0.0))
    strat = OFIMeanReversion(lookback_bars=20)
    pos = strat.generate_positions(df)
    assert (pos.iloc[:20] == 0.0).all()

def _deterministic_ofi(n: int) -> np.ndarray:
    return np.array([0.005 if i % 2 == 0 else -0.005 for i in range(n)])

def test_strategy_fades_extreme_positive_ofi() -> None:
    ofi = _deterministic_ofi(80)
    ofi[60] = 5.0
    df = _bars_with_ofi(ofi)
    strat = OFIMeanReversion(z_threshold=2.0, holding_bars=3, lookback_bars=30)
    pos = strat.generate_positions(df)
    assert pos.iloc[60] == -1.0
    assert pos.iloc[61] == -1.0
    assert pos.iloc[62] == -1.0
    assert pos.iloc[63] == 0.0

def test_strategy_fades_extreme_negative_ofi() -> None:
    ofi = _deterministic_ofi(80)
    ofi[60] = -5.0
    df = _bars_with_ofi(ofi)
    strat = OFIMeanReversion(z_threshold=2.0, holding_bars=2, lookback_bars=30)
    pos = strat.generate_positions(df)
    assert pos.iloc[60] == 1.0
    assert pos.iloc[61] == 1.0

def test_strategy_does_not_change_position_during_hold() -> None:
    ofi = _deterministic_ofi(120)
    ofi[60] = 5.0
    ofi[62] = -5.0
    df = _bars_with_ofi(ofi)
    strat = OFIMeanReversion(z_threshold=2.0, holding_bars=5, lookback_bars=30)
    pos = strat.generate_positions(df)
    assert pos.iloc[60] == -1.0
    assert pos.iloc[62] == -1.0
    assert pos.iloc[64] == -1.0
    assert pos.iloc[65] == 0.0

def test_strategy_rejects_missing_ofi_column() -> None:
    df = _bars_with_ofi(np.zeros(50)).drop(columns='ofi')
    with pytest.raises(KeyError, match='ofi'):
        OFIMeanReversion(lookback_bars=20).generate_positions(df)

def test_strategy_param_validation() -> None:
    with pytest.raises(ValueError, match='z_threshold'):
        OFIMeanReversion(z_threshold=0)
    with pytest.raises(ValueError, match='holding_bars'):
        OFIMeanReversion(holding_bars=0)
    with pytest.raises(ValueError, match='lookback_bars'):
        OFIMeanReversion(lookback_bars=5)

def test_build_ofi_bar_frame_has_ofi_column() -> None:
    rng = np.random.default_rng(seed=3)
    n = 300
    ts = pd.date_range('2026-05-15', periods=n, freq='200ms', tz='UTC')
    trades = pd.DataFrame({'agg_id': np.arange(n), 'price': 50000 + rng.normal(0, 1, n), 'qty': np.abs(rng.normal(0.05, 0.02, n)), 'first_id': np.arange(n), 'last_id': np.arange(n), 'ts': ts, 'is_buyer_maker': rng.random(n) < 0.5})
    bars = build_ofi_bar_frame(trades, bar='1s', ofi_window='500ms')
    assert 'ofi' in bars.columns
    assert bars['ofi'].between(-1.0, 1.0).all()

def test_strategy_satisfies_protocol() -> None:
    from tankx.strategies.base import VectorizedStrategy
    strat = OFIMeanReversion()
    assert isinstance(strat, VectorizedStrategy)
