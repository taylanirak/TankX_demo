from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tankx.quality import anomaly


def test_clean_data_is_clean(synthetic_ohlcv: pd.DataFrame) -> None:
    report = anomaly.run_all_checks(synthetic_ohlcv, timeframe='5m')
    assert report.severity in {'clean', 'minor'}
    assert report.counts['ohlc_invariants'] == 0
    assert report.counts['duplicates'] == 0
    assert report.counts['gaps'] == 0
    assert report.counts['stale_quotes'] == 0

def test_ohlc_invariants_flagged_when_violated(synthetic_ohlcv: pd.DataFrame) -> None:
    df = synthetic_ohlcv.copy()
    bad_ts = df.index[10]
    df.loc[bad_ts, 'high'] = df.loc[bad_ts, 'open'] - 1.0
    out = anomaly.check_ohlc_invariants(df)
    assert bad_ts in out.index
    assert 'open_oob' in out.loc[bad_ts, 'reason']

def test_gap_detection(synthetic_ohlcv: pd.DataFrame) -> None:
    df = synthetic_ohlcv.copy()
    df = df.drop(df.index[100:105])
    out = anomaly.check_gaps(df, '5m')
    assert len(out) == 1
    assert out['gap_seconds'].iloc[0] == 6 * 300

def test_duplicates_detection(synthetic_ohlcv: pd.DataFrame) -> None:
    df = pd.concat([synthetic_ohlcv, synthetic_ohlcv.iloc[[5]]]).sort_index()
    out = anomaly.check_duplicates(df)
    assert len(out) == 2

def test_zero_volume_detection(synthetic_ohlcv: pd.DataFrame) -> None:
    df = synthetic_ohlcv.copy()
    df.loc[df.index[7], 'volume'] = 0.0
    out = anomaly.check_zero_volume(df)
    assert df.index[7] in out.index

def test_volume_outliers_no_lookahead(synthetic_ohlcv: pd.DataFrame) -> None:
    df = synthetic_ohlcv.copy()
    out_clean = anomaly.check_volume_outliers(df, window=50, z_threshold=3.0)
    spike_ts = df.index[500]
    df.loc[spike_ts, 'volume'] = 1000000000.0
    out_dirty = anomaly.check_volume_outliers(df, window=50, z_threshold=3.0)
    flagged_before_clean = set(out_clean.index[out_clean.index < spike_ts])
    flagged_before_dirty = set(out_dirty.index[out_dirty.index < spike_ts])
    assert flagged_before_clean == flagged_before_dirty
    assert spike_ts in out_dirty.index

def test_return_outliers_flagged_on_spike(synthetic_ohlcv: pd.DataFrame) -> None:
    df = synthetic_ohlcv.copy()
    spike_ts = df.index[300]
    df.loc[spike_ts, 'close'] = df.loc[df.index[299], 'close'] * 1.5
    df.loc[spike_ts, 'high'] = max(df.loc[spike_ts, 'close'], df.loc[spike_ts, 'high'])
    out = anomaly.check_return_outliers(df, window=50, z_threshold=3.0)
    assert spike_ts in out.index

def test_stale_quote_detection() -> None:
    n = 50
    idx = pd.date_range('2026-01-01', periods=n, freq='5min', tz='UTC', name='timestamp')
    rng = np.random.default_rng(seed=0)
    close = 50000 + np.cumsum(rng.normal(0, 10, n))
    open_ = np.concatenate([[close[0]], close[:-1]])
    spread = np.abs(rng.normal(0, 5, n))
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    stale = 50321.5
    for i in range(20, 24):
        open_[i] = high[i] = low[i] = close[i] = stale
    volume = np.full(n, 100.0)
    df = pd.DataFrame({'open': open_, 'high': high, 'low': low, 'close': close, 'volume': volume}, index=idx)
    out = anomaly.check_stale_quotes(df, min_run_length=3)
    assert len(out) == 4
    assert (out['run_length'] == 4).all()

def test_stale_quote_shorter_than_min_not_flagged() -> None:
    n = 30
    idx = pd.date_range('2026-01-01', periods=n, freq='5min', tz='UTC', name='timestamp')
    rng = np.random.default_rng(seed=1)
    close = 50000 + np.cumsum(rng.normal(0, 10, n))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + 5
    low = np.minimum(open_, close) - 5
    open_[10] = high[10] = low[10] = close[10] = 50100.0
    df = pd.DataFrame({'open': open_, 'high': high, 'low': low, 'close': close, 'volume': np.full(n, 100.0)}, index=idx)
    out = anomaly.check_stale_quotes(df, min_run_length=3)
    assert len(out) == 0

def test_severity_clean() -> None:
    counts = dict.fromkeys(['ohlc_invariants', 'gaps', 'duplicates', 'zero_volume', 'volume_outliers', 'return_outliers', 'stale_quotes'], 0)
    assert anomaly._severity_from_counts(counts) == 'clean'

def test_severity_major_on_invariant_violation() -> None:
    counts = {'ohlc_invariants': 1, 'gaps': 0, 'duplicates': 0, 'zero_volume': 0, 'volume_outliers': 0, 'return_outliers': 0, 'stale_quotes': 0}
    assert anomaly._severity_from_counts(counts) == 'major'

def test_severity_minor_on_few_outliers() -> None:
    counts = {'ohlc_invariants': 0, 'gaps': 0, 'duplicates': 0, 'zero_volume': 0, 'volume_outliers': 5, 'return_outliers': 3, 'stale_quotes': 0}
    assert anomaly._severity_from_counts(counts) == 'minor'

def test_quality_report_summary_frame(synthetic_ohlcv: pd.DataFrame) -> None:
    report = anomaly.run_all_checks(synthetic_ohlcv, timeframe='5m')
    summary = report.summary_frame()
    assert {'check', 'rows_flagged'} <= set(summary.columns)
    assert len(summary) == 7
    assert summary['rows_flagged'].is_monotonic_decreasing

@pytest.mark.parametrize('threshold', [3.0, 4.0, 5.0])
def test_z_threshold_monotonicity(synthetic_ohlcv: pd.DataFrame, threshold: float) -> None:
    df = synthetic_ohlcv.copy()
    df.loc[df.index[500], 'volume'] = 1000000.0
    lo = anomaly.check_volume_outliers(df, window=50, z_threshold=threshold)
    hi = anomaly.check_volume_outliers(df, window=50, z_threshold=threshold + 1.0)
    assert len(hi) <= len(lo)
