from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

import pandas as pd

from tankx.config import timeframe_to_timedelta_seconds

DEFAULT_OUTLIER_WINDOW: Final[int] = 288
DEFAULT_Z_THRESHOLD: Final[float] = 4.0
DEFAULT_STALE_RUN_LENGTH: Final[int] = 3
Severity = Literal['clean', 'minor', 'major']

@dataclass(frozen=True)
class QualityReport:
    counts: dict[str, int]
    findings: dict[str, pd.DataFrame]
    severity: Severity = 'clean'
    total_rows: int = 0

    def summary_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{'check': k, 'rows_flagged': v} for k, v in self.counts.items()]).sort_values('rows_flagged', ascending=False).reset_index(drop=True)

def check_ohlc_invariants(df: pd.DataFrame) -> pd.DataFrame:
    bad_lh = df['low'] > df['high']
    bad_o = (df['open'] < df['low']) | (df['open'] > df['high'])
    bad_c = (df['close'] < df['low']) | (df['close'] > df['high'])
    mask = bad_lh | bad_o | bad_c
    out = df.loc[mask].copy()
    reasons: list[str] = []
    for ts in out.index:
        r: list[str] = []
        if bad_lh.loc[ts]:
            r.append('low>high')
        if bad_o.loc[ts]:
            r.append('open_oob')
        if bad_c.loc[ts]:
            r.append('close_oob')
        reasons.append(','.join(r))
    out['reason'] = reasons
    return out

def check_gaps(df: pd.DataFrame, timeframe: str, *, tolerance_factor: float=1.5) -> pd.DataFrame:
    expected_seconds = timeframe_to_timedelta_seconds(timeframe)
    deltas = df.index.to_series().diff().dt.total_seconds()
    mask = deltas > tolerance_factor * expected_seconds
    out = df.loc[mask, []].copy()
    out['previous'] = df.index.to_series().shift(1).loc[mask].values
    out['gap_seconds'] = deltas.loc[mask].astype(int).values
    return out

def check_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    mask = df.index.duplicated(keep=False)
    return df.loc[mask].copy()

def check_zero_volume(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df['volume'] == 0].copy()

def check_volume_outliers(df: pd.DataFrame, *, window: int=DEFAULT_OUTLIER_WINDOW, z_threshold: float=DEFAULT_Z_THRESHOLD) -> pd.DataFrame:
    vol = df['volume']
    mean = vol.shift(1).rolling(window=window, min_periods=window // 2).mean()
    std = vol.shift(1).rolling(window=window, min_periods=window // 2).std()
    z = (vol - mean) / std
    mask = z.abs() > z_threshold
    out = df.loc[mask.fillna(False)].copy()
    out['z_score'] = z.loc[out.index]
    out['reason'] = 'volume_outlier'
    return out

def check_return_outliers(df: pd.DataFrame, *, window: int=DEFAULT_OUTLIER_WINDOW, z_threshold: float=DEFAULT_Z_THRESHOLD) -> pd.DataFrame:
    ret = df['close'].pct_change()
    mean = ret.shift(1).rolling(window=window, min_periods=window // 2).mean()
    std = ret.shift(1).rolling(window=window, min_periods=window // 2).std()
    z = (ret - mean) / std
    mask = z.abs() > z_threshold
    out = df.loc[mask.fillna(False)].copy()
    out['return'] = ret.loc[out.index]
    out['z_score'] = z.loc[out.index]
    out['reason'] = 'return_outlier'
    return out

def check_stale_quotes(df: pd.DataFrame, *, min_run_length: int=DEFAULT_STALE_RUN_LENGTH) -> pd.DataFrame:
    o, h, low_, c = (df['open'], df['high'], df['low'], df['close'])
    stale_bar = (o == h) & (h == low_) & (low_ == c)
    group = (stale_bar != stale_bar.shift(1, fill_value=False)).cumsum()
    run_lengths = stale_bar.groupby(group).transform('sum')
    mask = stale_bar & (run_lengths >= min_run_length)
    out = df.loc[mask].copy()
    out['run_length'] = run_lengths.loc[mask].astype(int)
    out['reason'] = 'stale_quote'
    return out

def _severity_from_counts(counts: dict[str, int]) -> Severity:
    total = sum(counts.values())
    if total == 0:
        return 'clean'
    if counts.get('ohlc_invariants', 0) > 0 or counts.get('duplicates', 0) > 0:
        return 'major'
    if total > 100:
        return 'major'
    return 'minor'

def run_all_checks(df: pd.DataFrame, timeframe: str, *, outlier_window: int=DEFAULT_OUTLIER_WINDOW, z_threshold: float=DEFAULT_Z_THRESHOLD, stale_run_length: int=DEFAULT_STALE_RUN_LENGTH) -> QualityReport:
    findings: dict[str, pd.DataFrame] = {'ohlc_invariants': check_ohlc_invariants(df), 'gaps': check_gaps(df, timeframe), 'duplicates': check_duplicates(df), 'zero_volume': check_zero_volume(df), 'volume_outliers': check_volume_outliers(df, window=outlier_window, z_threshold=z_threshold), 'return_outliers': check_return_outliers(df, window=outlier_window, z_threshold=z_threshold), 'stale_quotes': check_stale_quotes(df, min_run_length=stale_run_length)}
    counts = {k: len(v) for k, v in findings.items()}
    return QualityReport(counts=counts, findings=findings, severity=_severity_from_counts(counts), total_rows=len(df))
__all__ = ['DEFAULT_OUTLIER_WINDOW', 'DEFAULT_STALE_RUN_LENGTH', 'DEFAULT_Z_THRESHOLD', 'QualityReport', 'Severity', 'check_duplicates', 'check_gaps', 'check_ohlc_invariants', 'check_return_outliers', 'check_stale_quotes', 'check_volume_outliers', 'check_zero_volume', 'run_all_checks']
