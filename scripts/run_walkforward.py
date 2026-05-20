from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from tankx.backtest.engine_vec import BacktestConfig
from tankx.backtest.walkforward import default_sharpe_score, make_windows, walk_forward
from tankx.config import (
    DEFAULT_FEE_BPS,
    DEFAULT_SLIPPAGE_BPS,
    DEFAULT_SYMBOL,
    DEFAULT_TIMEFRAME,
    bars_per_year,
)
from tankx.data.ohlcv import cache_location
from tankx.strategies.bollinger import BollingerMeanReversion
from tankx.strategies.ma_crossover import MACrossover

if TYPE_CHECKING:
    from collections.abc import Sequence
STRATEGIES = {'ma_crossover': {'factory': MACrossover, 'grid': {'short_window': [10, 20, 50], 'long_window': [50, 100, 200]}}, 'bollinger': {'factory': BollingerMeanReversion, 'grid': {'window': [10, 20, 50], 'num_std': [1.5, 2.0, 2.5]}}}

def _parse_args(argv: Sequence[str] | None=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--symbol', default=DEFAULT_SYMBOL)
    parser.add_argument('--timeframe', default=DEFAULT_TIMEFRAME)
    parser.add_argument('--strategy', default='ma_crossover', choices=list(STRATEGIES))
    parser.add_argument('--train-days', type=int, default=60)
    parser.add_argument('--test-days', type=int, default=14)
    parser.add_argument('--fee-bps', type=float, default=DEFAULT_FEE_BPS)
    parser.add_argument('--slippage-bps', type=float, default=DEFAULT_SLIPPAGE_BPS)
    parser.add_argument('--anchored', action='store_true', help='Anchored mode: train window starts at first bar')
    parser.add_argument('--out', default='data/walkforward_report.md', help='Output markdown report path')
    return parser.parse_args(argv)

def main(argv: Sequence[str] | None=None) -> int:
    args = _parse_args(argv)
    loc = cache_location(args.symbol, args.timeframe)
    if not loc.parquet.exists():
        print(f'No cached data at {loc.parquet}. Run scripts/fetch_data.py first.')
        return 2
    df = pd.read_parquet(loc.parquet)
    df.index = pd.DatetimeIndex(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    df.index.name = 'timestamp'
    spec = STRATEGIES[args.strategy]
    bpy = bars_per_year(args.timeframe)
    windows = make_windows(df.index, train_days=args.train_days, test_days=args.test_days, anchored=args.anchored)
    if not windows:
        print('Not enough data to construct any windows.')
        return 3
    cfg = BacktestConfig(fee_bps=args.fee_bps, slippage_bps=args.slippage_bps)
    report = walk_forward(df=df, strategy_factory=spec['factory'], param_grid=spec['grid'], windows=windows, config=cfg, score=default_sharpe_score(bpy))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_render_markdown(args, df, report), encoding='utf-8')
    print(f'Wrote walk-forward report to {out_path}')
    print(f'Windows: {len(report)}')
    print(f"Median OOS Sharpe: {report['test_score'].median():.3f}")
    print(f"Mean   OOS Sharpe: {report['test_score'].mean():.3f}")
    print(f"OOS > 0 fraction : {(report['test_score'] > 0).mean():.2%}")
    return 0

def _render_markdown(args: argparse.Namespace, df: pd.DataFrame, report: pd.DataFrame) -> str:
    median_test = float(report['test_score'].median()) if len(report) else float('nan')
    mean_test = float(report['test_score'].mean()) if len(report) else float('nan')
    mean_train = float(report['train_score'].mean()) if len(report) else float('nan')
    pos_frac = float((report['test_score'] > 0).mean()) if len(report) else float('nan')
    lines: list[str] = []
    lines.append(f'# Walk-Forward Report: {args.symbol} {args.timeframe} — {args.strategy}')
    lines.append('')
    lines.append(f'- Total bars: **{len(df):,}**')
    lines.append(f'- Span: {df.index.min()} → {df.index.max()}')
    lines.append(f'- Window: train={args.train_days}d, test={args.test_days}d, anchored={args.anchored}')
    lines.append(f'- Cost model: fee_bps={args.fee_bps}, slippage_bps={args.slippage_bps}')
    lines.append('')
    lines.append('## Summary')
    lines.append('')
    lines.append(f'- Number of windows: **{len(report)}**')
    lines.append(f'- Mean training-set Sharpe (IS):  {mean_train:.3f}')
    lines.append(f'- Mean test-set Sharpe (OOS):    {mean_test:.3f}')
    lines.append(f'- Median test-set Sharpe (OOS):  {median_test:.3f}')
    lines.append(f'- Fraction of windows with OOS Sharpe > 0: **{pos_frac:.1%}**')
    lines.append('')
    lines.append('> The IS/OOS gap is the canonical overfit indicator. A IS Sharpe of 3.0')
    lines.append('> alongside OOS Sharpe of 0.1 is the signature of an overfit grid search.')
    lines.append('')
    lines.append('## Per-window detail')
    lines.append('')
    if len(report):
        lines.append(report.to_markdown(index=False, floatfmt='.3f'))
    else:
        lines.append('_No windows._')
    return '\n'.join(lines)
if __name__ == '__main__':
    sys.exit(main())
