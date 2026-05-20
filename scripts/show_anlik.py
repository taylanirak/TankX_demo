from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from tankx.backtest.engine_vec import BacktestConfig, run_backtest
from tankx.data.ohlcv import cache_location
from tankx.strategies.ml_directional import MLDirectionalStrategy

SYMBOLS = ('BTC/USDT', 'ETH/USDT', 'SOL/USDT')
TIMEFRAMES = ('5m', '1h')
MODELS = ('Ridge', 'XGBoost', 'LightGBM', 'LSTM', 'GRU', 'Transformer')
INITIAL = 10000.0
DAYS_BACK = 180
CFG = BacktestConfig(initial_capital=INITIAL, fee_bps=0.0, slippage_bps=0.0, execution='next_close')
PALETTE = {'Buy & Hold': '#555555', 'Ridge': '#1f77b4', 'XGBoost': '#2ca02c', 'LightGBM': '#9467bd', 'LSTM': '#d62728', 'GRU': '#ff7f0e', 'Transformer': '#17becf'}

def _load(symbol: str, tf: str) -> pd.DataFrame:
    loc = cache_location(symbol, tf)
    df = pd.read_parquet(loc.parquet)
    df.index = pd.DatetimeIndex(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    df.index.name = 'timestamp'
    end = df.index.max()
    return df.loc[end - pd.Timedelta(days=DAYS_BACK):end].copy()

def _print_block(rows: list[dict[str, object]], symbol: str, tf: str) -> dict[str, object]:
    sub = [r for r in rows if r['symbol'] == symbol and r['timeframe'] == tf]
    sub.sort(key=lambda r: r['final_equity'], reverse=True)
    print(f'\n  -- {symbol} @ {tf} --  ${INITIAL:,.0f} baslangic, son {DAYS_BACK} gun')
    winner_label = '  -> KAZANAN'
    for i, r in enumerate(sub):
        tag = winner_label if i == 0 else ''
        print(f"    {r['strategy']:<13} ${r['final_equity']:>9,.0f} ({r['return_pct']:+.2%}){tag}")
    return sub[0]

def main() -> None:
    t0 = time.perf_counter()
    rows: list[dict[str, object]] = []
    fig, axes = plt.subplots(len(SYMBOLS), len(TIMEFRAMES), figsize=(13, 10), squeeze=False)
    for r_idx, symbol in enumerate(SYMBOLS):
        for c_idx, tf in enumerate(TIMEFRAMES):
            print(f'\n=== {symbol} @ {tf} ===')
            df = _load(symbol, tf)
            n = len(df)
            price_move = df['close'].iloc[-1] / df['close'].iloc[0] - 1.0
            print(f'  {n:,} bars from {df.index.min().date()} to {df.index.max().date()}  (price {price_move:+.2%})')
            ax = axes[r_idx][c_idx]
            bh_pos = pd.Series(1.0, index=df.index, name='position')
            bh = run_backtest(df, bh_pos, CFG)
            rows.append({'symbol': symbol, 'timeframe': tf, 'strategy': 'Buy & Hold', 'final_equity': float(bh.equity.iloc[-1]), 'return_pct': float(bh.equity.iloc[-1] / INITIAL - 1.0), 'equity': bh.equity})
            ax.plot(bh.equity.index, bh.equity.values, label='Buy & Hold', color=PALETTE['Buy & Hold'], linewidth=1.2, linestyle='--')
            for model_name in MODELS:
                t_start = time.perf_counter()
                try:
                    strat = MLDirectionalStrategy(model_name=model_name, target_kind='direction', train_fraction=0.6, direction_band=0.0)
                    positions = strat.generate_positions(df)
                    res = run_backtest(df, positions, CFG)
                except Exception as exc:
                    print(f'  ! {model_name} FAILED: {exc}')
                    continue
                elapsed = time.perf_counter() - t_start
                final = float(res.equity.iloc[-1])
                ret = final / INITIAL - 1.0
                rows.append({'symbol': symbol, 'timeframe': tf, 'strategy': model_name, 'final_equity': final, 'return_pct': ret, 'equity': res.equity})
                print(f'  {model_name:<13} -> ${final:>9,.0f} ({ret:+.2%})  [{elapsed:>5.1f}s]')
                ax.plot(res.equity.index, res.equity.values, label=model_name, color=PALETTE[model_name], linewidth=1.2)
            ax.axhline(y=INITIAL, color='black', linestyle=':', linewidth=0.7)
            ax.set_title(f'{symbol} @ {tf}', fontweight='bold')
            ax.set_ylabel('Equity (USD)')
            ax.grid(alpha=0.3)
            ax.legend(loc='best', fontsize=7.5)
    fig.suptitle(f'Anlik Demo: {len(MODELS)} ML model x {len(SYMBOLS)} coin x {len(TIMEFRAMES)} timeframe — komisyonsuz, son {DAYS_BACK} gun, ${INITIAL:,.0f} baslangic', fontsize=13, fontweight='bold')
    fig.tight_layout()
    out_png = Path('data/anlik_demo.png')
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130, bbox_inches='tight')
    print('\n' + '=' * 70)
    print('SIRALI OZET (her panelde en yuksekten en dusuge)')
    print('=' * 70)
    panel_winners: list[dict[str, object]] = []
    for symbol in SYMBOLS:
        for tf in TIMEFRAMES:
            w = _print_block(rows, symbol, tf)
            panel_winners.append(w)
    overall = max(panel_winners, key=lambda r: r['final_equity'])
    print('\n' + '=' * 70)
    print('HEADLINE')
    print('=' * 70)
    print(f"  Tum kombinasyonlar arasinda en iyi: {overall['strategy']} on {overall['symbol']} @ {overall['timeframe']}")
    print(f"  Bu strateji ile son {DAYS_BACK} gun al-sat yapsaydin: ${INITIAL:,.0f} -> ${overall['final_equity']:,.0f} ({overall['return_pct']:+.2%})")
    print(f'\nChart -> {out_png.resolve()}')
    print(f'Toplam sure: {time.perf_counter() - t0:.0f}s')
if __name__ == '__main__':
    main()
