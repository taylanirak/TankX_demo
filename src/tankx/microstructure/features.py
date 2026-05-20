from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def signed_volume(trades: pd.DataFrame) -> pd.Series:
    sign = np.where(trades['is_buyer_maker'].to_numpy(), -1.0, +1.0)
    return pd.Series(sign * trades['qty'].to_numpy(dtype=np.float64), index=pd.DatetimeIndex(trades['ts']), name='signed_volume')

def order_flow_imbalance(trades: pd.DataFrame, window: str='10s') -> pd.Series:
    sv = signed_volume(trades)
    abs_v = sv.abs()
    num = sv.rolling(window).sum()
    den = abs_v.rolling(window).sum().replace(0.0, np.nan)
    return (num / den).clip(-1.0, 1.0).rename('ofi')

def effective_spread(trades: pd.DataFrame, mid: pd.Series) -> pd.Series:
    aligned_mid = mid.reindex(trades['ts'].to_numpy(), method='ffill').to_numpy()
    eff = 2.0 * np.abs(trades['price'].to_numpy() - aligned_mid) / aligned_mid
    return pd.Series(eff, index=pd.DatetimeIndex(trades['ts']), name='effective_spread')

def microprice_from_book(bid_price: pd.Series, ask_price: pd.Series, bid_size: pd.Series, ask_size: pd.Series) -> pd.Series:
    total = bid_size + ask_size
    out = (bid_price * ask_size + ask_price * bid_size) / total
    return out.where(total > 0).rename('microprice')

def kyle_lambda(trades: pd.DataFrame, *, bar: str='1min') -> tuple[float, pd.DataFrame]:
    sv = signed_volume(trades).resample(bar).sum().rename('signed_vol')
    price = trades.set_index(pd.DatetimeIndex(trades['ts']))['price']
    last_px = price.resample(bar).last().rename('close')
    df = pd.concat([sv, last_px], axis=1).dropna()
    df['delta_p_abs'] = df['close'].diff().abs()
    df['abs_signed_vol'] = df['signed_vol'].abs()
    df = df.dropna()
    if len(df) < 5:
        return (float('nan'), df)
    x = df['abs_signed_vol'].to_numpy(dtype=np.float64)
    y = df['delta_p_abs'].to_numpy(dtype=np.float64)
    denom = float((x ** 2).sum())
    if denom == 0:
        return (float('nan'), df)
    lam = float((x * y).sum() / denom)
    return (lam, df)

@dataclass(frozen=True)
class TradeSizeDistribution:
    p50: float
    p90: float
    p99: float
    p999: float
    hill_alpha: float
    whale_threshold: float

    def to_dict(self) -> dict[str, float]:
        return {'p50': self.p50, 'p90': self.p90, 'p99': self.p99, 'p999': self.p999, 'hill_alpha': self.hill_alpha, 'whale_threshold': self.whale_threshold}

def trade_size_distribution(trades: pd.DataFrame) -> TradeSizeDistribution:
    qty = trades['qty'].to_numpy(dtype=np.float64)
    qty = qty[qty > 0]
    if qty.size < 100:
        return TradeSizeDistribution(p50=float('nan'), p90=float('nan'), p99=float('nan'), p999=float('nan'), hill_alpha=float('nan'), whale_threshold=float('nan'))
    p50, p90, p99, p999 = np.quantile(qty, [0.5, 0.9, 0.99, 0.999])
    k = max(20, int(0.05 * qty.size))
    top = np.sort(qty)[-k:]
    threshold = float(top[0])
    hill_alpha = float('nan') if threshold <= 0 else float(1.0 / np.mean(np.log(top / threshold)))
    return TradeSizeDistribution(p50=float(p50), p90=float(p90), p99=float(p99), p999=float(p999), hill_alpha=hill_alpha, whale_threshold=float(p99))
__all__ = ['TradeSizeDistribution', 'effective_spread', 'kyle_lambda', 'microprice_from_book', 'order_flow_imbalance', 'signed_volume', 'trade_size_distribution']
