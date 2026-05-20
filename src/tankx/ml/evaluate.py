from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tankx.backtest.engine_vec import BacktestConfig, run_backtest
from tankx.backtest.walkforward import WFWindow, make_windows
from tankx.config import DEFAULT_FEE_BPS, DEFAULT_SLIPPAGE_BPS, bars_per_year
from tankx.ml.features import FeatureConfig
from tankx.ml.models.base import TargetKind
from tankx.ml.models.registry import MODEL_REGISTRY, list_models
from tankx.ml.training import TrainingResult, fit_predict_walkforward
from tankx.stats.metrics import annualized_sharpe, max_drawdown


@dataclass(frozen=True)
class BenchmarkSpec:
    symbol: str
    timeframe: str
    target_kind: TargetKind
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)
    train_days: int = 14
    test_days: int = 3
    fee_bps: float = DEFAULT_FEE_BPS
    fee_bps_maker: float = 0.0
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS
    threshold: float = 0.0
    direction_band: float = 0.05
    return_band: float = 0.0
    assume_maker: bool = False

def _classification_metrics(pred: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    if len(actual) == 0:
        return {'n_test': 0.0, 'accuracy': float('nan'), 'f1': float('nan'), 'auc': float('nan')}
    pred_label = (pred > 0.5).astype(np.float64)
    actual_label = (actual > 0.5).astype(np.float64)
    accuracy = float((pred_label == actual_label).mean())
    tp = float(((pred_label == 1) & (actual_label == 1)).sum())
    fp = float(((pred_label == 1) & (actual_label == 0)).sum())
    fn = float(((pred_label == 0) & (actual_label == 1)).sum())
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(actual_label, pred)) if len(np.unique(actual_label)) > 1 else float('nan')
    except Exception:
        auc = float('nan')
    return {'n_test': float(len(actual)), 'accuracy': accuracy, 'f1': float(f1), 'auc': auc}

def _triple_barrier_metrics(pred: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    if len(actual) == 0:
        return {'n_test': 0.0, 'accuracy': float('nan'), 'f1': float('nan'), 'auc': float('nan'), 'timeout_frac': float('nan')}
    mask = actual != 0
    n_signal = int(mask.sum())
    timeout_frac = 1.0 - n_signal / len(actual)
    if n_signal == 0:
        return {'n_test': float(len(actual)), 'accuracy': float('nan'), 'f1': float('nan'), 'auc': float('nan'), 'timeout_frac': timeout_frac}
    pred_sign = np.sign(pred[mask])
    actual_sign = np.sign(actual[mask])
    accuracy = float((pred_sign == actual_sign).mean())
    tp = float(((pred_sign == 1) & (actual_sign == 1)).sum())
    fp = float(((pred_sign == 1) & (actual_sign == -1)).sum())
    fn = float(((pred_sign == -1) & (actual_sign == 1)).sum())
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {'n_test': float(len(actual)), 'accuracy': accuracy, 'f1': float(f1), 'auc': float('nan'), 'timeout_frac': float(timeout_frac)}

def _regression_metrics(pred: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    if len(actual) == 0:
        return {'n_test': 0.0, 'rmse': float('nan'), 'mae': float('nan'), 'r2': float('nan')}
    err = pred - actual
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    return {'n_test': float(len(actual)), 'rmse': rmse, 'mae': mae, 'r2': r2}

def predictions_to_positions(pred: pd.Series, kind: TargetKind, *, direction_band: float=0.0, return_band: float=0.0) -> pd.Series:
    arr = pred.to_numpy(dtype=np.float64)
    if kind == 'direction':
        high = 0.5 + direction_band
        low = 0.5 - direction_band
    else:
        high = return_band
        low = -return_band
    positions = np.where(arr > high, 1.0, np.where(arr < low, -1.0, 0.0))
    return pd.Series(positions, index=pred.index, name='position')

def _strategy_metrics(df: pd.DataFrame, positions: pd.Series, *, fee_bps: float, slippage_bps: float, bpy: int, fee_bps_maker: float=0.0, assume_maker: bool=False) -> dict[str, float]:
    aligned = positions.reindex(df.index, fill_value=0.0)
    if (aligned == 0).all():
        return {'strategy_sharpe': 0.0, 'strategy_total_return': 0.0, 'strategy_max_drawdown': 0.0, 'strategy_hit_rate': float('nan'), 'turnover': 0.0}
    cfg = BacktestConfig(fee_bps=fee_bps, slippage_bps=slippage_bps, fee_bps_maker=fee_bps_maker, assume_maker=assume_maker, execution='next_close')
    result = run_backtest(df, aligned, cfg)
    sharpe = annualized_sharpe(result.net_returns, bpy)
    total_return = float(result.equity.iloc[-1] / result.equity.iloc[0] - 1.0)
    mdd, _, _ = max_drawdown(result.equity)
    if len(result.trades) > 0:
        pnls = result.trades['exit_equity'] / result.trades['entry_equity'] - 1.0
        hit_rate = float((pnls > 0).mean())
    else:
        hit_rate = float('nan')
    turnover = float(aligned.diff().abs().fillna(0.0).sum())
    return {'strategy_sharpe': float(sharpe) if not math.isnan(sharpe) else 0.0, 'strategy_total_return': total_return, 'strategy_max_drawdown': float(mdd), 'strategy_hit_rate': hit_rate, 'turnover': turnover}

def evaluate_one(df: pd.DataFrame, spec: BenchmarkSpec, model_name: str, *, windows: Iterable[WFWindow] | None=None) -> tuple[dict[str, object], TrainingResult]:
    if windows is None:
        windows = make_windows(pd.DatetimeIndex(df.index), train_days=spec.train_days, test_days=spec.test_days)
    factory = MODEL_REGISTRY[model_name]
    training_result = fit_predict_walkforward(df=df, factory=factory, target_kind=spec.target_kind, windows=windows, feature_config=spec.feature_config, model_name=model_name, symbol=spec.symbol)
    pred = training_result.predictions
    actual = training_result.actuals
    if spec.target_kind == 'direction':
        base_metrics = _classification_metrics(pred.to_numpy(), actual.to_numpy())
    elif spec.target_kind == 'triple_barrier':
        base_metrics = _triple_barrier_metrics(pred.to_numpy(), actual.to_numpy())
    else:
        base_metrics = _regression_metrics(pred.to_numpy(), actual.to_numpy())
    positions = predictions_to_positions(pred, kind=spec.target_kind, direction_band=spec.direction_band, return_band=spec.return_band)
    bpy = bars_per_year(spec.timeframe)
    strat_metrics = _strategy_metrics(df, positions, fee_bps=spec.fee_bps, slippage_bps=spec.slippage_bps, bpy=bpy, fee_bps_maker=spec.fee_bps_maker, assume_maker=spec.assume_maker)
    row: dict[str, object] = {'symbol': spec.symbol, 'timeframe': spec.timeframe, 'target': spec.target_kind, 'model': model_name, 'fee_mode': 'maker' if spec.assume_maker else 'taker', **base_metrics, **strat_metrics}
    return (row, training_result)

def evaluate_all(df: pd.DataFrame, spec: BenchmarkSpec, *, model_names: Iterable[str] | None=None, windows: Iterable[WFWindow] | None=None) -> tuple[pd.DataFrame, dict[str, TrainingResult]]:
    if model_names is None:
        model_names = list_models()
    if windows is None:
        windows = list(make_windows(pd.DatetimeIndex(df.index), train_days=spec.train_days, test_days=spec.test_days))
    rows: list[dict[str, object]] = []
    trainings: dict[str, TrainingResult] = {}
    for name in model_names:
        row, training_result = evaluate_one(df, spec, name, windows=windows)
        rows.append(row)
        trainings[name] = training_result
    return (pd.DataFrame(rows), trainings)
__all__ = ['BenchmarkSpec', 'evaluate_all', 'evaluate_one', 'predictions_to_positions']
