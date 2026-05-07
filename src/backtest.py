"""
Backtest: convert per-day model predictions into a simple long/short strategy
and report risk-adjusted performance.

Strategy:
    On each day t, sort assets by predicted return.
    Long the top-k assets (equal weight 1/k inside the long basket).
    Short the bottom-k assets (equal weight -1/k).
    Hold for one day; realise the next-day actual return; rebalance.
    Apply a small per-trade transaction cost in basis points.

Reported metrics:
    - Annualised mean return
    - Annualised volatility
    - Annualised Sharpe (mean / vol * sqrt(252))
    - Maximum drawdown
    - Hit rate (fraction of profitable days)
    - Average daily turnover

Note on framing: we DO NOT claim the strategy makes money in absolute terms.
We compare across models and against an equal-weight baseline to assess whether
the model's directional signals add risk-adjusted value relative to a naive
benchmark. This is the standard framing in academic forecasting work.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
@dataclass
class BacktestResult:
    daily_returns: np.ndarray            # (T,) realised strategy return per day
    equity_curve:  np.ndarray            # (T,) cumulative wealth from $1
    weights:       np.ndarray            # (T, N) per-day position weights
    turnover:      np.ndarray            # (T,) sum of |weight_t - weight_{t-1}|
    metrics:       dict[str, float] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Strategy
# -----------------------------------------------------------------------------
def long_short_topk_weights(preds: np.ndarray, k: int) -> np.ndarray:
    """
    Build position weights from predictions:
      +1/k for the top-k predicted assets,
      -1/k for the bottom-k,
      0 for the rest.

    Each day's gross exposure = 2 (1 long + 1 short, dollar-neutral).
    Net exposure = 0.

    If k*2 exceeds N, we automatically cap k = N // 2.
    """
    T, N = preds.shape
    if k * 2 > N:
        new_k = max(1, N // 2)
        import logging
        logging.getLogger(__name__).warning(
            f"k={k} * 2 > N={N}; capping k to {new_k}"
        )
        k = new_k

    w = np.zeros((T, N), dtype=np.float32)
    for t in range(T):
        order = np.argsort(preds[t])         # ascending
        shorts = order[:k]
        longs  = order[-k:]
        w[t, longs]  =  1.0 / k
        w[t, shorts] = -1.0 / k
    return w


def equal_weight_long_only(preds_shape: tuple[int, int]) -> np.ndarray:
    """Naive 1/N long-only benchmark, ignoring predictions."""
    T, N = preds_shape
    return np.full((T, N), 1.0 / N, dtype=np.float32)


# -----------------------------------------------------------------------------
# Backtest engine
# -----------------------------------------------------------------------------
def run_backtest(
    preds: np.ndarray,
    actuals: np.ndarray,
    *,
    strategy: str = 'long_short_topk',
    k: int = 5,
    cost_bps: float = 1.0,
    ann_factor: int = 252,
) -> BacktestResult:
    """
    preds, actuals: (T, N)  T days of next-day-return predictions and realisations
    cost_bps: round-trip transaction cost in basis points applied to turnover

    The actuals here are *log returns*. We treat them as simple-return
    approximations (valid for small daily moves <= a few %); for higher
    precision swap to (np.exp(actuals) - 1) and adjust the equity curve
    multiplicatively. We use the linear approximation to keep the math
    transparent for the demo.
    """
    T, N = preds.shape
    assert actuals.shape == (T, N), f"shape mismatch: {preds.shape} vs {actuals.shape}"

    if strategy == 'long_short_topk':
        w = long_short_topk_weights(preds, k=k)
    elif strategy == 'equal_weight_long':
        w = equal_weight_long_only(preds.shape)
    else:
        raise ValueError(f"unknown strategy {strategy}")

    # Pre-cost daily P&L
    gross_ret = (w * actuals).sum(axis=1)            # (T,)

    # Turnover = sum of |Δw_j|.  Day 0 turnover = sum of |w_0|.
    w_prev = np.zeros_like(w[0])
    turnover = np.zeros(T)
    for t in range(T):
        turnover[t] = np.abs(w[t] - w_prev).sum()
        w_prev = w[t]

    # Costs: cost_bps applied to turnover (round-trip already implied by abs diff)
    cost = (cost_bps / 1e4) * turnover
    daily_ret = gross_ret - cost

    equity = np.cumprod(1.0 + daily_ret)             # starting from $1

    metrics = compute_perf_metrics(daily_ret, ann_factor=ann_factor)
    metrics['avg_turnover'] = float(turnover.mean())

    return BacktestResult(
        daily_returns=daily_ret.astype(np.float32),
        equity_curve=equity.astype(np.float32),
        weights=w,
        turnover=turnover.astype(np.float32),
        metrics=metrics,
    )


# -----------------------------------------------------------------------------
# Performance metrics
# -----------------------------------------------------------------------------
def compute_perf_metrics(daily_ret: np.ndarray,
                         ann_factor: int = 252) -> dict[str, float]:
    if len(daily_ret) < 5:
        return {k: float('nan') for k in
                ['ann_return', 'ann_vol', 'sharpe', 'max_dd', 'hit_rate']}

    mean = daily_ret.mean()
    vol  = daily_ret.std(ddof=0)
    ann_return = mean * ann_factor
    ann_vol    = vol  * np.sqrt(ann_factor)
    sharpe     = ann_return / (ann_vol + 1e-12)

    equity = np.cumprod(1.0 + daily_ret)
    peak = np.maximum.accumulate(equity)
    drawdown = equity / peak - 1.0
    max_dd = float(drawdown.min())                 # most negative

    hit_rate = float((daily_ret > 0).mean())

    return {
        'ann_return': float(ann_return),
        'ann_vol':    float(ann_vol),
        'sharpe':     float(sharpe),
        'max_dd':     float(max_dd),
        'hit_rate':   hit_rate,
    }


# -----------------------------------------------------------------------------
# Convenience: backtest a leaderboard worth of predictions
# -----------------------------------------------------------------------------
def backtest_all(
    model_preds: dict[str, np.ndarray],
    actuals: np.ndarray,
    *,
    k: int = 5,
    cost_bps: float = 1.0,
    ann_factor: int = 252,
) -> dict[str, BacktestResult]:
    out: dict[str, BacktestResult] = {}

    # Equal-weight benchmark using predictions of the first model just for shape
    first = next(iter(model_preds.values()))
    out['equal_weight'] = run_backtest(
        first, actuals, strategy='equal_weight_long',
        k=k, cost_bps=cost_bps, ann_factor=ann_factor,
    )
    for name, preds in model_preds.items():
        out[name] = run_backtest(
            preds, actuals, strategy='long_short_topk',
            k=k, cost_bps=cost_bps, ann_factor=ann_factor,
        )
    return out


def backtest_summary_df(results: dict[str, BacktestResult]) -> pd.DataFrame:
    rows = []
    for name, res in results.items():
        rows.append({'strategy': name, **res.metrics})
    return pd.DataFrame(rows).set_index('strategy')[
        ['ann_return', 'ann_vol', 'sharpe', 'max_dd', 'hit_rate', 'avg_turnover']
    ]
