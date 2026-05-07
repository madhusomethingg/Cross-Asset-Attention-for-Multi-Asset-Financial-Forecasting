"""
Evaluation metrics for multi-asset return forecasting.

We compute three families of metrics:

  STATISTICAL ACCURACY (per-asset, then averaged):
    - RMSE  : root mean squared error
    - MAE   : mean absolute error

  DIRECTIONAL / RANKING:
    - DA    : directional accuracy = P(sign(pred) == sign(actual))
    - IC    : information coefficient = mean over time of cross-sectional
              Spearman rank correlation between predicted and realized returns
              on the same day. This is the standard "alpha quality" metric in
              quant finance.

  STATISTICAL TESTS:
    - Diebold-Mariano test of equal predictive accuracy between two models,
      using squared-error loss (Harvey small-sample correction).

The economic metrics (Sharpe, MaxDD, etc.) live in `backtest.py`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# -----------------------------------------------------------------------------
# Pointwise accuracy
# -----------------------------------------------------------------------------
def rmse(preds: np.ndarray, actuals: np.ndarray) -> float:
    """Mean across assets of per-asset RMSE."""
    err = preds - actuals
    per_asset = np.sqrt(np.mean(err ** 2, axis=0))     # (N,)
    return float(per_asset.mean())


def mae(preds: np.ndarray, actuals: np.ndarray) -> float:
    err = preds - actuals
    per_asset = np.mean(np.abs(err), axis=0)           # (N,)
    return float(per_asset.mean())


# -----------------------------------------------------------------------------
# Directional accuracy
# -----------------------------------------------------------------------------
def directional_accuracy(preds: np.ndarray, actuals: np.ndarray) -> float:
    """
    Fraction of (day, asset) pairs where sign(pred) == sign(actual).
    Excludes (rare) ties where either is exactly zero.
    """
    p_sign = np.sign(preds)
    a_sign = np.sign(actuals)
    valid = (p_sign != 0) & (a_sign != 0)
    if not valid.any():
        return float('nan')
    return float((p_sign[valid] == a_sign[valid]).mean())


def directional_accuracy_per_asset(preds: np.ndarray,
                                   actuals: np.ndarray) -> np.ndarray:
    """Per-asset DA -- useful for diagnosing which assets the model handles well."""
    p_sign = np.sign(preds); a_sign = np.sign(actuals)
    out = np.zeros(preds.shape[1])
    for j in range(preds.shape[1]):
        valid = (p_sign[:, j] != 0) & (a_sign[:, j] != 0)
        out[j] = (p_sign[valid, j] == a_sign[valid, j]).mean() if valid.any() else np.nan
    return out


# -----------------------------------------------------------------------------
# Information Coefficient (cross-sectional)
# -----------------------------------------------------------------------------
def information_coefficient(preds: np.ndarray, actuals: np.ndarray) -> float:
    """
    Daily cross-sectional Spearman rank correlation, averaged over days.
    A model that ranks the assets in the right order on each day will score high.
    """
    T, N = preds.shape
    if N < 3:
        return float('nan')
    daily_ic = []
    for t in range(T):
        # if preds or actuals are constant, scipy returns NaN -> skip
        if np.std(preds[t]) < 1e-12 or np.std(actuals[t]) < 1e-12:
            continue
        rho, _ = stats.spearmanr(preds[t], actuals[t])
        if not np.isnan(rho):
            daily_ic.append(rho)
    return float(np.mean(daily_ic)) if daily_ic else float('nan')


def information_coefficient_series(preds: np.ndarray,
                                   actuals: np.ndarray) -> np.ndarray:
    """Daily IC time series -- useful for plots and ICIR computation."""
    T = preds.shape[0]
    out = np.full(T, np.nan)
    for t in range(T):
        if np.std(preds[t]) < 1e-12 or np.std(actuals[t]) < 1e-12:
            continue
        rho, _ = stats.spearmanr(preds[t], actuals[t])
        out[t] = rho
    return out


def ic_ir(preds: np.ndarray, actuals: np.ndarray) -> float:
    """IC information ratio = mean(IC_t) / std(IC_t).  Higher = more consistent."""
    ics = information_coefficient_series(preds, actuals)
    valid = ~np.isnan(ics)
    if valid.sum() < 5:
        return float('nan')
    return float(np.mean(ics[valid]) / (np.std(ics[valid]) + 1e-12))


# -----------------------------------------------------------------------------
# Diebold-Mariano test
# -----------------------------------------------------------------------------
def diebold_mariano(
    preds_a: np.ndarray, preds_b: np.ndarray, actuals: np.ndarray, h: int = 1,
) -> tuple[float, float]:
    """
    Diebold-Mariano test of equal predictive accuracy with Harvey-Leybourne-Newbold
    small-sample correction. Squared-error loss.

    H0: model A and model B have equal MSE.
    Returns: (DM_statistic, two-sided_p_value)

    Negative statistic favours A (lower MSE), positive favours B.

    Reference: Diebold & Mariano (1995); Harvey, Leybourne, Newbold (1997).
    """
    e_a = (preds_a - actuals).reshape(-1) ** 2
    e_b = (preds_b - actuals).reshape(-1) ** 2
    d = e_a - e_b
    T = len(d)
    if T < 10:
        return float('nan'), float('nan')

    mean_d = d.mean()
    # Long-run variance: include autocovariances up to lag h-1 (for h-step ahead, h-1 lags)
    var_d = np.var(d, ddof=0)
    for lag in range(1, h):
        cov = np.cov(d[lag:], d[:-lag], ddof=0)[0, 1]
        var_d += 2 * cov
    var_d = max(var_d, 1e-12)

    dm_stat = mean_d / np.sqrt(var_d / T)

    # Harvey-Leybourne-Newbold small-sample correction
    correction = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    dm_stat *= correction

    # Use Student-t with T-1 dof for small-sample p-value
    p_value = 2 * (1 - stats.t.cdf(abs(dm_stat), df=T - 1))
    return float(dm_stat), float(p_value)


# -----------------------------------------------------------------------------
# Top-level summary table builder
# -----------------------------------------------------------------------------
def evaluate_predictions(
    preds: np.ndarray, actuals: np.ndarray,
) -> dict[str, float]:
    """One-shot computation of all statistical metrics for a single model."""
    return {
        'RMSE':  rmse(preds, actuals),
        'MAE':   mae(preds, actuals),
        'DA':    directional_accuracy(preds, actuals),
        'IC':    information_coefficient(preds, actuals),
        'IC_IR': ic_ir(preds, actuals),
    }


def build_leaderboard(
    model_preds: dict[str, np.ndarray],
    actuals: np.ndarray,
) -> pd.DataFrame:
    """
    model_preds: {model_name: (T, N) preds}
    actuals:     (T, N)
    Returns a DataFrame with one row per model and columns for each metric.
    """
    rows = []
    for name, preds in model_preds.items():
        m = evaluate_predictions(preds, actuals)
        m['model'] = name
        rows.append(m)
    df = pd.DataFrame(rows).set_index('model')
    return df[['RMSE', 'MAE', 'DA', 'IC', 'IC_IR']]
