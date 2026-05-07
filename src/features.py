"""
Technical feature engineering — pure pandas/numpy, no external TA library.

We compute 16 features per asset that capture momentum, volatility, trend,
and volume dynamics. All features are designed to be stationary or
quasi-stationary; we deliberately avoid raw price levels.

Target: next-day log return.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Indicator helpers
# -----------------------------------------------------------------------------
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI, scaled to [0, 1]."""
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs  = avg_gain / (avg_loss + 1e-9)
    return (100.0 / (1.0 + rs)) / 100.0


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
         ) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger(series: pd.Series, period: int = 20, num_std: float = 2.0
              ) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid   = series.rolling(period).mean()
    std   = series.rolling(period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


def atr(high: pd.Series, low: pd.Series, close: pd.Series,
        period: int = 14) -> pd.Series:
    """Average True Range, normalized by close."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low),
         (high - prev_close).abs(),
         (low  - prev_close).abs()],
        axis=1
    ).max(axis=1)
    atr_val = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    return atr_val / (close + 1e-9)


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume, normalized to a manageable scale via z-score."""
    direction = np.sign(close.diff().fillna(0))
    obv_raw = (direction * volume).cumsum()
    # Normalise by rolling mean to keep magnitude roughly stable
    return obv_raw / (obv_raw.rolling(60, min_periods=10).std() + 1e-9)


# -----------------------------------------------------------------------------
# Main feature builder
# -----------------------------------------------------------------------------
def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    df: DataFrame with columns [open, high, low, close, volume]
    Returns: DataFrame with 16 feature columns + 'close_raw' + 'target',
             indexed identically to df.
    """
    out = pd.DataFrame(index=df.index)

    close = df['close']
    high  = df['high']
    low   = df['low']
    vol   = df['volume']

    # -- 1: log returns at multiple horizons
    log_ret = np.log(close / close.shift(1))
    out['log_ret']    = log_ret
    out['ret_5d']     = np.log(close / close.shift(5))
    out['ret_20d']    = np.log(close / close.shift(20))

    # -- 2: rolling volatility
    out['vol_5d']     = log_ret.rolling(5).std()
    out['vol_20d']    = log_ret.rolling(20).std()

    # -- 3: RSI (14)
    out['rsi_14']     = rsi(close, 14)

    # -- 4: MACD line, signal, histogram
    macd_line, signal_line, hist = macd(close)
    # Normalize by price so it transfers across tickers with different scales
    out['macd']         = macd_line   / (close + 1e-9)
    out['macd_signal']  = signal_line / (close + 1e-9)
    out['macd_hist']    = hist        / (close + 1e-9)

    # -- 5: Bollinger band position (where is the close inside the band?)
    upper, mid, lower = bollinger(close)
    out['bb_pct']     = (close - lower) / (upper - lower + 1e-9)   # in [0,1] roughly
    out['bb_width']   = (upper - lower) / (mid + 1e-9)

    # -- 6: ATR (14, normalized)
    out['atr_14']     = atr(high, low, close, 14)

    # -- 7: OBV (normalized)
    out['obv']        = obv(close, vol)

    # -- 8: volume change
    out['vol_chg']    = np.log(vol / vol.shift(1).replace(0, np.nan))

    # -- 9: close normalized by 20-day MA
    out['close_norm'] = close / close.rolling(20).mean() - 1.0

    # -- 10: high-low range as fraction of close (intra-day vol proxy)
    out['hl_range']   = (high - low) / (close + 1e-9)

    # ---- Sanity: 16 feature columns
    assert len(out.columns) == 16, f"Expected 16 features, got {len(out.columns)}"

    # ---- Target: NEXT-day log return
    out['target']    = log_ret.shift(-1)

    # ---- Keep raw close around for backtest P&L
    out['close_raw'] = close

    # Replace any inf from divisions
    out = out.replace([np.inf, -np.inf], np.nan)

    return out
