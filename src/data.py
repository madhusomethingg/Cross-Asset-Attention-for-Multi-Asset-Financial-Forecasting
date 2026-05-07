"""
Data loading and preparation for the multi-asset forecasting pipeline.

Responsibilities:
- Download daily OHLCV data via yfinance for the configured ticker universe
- Forward/back-fill missing values; drop tickers with excessive missingness
- Align all tickers to a shared trading-day index
- Apply chronological train/val/test split (no leakage)
- Z-score normalize features using *training-set statistics only*

Output: a `DataBundle` containing:
  - `X_train, X_val, X_test`  shape (T, N, F)  -- feature tensors
  - `y_train, y_val, y_test`  shape (T, N)     -- next-day log returns
  - `tickers`                 list[str]
  - `feature_cols`            list[str]
  - `dates_train/val/test`    pd.DatetimeIndex
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler

from src.features import build_feature_table

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Container
# -----------------------------------------------------------------------------
@dataclass
class DataBundle:
    """Holds aligned feature/target arrays plus metadata."""
    X_train: np.ndarray   # (T_train, N, F)
    X_val:   np.ndarray
    X_test:  np.ndarray
    y_train: np.ndarray   # (T_train, N)
    y_val:   np.ndarray
    y_test:  np.ndarray
    tickers: list[str]
    feature_cols: list[str]
    dates_train: pd.DatetimeIndex
    dates_val:   pd.DatetimeIndex
    dates_test:  pd.DatetimeIndex
    # raw (unscaled) close prices on the aligned index — needed for backtest
    close_prices: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def n_assets(self) -> int:
        return len(self.tickers)

    @property
    def n_features(self) -> int:
        return len(self.feature_cols)

    def summary(self) -> str:
        return (
            f"DataBundle:\n"
            f"  Tickers   : {self.n_assets}\n"
            f"  Features  : {self.n_features}\n"
            f"  Train     : {self.X_train.shape}  "
            f"({self.dates_train[0].date()} -> {self.dates_train[-1].date()})\n"
            f"  Val       : {self.X_val.shape}    "
            f"({self.dates_val[0].date()} -> {self.dates_val[-1].date()})\n"
            f"  Test      : {self.X_test.shape}   "
            f"({self.dates_test[0].date()} -> {self.dates_test[-1].date()})\n"
        )


# -----------------------------------------------------------------------------
# Download
# -----------------------------------------------------------------------------
def download_ohlcv(tickers: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    """
    Pulls daily OHLCV for each ticker via yfinance.
    Returns a dict {ticker: DataFrame[Open, High, Low, Close, Volume]}.
    """
    logger.info(f"Downloading {len(tickers)} tickers from {start} to {end}")
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=True,
        group_by='ticker',
    )

    # yfinance returns a multi-level column DataFrame when given a list of tickers.
    # Convert to a clean dict of per-ticker DataFrames.
    out: dict[str, pd.DataFrame] = {}
    for t in tickers:
        if t not in raw.columns.get_level_values(0):
            logger.warning(f"  Ticker {t} not present in yfinance response; skipping")
            continue
        df = raw[t][['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df.columns = ['open', 'high', 'low', 'close', 'volume']
        out[t] = df
    return out


# -----------------------------------------------------------------------------
# Cleaning + feature engineering per ticker
# -----------------------------------------------------------------------------
def clean_and_featurize(
    raw: dict[str, pd.DataFrame],
    max_missing_frac: float = 0.05,
) -> dict[str, pd.DataFrame]:
    """
    Forward-fills then back-fills missing OHLCV values, drops tickers that
    are too sparse, and computes the technical-feature panel for each ticker.

    Returns: {ticker: DataFrame[features..., target]}
    """
    feature_dict: dict[str, pd.DataFrame] = {}
    for t, df in raw.items():
        # initial cleaning
        df = df.ffill().bfill()
        miss_frac = df['close'].isna().mean()
        if miss_frac > max_missing_frac:
            logger.warning(f"  Dropping {t}: {miss_frac:.1%} missing in close")
            continue
        # build features (returns DataFrame indexed same as df)
        feats = build_feature_table(df)
        feats = feats.dropna()           # drop warmup rows from rolling stats
        if len(feats) < 200:
            logger.warning(f"  Dropping {t}: only {len(feats)} clean rows")
            continue
        feature_dict[t] = feats
        logger.info(f"  {t}: {feats.shape[0]} clean rows, "
                    f"{feats.shape[1]-1} features")
    return feature_dict


# -----------------------------------------------------------------------------
# Alignment and tensor construction
# -----------------------------------------------------------------------------
def align_panel(feature_dict: dict[str, pd.DataFrame]) -> tuple[
    list[str], pd.DatetimeIndex, list[str]
]:
    """Find the date index common to ALL tickers and the feature column order."""
    tickers = list(feature_dict.keys())
    common = feature_dict[tickers[0]].index
    for t in tickers[1:]:
        common = common.intersection(feature_dict[t].index)

    # Exclude target and close_raw (kept for backtesting, not a feature)
    non_feature_cols = {'target', 'close_raw'}
    feature_cols = [c for c in feature_dict[tickers[0]].columns
                    if c not in non_feature_cols]
    return tickers, common, feature_cols


def stack_to_array(
    feature_dict: dict[str, pd.DataFrame],
    tickers: list[str],
    dates: pd.DatetimeIndex,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build:
      X: (T, N, F)  feature tensor
      y: (T, N)     target (next-day log return) tensor
    """
    T, N, F = len(dates), len(tickers), len(feature_cols)
    X = np.zeros((T, N, F), dtype=np.float32)
    y = np.zeros((T, N),    dtype=np.float32)
    for j, t in enumerate(tickers):
        df = feature_dict[t].loc[dates]
        X[:, j, :] = df[feature_cols].values
        y[:, j]    = df['target'].values
    return X, y


# -----------------------------------------------------------------------------
# Splitting + normalization
# -----------------------------------------------------------------------------
def split_indices(
    dates: pd.DatetimeIndex, train_frac: float, val_frac: float,
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex]:
    n = len(dates)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)
    return (
        dates[:n_train],
        dates[n_train:n_train + n_val],
        dates[n_train + n_val:],
    )


def normalize(
    X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[StandardScaler]]:
    """
    Per-(asset, feature) z-score normalization.
    Scalers are fit ONLY on the training data to avoid leakage.

    Returns the three scaled arrays and the list of fitted scalers
    (one StandardScaler per asset; each scaler has F-dim mean/std).
    """
    T_tr, N, F = X_train.shape
    scalers: list[StandardScaler] = []

    Xtr = X_train.copy()
    Xva = X_val.copy()
    Xte = X_test.copy()

    for j in range(N):
        sc = StandardScaler()
        Xtr[:, j, :] = sc.fit_transform(X_train[:, j, :])
        Xva[:, j, :] = sc.transform(X_val[:, j, :])
        Xte[:, j, :] = sc.transform(X_test[:, j, :])
        scalers.append(sc)

    return Xtr, Xva, Xte, scalers


# -----------------------------------------------------------------------------
# Top-level pipeline
# -----------------------------------------------------------------------------
def build_data_bundle(
    tickers: list[str],
    start_date: str,
    end_date: str,
    train_frac: float = 0.70,
    val_frac:   float = 0.15,
) -> DataBundle:
    """
    End-to-end pipeline: download -> clean -> featurize -> align -> split
    -> normalize -> assemble DataBundle.
    """
    raw = download_ohlcv(tickers, start_date, end_date)
    feature_dict = clean_and_featurize(raw)

    if len(feature_dict) < 2:
        raise RuntimeError("Too few tickers survived cleaning. Check date range.")

    tickers_kept, common_dates, feature_cols = align_panel(feature_dict)
    logger.info(f"Aligned {len(tickers_kept)} tickers over "
                f"{len(common_dates)} dates with {len(feature_cols)} features")

    X, y = stack_to_array(feature_dict, tickers_kept, common_dates, feature_cols)

    # raw close prices, retained for backtest P&L
    close_prices = pd.DataFrame(
        {t: feature_dict[t].loc[common_dates, 'close_raw'] for t in tickers_kept},
        index=common_dates,
    )

    # split
    dates_tr, dates_va, dates_te = split_indices(common_dates, train_frac, val_frac)
    n_tr, n_va = len(dates_tr), len(dates_va)
    Xtr_raw = X[:n_tr]
    Xva_raw = X[n_tr:n_tr + n_va]
    Xte_raw = X[n_tr + n_va:]
    ytr = y[:n_tr]
    yva = y[n_tr:n_tr + n_va]
    yte = y[n_tr + n_va:]

    # normalize
    Xtr, Xva, Xte, _ = normalize(Xtr_raw, Xva_raw, Xte_raw)

    return DataBundle(
        X_train=Xtr, X_val=Xva, X_test=Xte,
        y_train=ytr, y_val=yva, y_test=yte,
        tickers=tickers_kept, feature_cols=feature_cols,
        dates_train=dates_tr, dates_val=dates_va, dates_test=dates_te,
        close_prices=close_prices,
    )


# -----------------------------------------------------------------------------
# Save / load (numpy + parquet)
# -----------------------------------------------------------------------------
def save_bundle(bundle: DataBundle, path: Path) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        path / 'arrays.npz',
        X_train=bundle.X_train, X_val=bundle.X_val, X_test=bundle.X_test,
        y_train=bundle.y_train, y_val=bundle.y_val, y_test=bundle.y_test,
    )
    bundle.close_prices.to_csv(path / 'close_prices.csv')

    meta = {
        'tickers': bundle.tickers,
        'feature_cols': bundle.feature_cols,
        'dates_train': [d.isoformat() for d in bundle.dates_train],
        'dates_val':   [d.isoformat() for d in bundle.dates_val],
        'dates_test':  [d.isoformat() for d in bundle.dates_test],
    }
    import json
    with open(path / 'meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Saved DataBundle to {path}")


def load_bundle(path: Path) -> DataBundle:
    path = Path(path)
    arrs = np.load(path / 'arrays.npz')
    close_prices = pd.read_csv(
        path / 'close_prices.csv', index_col=0, parse_dates=True,
    )

    import json
    with open(path / 'meta.json') as f:
        meta = json.load(f)

    return DataBundle(
        X_train=arrs['X_train'], X_val=arrs['X_val'], X_test=arrs['X_test'],
        y_train=arrs['y_train'], y_val=arrs['y_val'], y_test=arrs['y_test'],
        tickers=meta['tickers'], feature_cols=meta['feature_cols'],
        dates_train=pd.DatetimeIndex(meta['dates_train']),
        dates_val=pd.DatetimeIndex(meta['dates_val']),
        dates_test=pd.DatetimeIndex(meta['dates_test']),
        close_prices=close_prices,
    )
