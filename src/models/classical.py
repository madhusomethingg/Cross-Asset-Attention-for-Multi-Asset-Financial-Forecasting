"""
Classical / traditional baselines: Ridge, ARIMA, XGBoost.

All three expose the same shape of predictions as the DL models: aligned with
the DL test-set predictions starting at index `lookback` of the test set.
This makes the leaderboard comparison apples-to-apples.

Conventions:
  bundle.X_*  : (T, N, F) feature panels (already normalized)
  bundle.y_*  : (T, N) next-day log returns
  lookback L  : we predict y[t] using features[t-L+1 .. t] (DL convention).
                For consistency, the "valid" prediction indices on test are
                t = L-1 .. T_test-1, but actually since DL predicts y[t+L-1]
                from window starting at t, the equivalent set of test
                predictions is len(X_test) - L of them.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

logger = logging.getLogger(__name__)


# =============================================================================
# Helpers
# =============================================================================
def _flatten_window(X_panel: np.ndarray, lookback: int) -> np.ndarray:
    """
    Convert (T, N, F) -> (T-L, N, L*F) by flattening each window into a flat
    feature vector. The output[t] corresponds to the DL prediction for
    time index t+L (using window t..t+L-1), so it aligns with what
    MultiAssetDataset emits (with target y[t+L-1] i.e. last index of window).

    Wait, careful: DL's __getitem__ returns y[idx + L - 1] for window starting
    at idx, with idx in [0, T-L). So target indices are [L-1, T-1], and y[T-1]
    is NaN. We use idx in [0, T-L), which gives y[L-1 .. T-2], total T-L preds.

    We mirror that here: output[i] uses window X[i:i+L] and corresponds to
    target y[i+L-1] for i in [0, T-L).
    """
    T, N, F = X_panel.shape
    n_samples = T - lookback
    if n_samples <= 0:
        raise ValueError(f"T={T} <= L={lookback}")
    out = np.zeros((n_samples, N, lookback * F), dtype=np.float32)
    for i in range(n_samples):
        out[i] = X_panel[i : i + lookback].transpose(1, 0, 2).reshape(N, lookback * F)
    return out


def _flatten_targets(y_panel: np.ndarray, lookback: int) -> np.ndarray:
    """y[L-1 : T-1] -> (T-L, N), aligned with _flatten_window output."""
    return y_panel[lookback - 1 : -1]   # length T-L


# =============================================================================
# Ridge -- one global model trained on the panel
# =============================================================================
class RidgePanel:
    """
    Train one Ridge regression on the panel of (sample, asset) rows, with the
    asset identity injected via one-hot columns. This gives a single model
    that can predict for any asset.

    Note: with a 40-day lookback and 16 features, the design matrix has
    L*F = 640 columns, plus N one-hot columns. For numerical stability we
    clip any feature outliers to +/-10 standard deviations (the input is
    already z-scored, so this is mild) and use a relatively strong alpha.
    """
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.model: Ridge | None = None
        self.n_assets: int | None = None

    def _design_matrix(self, X_flat: np.ndarray) -> np.ndarray:
        """
        X_flat: (n_samples, N, L*F)
        Returns: (n_samples * N, L*F + N)  with one-hot asset id appended.
        """
        n, N, D = X_flat.shape
        X_long = X_flat.reshape(n * N, D).astype(np.float64)   # float64 for stability
        # Replace any inf/nan defensively, clip extreme outliers
        X_long = np.nan_to_num(X_long, nan=0.0, posinf=0.0, neginf=0.0)
        X_long = np.clip(X_long, -10.0, 10.0)
        asset_oh = np.tile(np.eye(N, dtype=np.float64), (n, 1))
        return np.concatenate([X_long, asset_oh], axis=1)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, lookback: int) -> None:
        Xw = _flatten_window(X_train, lookback)
        yw = _flatten_targets(y_train, lookback)
        self.n_assets = Xw.shape[1]
        Xd = self._design_matrix(Xw)
        yd = yw.reshape(-1).astype(np.float64)
        # solver='lsqr' is more robust on ill-conditioned matrices than the
        # default Cholesky-based solver, and tolerates the panel's scale spread
        self.model = Ridge(alpha=self.alpha, solver='lsqr')
        self.model.fit(Xd, yd)
        logger.info(f"  Ridge fit: design matrix {Xd.shape}, "
                    f"R^2 train={self.model.score(Xd, yd):.4f}")

    def predict(self, X_test: np.ndarray, lookback: int) -> np.ndarray:
        Xw = _flatten_window(X_test, lookback)
        n, N, D = Xw.shape
        Xd = self._design_matrix(Xw)
        preds = self.model.predict(Xd)
        return preds.reshape(n, N).astype(np.float32)


# =============================================================================
# =============================================================================
# Gradient-boosted trees (sklearn HistGradientBoosting) -- panel model
#
# Why sklearn's HistGradientBoosting and not XGBoost or LightGBM?
# Both XGBoost and LightGBM ship native compiled C++ libraries that have
# repeatedly shown stability issues on macOS Apple Silicon. We instead use
# scikit-learn's HistGradientBoostingRegressor, which:
#   - Implements the same histogram-based gradient boosting algorithm pioneered
#     by LightGBM
#   - Is pure Python+Cython, ships with sklearn, zero compatibility risk
#   - Performs competitively on tabular data benchmarks
#   - Accepts the same conceptual hyperparameters (n_estimators, max_depth,
#     learning_rate, etc.)
#
# We keep the class name `XGBoostPanel` so the rest of the project doesn't
# need to change. From the comparison/leaderboard perspective this fills the
# same "strong gradient-boosted tabular baseline" slot.
# =============================================================================
class XGBoostPanel:
    """
    Single global gradient-boosted regressor (sklearn HistGradientBoosting)
    over (sample, asset) rows, with the asset index as a numeric feature.
    """
    def __init__(
        self,
        n_estimators: int = 400,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        subsample: float = 0.8,        # accepted but not used (sklearn API)
        colsample_bytree: float = 0.8, # accepted but not used (sklearn API)
    ):
        from sklearn.ensemble import HistGradientBoostingRegressor
        # max_iter in sklearn's API == n_estimators in xgboost/lightgbm
        # max_leaf_nodes is the tree-shape control (analogous to num_leaves)
        self.model = HistGradientBoostingRegressor(
            max_iter=n_estimators,
            max_depth=max_depth,
            max_leaf_nodes=max(2, 2 ** max_depth - 1),
            learning_rate=learning_rate,
            l2_regularization=0.1,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=42,
        )
        self.n_assets: int | None = None

    def _design(self, X_flat: np.ndarray) -> np.ndarray:
        # Append asset id (integer column) instead of one-hot for tree models
        n, N, D = X_flat.shape
        X_long = X_flat.reshape(n * N, D).astype(np.float32)
        # Defensive: clip outliers and replace inf/nan
        X_long = np.nan_to_num(X_long, nan=0.0, posinf=0.0, neginf=0.0)
        X_long = np.clip(X_long, -10.0, 10.0)
        asset_id = np.tile(np.arange(N, dtype=np.float32), n).reshape(-1, 1)
        return np.concatenate([X_long, asset_id], axis=1)

    def fit(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val:   np.ndarray, y_val:   np.ndarray,
        lookback: int,
    ) -> None:
        # We fold val into the training data; sklearn's HistGradientBoosting
        # uses its OWN internal validation_fraction split for early stopping,
        # so we don't need to pass val explicitly.
        Xw_tr = _flatten_window(X_train, lookback)
        yw_tr = _flatten_targets(y_train, lookback)

        self.n_assets = Xw_tr.shape[1]
        Xd_tr = self._design(Xw_tr); yd_tr = yw_tr.reshape(-1)

        self.model.fit(Xd_tr, yd_tr)
        best_iter = self.model.n_iter_
        logger.info(f"  GBM fit (sklearn HistGradientBoosting): "
                    f"n_iter={best_iter}, design {Xd_tr.shape}")

    def predict(self, X_test: np.ndarray, lookback: int) -> np.ndarray:
        Xw = _flatten_window(X_test, lookback)
        n, N, _ = Xw.shape
        Xd = self._design(Xw)
        preds = self.model.predict(Xd)
        return preds.reshape(n, N).astype(np.float32)


# =============================================================================
# ARIMA -- one tiny model per asset, predicts on log returns directly
# =============================================================================
class ARIMAPerAsset:
    """
    Univariate ARIMA per asset. We fit on the training-set log_ret series,
    then use a rolling one-step-ahead forecast over the test period.

    We use a simple ARMA(p, q) order; the focus of this project is the
    Transformer, so ARIMA exists primarily as a sanity-check baseline.
    """
    def __init__(self, order: tuple[int, int, int] = (1, 0, 1)):
        self.order = tuple(order)
        self.results = []   # list of fitted results, one per asset

    def fit(self, log_ret_train: np.ndarray) -> None:
        from statsmodels.tsa.arima.model import ARIMA
        T, N = log_ret_train.shape
        self.results = []
        for j in range(N):
            try:
                model = ARIMA(log_ret_train[:, j], order=self.order)
                fit = model.fit()
                self.results.append(fit)
            except Exception as e:
                logger.warning(f"    ARIMA fit failed for asset {j}: {e}; "
                               f"falling back to mean")
                self.results.append(None)
        logger.info(f"  ARIMA: fit {N} per-asset models with order {self.order}")

    def predict_rolling(
        self,
        log_ret_train: np.ndarray,
        log_ret_test:  np.ndarray,
        lookback: int,
    ) -> np.ndarray:
        """
        For each test day t (in [lookback-1, T_test-2]), produce the one-step-
        ahead forecast given the FULL history (train + test up to t).
        Returns: predictions aligned with DL test predictions, shape (T_test-L, N).
        """
        from statsmodels.tsa.arima.model import ARIMA
        T_te, N = log_ret_test.shape
        n_pred = T_te - lookback
        preds = np.zeros((n_pred, N), dtype=np.float32)

        # For computational sanity, we re-fit using the train residuals and
        # then iterate `forecast(steps=1)` while appending observations.
        # statsmodels supports `append` for ARIMA results.
        for j in range(N):
            res = self.results[j]
            if res is None:
                preds[:, j] = log_ret_train[:, j].mean()
                continue
            try:
                # Bring the model up to "test t = lookback-1"
                history_extra = log_ret_test[:lookback - 1, j]
                if len(history_extra) > 0:
                    res = res.append(history_extra, refit=False)
                for k in range(n_pred):
                    fc = res.forecast(steps=1)
                    preds[k, j] = float(fc[0]) if hasattr(fc, '__len__') else float(fc)
                    # Append the actual value for next-step conditioning
                    res = res.append([log_ret_test[lookback - 1 + k, j]], refit=False)
            except Exception as e:
                logger.warning(f"    ARIMA rolling forecast failed for asset {j}: "
                               f"{e}; using train mean")
                preds[:, j] = log_ret_train[:, j].mean()
        return preds
