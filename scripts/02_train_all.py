"""
Stage 2: Train all 7 models and save their predictions.

Runs (in this order):
  1. Ridge       (panel)        -- traditional
  2. ARIMA       (per-asset)    -- traditional
  3. XGBoost     (panel)        -- traditional
  4. LSTM                       -- DL baseline
  5. GRU                        -- DL baseline
  6. TCN                        -- DL baseline
  7. Transformer (cross-asset)  -- main model

Saves to artifacts/predictions/ a single .npz file per model with:
  - val_preds   (n_val_samples, N)
  - test_preds  (n_test_samples, N)
  - val_dates   (aligned to predictions)
  - test_dates  (aligned to predictions)

Saves trained DL model state dicts to artifacts/models/.

Usage:
    python scripts/02_train_all.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Mac MPS: enable CPU fallback for any unsupported op
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

from src.data import load_bundle
from src.dataset import make_loaders
from src.train import get_device, train_model
from src.losses import make_loss
from src.models.transformer import CrossAssetTransformer
from src.models.lstm import LSTMForecaster
from src.models.gru import GRUForecaster
from src.models.tcn import TCNForecaster
from src.models.classical import RidgePanel, XGBoostPanel, ARIMAPerAsset


# -----------------------------------------------------------------------------
def aligned_pred_dates(dates, lookback: int):
    """
    DL predictions correspond to indices [L-1 .. T-2] of a split (since the
    last index has no next-day target). Same for classical helpers.
    Returns the list of dates the predictions are aligned to.
    """
    return list(dates[lookback - 1: -1])


# -----------------------------------------------------------------------------
def save_predictions(out_dir: Path, name: str,
                     val_preds, test_preds,
                     val_dates, test_dates,
                     extras: dict | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        'val_preds':  val_preds.astype(np.float32),
        'test_preds': test_preds.astype(np.float32),
        'val_dates':  np.array([d.isoformat() for d in val_dates]),
        'test_dates': np.array([d.isoformat() for d in test_dates]),
    }
    if extras:
        payload.update(extras)
    np.savez_compressed(out_dir / f'{name}.npz', **payload)


# -----------------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S',
    )
    log = logging.getLogger('train_all')

    cfg_path = ROOT / 'configs' / 'config.yaml'
    cfg = yaml.safe_load(open(cfg_path))

    # ---- Reproducibility
    SEED = cfg['training']['seed']
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # ---- Load bundle
    bundle = load_bundle(ROOT / 'artifacts' / 'data')
    log.info(bundle.summary())

    L = cfg['data']['lookback']
    N = bundle.n_assets
    F = bundle.n_features

    # Loaders for DL models
    train_dl, val_dl, test_dl = make_loaders(
        bundle, lookback=L,
        batch_size=cfg['training']['batch_size'],
    )
    log.info(f"DL loaders: train batches={len(train_dl)}, "
             f"val batches={len(val_dl)}, test batches={len(test_dl)}")

    # Aligned date indices for predictions
    val_dates_pred  = aligned_pred_dates(bundle.dates_val,  L)
    test_dates_pred = aligned_pred_dates(bundle.dates_test, L)

    pred_dir   = ROOT / 'artifacts' / 'predictions'
    model_dir  = ROOT / 'artifacts' / 'models'
    pred_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # 1. RIDGE
    # =========================================================================
    log.info("=" * 60); log.info(">> Ridge Regression (panel)"); log.info("=" * 60)
    t0 = time.time()
    ridge = RidgePanel(alpha=cfg['models']['ridge']['alpha'])
    ridge.fit(bundle.X_train, bundle.y_train, lookback=L)
    val_preds  = ridge.predict(bundle.X_val,  L)
    test_preds = ridge.predict(bundle.X_test, L)
    log.info(f"Ridge done in {time.time()-t0:.1f}s")
    save_predictions(pred_dir, 'ridge', val_preds, test_preds,
                     val_dates_pred, test_dates_pred)

    # =========================================================================
    # 2. ARIMA
    # =========================================================================
    log.info("=" * 60); log.info(">> ARIMA (per asset, rolling)"); log.info("=" * 60)
    t0 = time.time()
    # ARIMA fits on the log_ret column directly, which is feature index 0
    log_ret_idx = bundle.feature_cols.index('log_ret')
    log_ret_train = bundle.X_train[:, :, log_ret_idx]   # (T_tr, N)
    log_ret_val   = bundle.X_val[:,   :, log_ret_idx]
    log_ret_test  = bundle.X_test[:,  :, log_ret_idx]

    arima = ARIMAPerAsset(order=tuple(cfg['models']['arima']['order']))
    arima.fit(log_ret_train)
    val_preds  = arima.predict_rolling(log_ret_train, log_ret_val,  lookback=L)
    test_preds = arima.predict_rolling(log_ret_train, log_ret_test, lookback=L)
    log.info(f"ARIMA done in {time.time()-t0:.1f}s")
    save_predictions(pred_dir, 'arima', val_preds, test_preds,
                     val_dates_pred, test_dates_pred)

    # =========================================================================
    # 3. XGBOOST
    # =========================================================================
    log.info("=" * 60); log.info(">> XGBoost (panel)"); log.info("=" * 60)
    t0 = time.time()
    xgb = XGBoostPanel(**cfg['models']['xgboost'])
    xgb.fit(bundle.X_train, bundle.y_train,
            bundle.X_val,   bundle.y_val, lookback=L)
    val_preds  = xgb.predict(bundle.X_val,  L)
    test_preds = xgb.predict(bundle.X_test, L)
    log.info(f"XGBoost done in {time.time()-t0:.1f}s")
    save_predictions(pred_dir, 'xgboost', val_preds, test_preds,
                     val_dates_pred, test_dates_pred)

    # =========================================================================
    # 4-7. DEEP LEARNING MODELS (uniform loop)
    # =========================================================================
    device = get_device()
    log.info(f"DL device: {device}")

    dl_specs = [
        ('lstm', LSTMForecaster, dict(
            n_features=F, n_assets=N, **cfg['models']['lstm'])),
        ('gru', GRUForecaster, dict(
            n_features=F, n_assets=N, **cfg['models']['gru'])),
        ('tcn', TCNForecaster, dict(
            n_features=F, n_assets=N, **cfg['models']['tcn'])),
        ('transformer', CrossAssetTransformer, dict(
            n_features=F, n_assets=N, **cfg['models']['transformer'])),
    ]

    train_cfg = cfg['training']
    # Loss config: pull from YAML if present, else default to combined loss
    loss_cfg = cfg.get('loss', {
        'name': 'combined',
        'lambda_dir': 5.0,
        'down_weight': 2.0,
    })
    log.info(f"DL loss: {loss_cfg}")

    for name, cls, kwargs in dl_specs:
        log.info("=" * 60); log.info(f">> {name.upper()}"); log.info("=" * 60)
        torch.manual_seed(SEED)
        model = cls(**kwargs)

        loss_fn = make_loss(
            loss_cfg['name'],
            **{k: v for k, v in loss_cfg.items() if k != 'name'},
        )

        result = train_model(
            model, train_dl, val_dl, test_dl,
            max_epochs=train_cfg['max_epochs'],
            lr=train_cfg['lr'],
            weight_decay=train_cfg['weight_decay'],
            grad_clip=train_cfg['grad_clip'],
            early_stopping_patience=train_cfg['early_stopping_patience'],
            device=device, model_name=name,
            loss_fn=loss_fn,
        )
        log.info(f"[{name}] done. epochs={result.epochs_trained} "
                 f"best_val={result.best_val_loss:.6f}  "
                 f"time={result.seconds:.1f}s")

        save_predictions(
            pred_dir, name,
            result.val_preds, result.test_preds,
            val_dates_pred, test_dates_pred,
            extras={
                'train_losses': np.array(result.train_losses),
                'val_losses':   np.array(result.val_losses),
            },
        )
        torch.save(result.model.state_dict(), model_dir / f'{name}.pt')

    log.info("All models trained.")
    log.info(f"Predictions: {pred_dir}")
    log.info(f"Models     : {model_dir}")


if __name__ == '__main__':
    main()
