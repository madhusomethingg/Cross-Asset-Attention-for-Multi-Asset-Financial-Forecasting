"""
Stage 3b: Run a long/short top-k backtest on each model's test-set
predictions, plus an equal-weight long-only benchmark.

Inputs:
    artifacts/predictions/*.npz
    artifacts/predictions/_actuals.npz

Outputs:
    artifacts/backtest_summary.csv
    artifacts/backtest/{model}.npz  (daily returns + equity curve + weights)

Usage:
    python scripts/04_backtest.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.backtest import backtest_all, backtest_summary_df


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S',
    )
    log = logging.getLogger('backtest')

    cfg = yaml.safe_load(open(ROOT / 'configs' / 'config.yaml'))
    bt_cfg = cfg['backtest']

    pred_dir = ROOT / 'artifacts' / 'predictions'
    actuals_npz = np.load(pred_dir / '_actuals.npz', allow_pickle=False)
    test_actuals = actuals_npz['test_actuals']
    test_dates   = actuals_npz['test_dates']
    log.info(f"Test actuals: {test_actuals.shape}, dates {len(test_dates)}")

    # Load test predictions for all models
    model_preds: dict[str, np.ndarray] = {}
    for npz_path in sorted(pred_dir.glob('*.npz')):
        if npz_path.name.startswith('_'):
            continue
        name = npz_path.stem
        d = np.load(npz_path, allow_pickle=False)
        model_preds[name] = d['test_preds']
    log.info(f"Loaded predictions for: {list(model_preds.keys())}")

    # Run backtest for each model + equal-weight benchmark
    results = backtest_all(
        model_preds, test_actuals,
        k=bt_cfg['top_k'],
        cost_bps=bt_cfg['cost_bps'],
        ann_factor=bt_cfg['ann_factor'],
    )

    summary = backtest_summary_df(results)
    log.info("\nBacktest summary (test period):\n"
             + summary.round(4).to_string())
    summary.to_csv(ROOT / 'artifacts' / 'backtest_summary.csv')

    # Save per-strategy time series for the Streamlit app
    bt_dir = ROOT / 'artifacts' / 'backtest'
    bt_dir.mkdir(parents=True, exist_ok=True)
    for name, res in results.items():
        np.savez_compressed(
            bt_dir / f'{name}.npz',
            daily_returns=res.daily_returns,
            equity_curve =res.equity_curve,
            weights      =res.weights,
            turnover     =res.turnover,
            test_dates   =test_dates,
        )
    log.info(f"Saved per-strategy time series to {bt_dir}")


if __name__ == '__main__':
    main()
