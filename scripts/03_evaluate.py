"""
Stage 3a: Evaluate all 7 saved models on validation and test predictions.

Inputs:
    artifacts/data/        (DataBundle from stage 1)
    artifacts/predictions/ (per-model val/test preds from stage 2)

Outputs:
    artifacts/leaderboard_val.csv
    artifacts/leaderboard_test.csv
    artifacts/dm_test_pvalues.csv         (Diebold-Mariano p-values vs Transformer)
    artifacts/predictions/_actuals.npz    (val and test actuals, aligned)

Usage:
    python scripts/03_evaluate.py
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

from src.data import load_bundle
from src.evaluate import build_leaderboard, diebold_mariano


MODEL_NAMES = ['ridge', 'arima', 'xgboost', 'lstm', 'gru', 'tcn', 'transformer']


def aligned_actuals(bundle, lookback: int):
    """
    DL prediction at index i corresponds to target y[i+L-1] in its split.
    For a split with T rows, we make T-L predictions covering target indices
    [L-1 ... T-2]. So the aligned actuals are bundle.y_split[L-1:-1].
    """
    val_actuals  = bundle.y_val[lookback - 1: -1]
    test_actuals = bundle.y_test[lookback - 1: -1]
    return val_actuals, test_actuals


def load_predictions(pred_dir: Path) -> dict[str, dict]:
    out = {}
    for name in MODEL_NAMES:
        path = pred_dir / f'{name}.npz'
        if not path.exists():
            logging.warning(f"Missing predictions: {path}; skipping")
            continue
        d = np.load(path, allow_pickle=False)
        out[name] = {
            'val_preds':  d['val_preds'],
            'test_preds': d['test_preds'],
            'val_dates':  d['val_dates'],
            'test_dates': d['test_dates'],
        }
    return out


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S',
    )
    log = logging.getLogger('evaluate')

    cfg = yaml.safe_load(open(ROOT / 'configs' / 'config.yaml'))
    L = cfg['data']['lookback']

    bundle = load_bundle(ROOT / 'artifacts' / 'data')
    val_actuals, test_actuals = aligned_actuals(bundle, L)
    log.info(f"Aligned actuals: val {val_actuals.shape}, test {test_actuals.shape}")

    pred_dir = ROOT / 'artifacts' / 'predictions'
    preds = load_predictions(pred_dir)
    log.info(f"Loaded predictions for: {list(preds.keys())}")

    # Save aligned actuals + dates for downstream scripts
    np.savez_compressed(
        pred_dir / '_actuals.npz',
        val_actuals=val_actuals.astype(np.float32),
        test_actuals=test_actuals.astype(np.float32),
        val_dates=np.array([d.isoformat() for d in
                            bundle.dates_val[L - 1: -1]]),
        test_dates=np.array([d.isoformat() for d in
                             bundle.dates_test[L - 1: -1]]),
        tickers=np.array(bundle.tickers),
    )

    # ---- Leaderboards
    val_preds_dict  = {n: p['val_preds']  for n, p in preds.items()}
    test_preds_dict = {n: p['test_preds'] for n, p in preds.items()}

    # Sanity: every prediction array must match actuals shape
    for name, p in val_preds_dict.items():
        assert p.shape == val_actuals.shape, \
            f"{name} val preds shape {p.shape} != actuals {val_actuals.shape}"
    for name, p in test_preds_dict.items():
        assert p.shape == test_actuals.shape, \
            f"{name} test preds shape {p.shape} != actuals {test_actuals.shape}"

    lb_val  = build_leaderboard(val_preds_dict,  val_actuals)
    lb_test = build_leaderboard(test_preds_dict, test_actuals)

    log.info("\nValidation leaderboard:\n" + lb_val.round(5).to_string())
    log.info("\nTest leaderboard:\n"        + lb_test.round(5).to_string())

    out_dir = ROOT / 'artifacts'
    lb_val.to_csv (out_dir / 'leaderboard_val.csv')
    lb_test.to_csv(out_dir / 'leaderboard_test.csv')

    # ---- Diebold-Mariano pairwise vs Transformer
    if 'transformer' in test_preds_dict:
        rows = []
        ref = test_preds_dict['transformer']
        for name, preds_other in test_preds_dict.items():
            if name == 'transformer':
                continue
            stat, p = diebold_mariano(ref, preds_other, test_actuals, h=1)
            rows.append({
                'baseline':  name,
                'DM_stat':   stat,
                'p_value':   p,
                'verdict':   'transformer better'
                             if (stat < 0 and p < 0.05) else
                             ('baseline better' if (stat > 0 and p < 0.05)
                              else 'tie (p >= 0.05)'),
            })
        dm_df = pd.DataFrame(rows).set_index('baseline')
        log.info("\nDiebold-Mariano (test, vs transformer):\n"
                 + dm_df.round(4).to_string())
        dm_df.to_csv(out_dir / 'dm_test_pvalues.csv')
    log.info(f"Saved leaderboards and DM tests to {out_dir}")


if __name__ == '__main__':
    main()
