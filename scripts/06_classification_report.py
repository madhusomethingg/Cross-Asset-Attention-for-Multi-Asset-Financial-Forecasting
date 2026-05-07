"""
Stage 3d: Classification report.

Our models are trained as regressors that predict the magnitude of next-day
log returns. However, in a trading context what matters most is the SIGN of
the prediction (up vs down). This script re-evaluates each model's
predictions as a binary classification problem:

    predicted class = 'UP'   if pred  >= 0  else 'DOWN'
    actual class    = 'UP'   if actual >= 0 else 'DOWN'

We report per-model:
    accuracy, precision, recall, F1 (binary, "UP" as positive class)
    macro F1     (averages "UP" and "DOWN" F1)
    confusion matrix (TN, FP, FN, TP)

This is purely a re-interpretation of the same predictions; no retraining.

Usage:
    python scripts/06_classification_report.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


MODEL_NAMES = ['ridge', 'arima', 'xgboost', 'lstm', 'gru', 'tcn', 'transformer']


def to_classes(arr: np.ndarray) -> np.ndarray:
    """Map continuous returns to {0: DOWN, 1: UP}; ties (== 0) treated as DOWN."""
    return (arr >= 0).astype(int).reshape(-1)


def per_model_metrics(preds: np.ndarray, actuals: np.ndarray) -> dict:
    y_pred = to_classes(preds)
    y_true = to_classes(actuals)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        'accuracy':       accuracy_score (y_true, y_pred),
        'precision_UP':   precision_score(y_true, y_pred, zero_division=0),
        'recall_UP':      recall_score   (y_true, y_pred, zero_division=0),
        'f1_UP':          f1_score       (y_true, y_pred, zero_division=0),
        'f1_macro':       f1_score       (y_true, y_pred,
                                          average='macro', zero_division=0),
        'TP': int(tp), 'FP': int(fp), 'TN': int(tn), 'FN': int(fn),
        'n_actual_UP':    int((y_true == 1).sum()),
        'n_actual_DOWN':  int((y_true == 0).sum()),
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S',
    )
    log = logging.getLogger('classification')

    pred_dir = ROOT / 'artifacts' / 'predictions'
    actuals_npz = np.load(pred_dir / '_actuals.npz', allow_pickle=False)
    test_actuals = actuals_npz['test_actuals']
    val_actuals  = actuals_npz['val_actuals']

    n_total_test = test_actuals.size
    n_up_test    = int((test_actuals >= 0).sum())
    log.info(f"Test set: {test_actuals.shape} = {n_total_test} (asset, day) pairs")
    log.info(f"Class balance: UP={n_up_test} ({n_up_test/n_total_test:.1%}), "
             f"DOWN={n_total_test - n_up_test} "
             f"({(n_total_test - n_up_test)/n_total_test:.1%})")

    # Build leaderboard (test only -- the meaningful one)
    rows = []
    for name in MODEL_NAMES:
        npz_path = pred_dir / f'{name}.npz'
        if not npz_path.exists():
            log.warning(f"Skipping missing predictions: {name}")
            continue
        d = np.load(npz_path, allow_pickle=False)
        m = per_model_metrics(d['test_preds'], test_actuals)
        m['model'] = name
        rows.append(m)

    df = pd.DataFrame(rows).set_index('model')[
        ['accuracy', 'precision_UP', 'recall_UP', 'f1_UP', 'f1_macro',
         'TP', 'FP', 'TN', 'FN']
    ]
    log.info("\nClassification report (test set, UP/DOWN direction):\n"
             + df.round(4).to_string())

    # Save
    out_path = ROOT / 'artifacts' / 'classification_report_test.csv'
    df.to_csv(out_path)
    log.info(f"Saved to {out_path}")

    # Also generate a sklearn-format report for the Transformer specifically
    # (the headline model for the slides)
    if 'transformer' in df.index:
        transformer_npz = np.load(pred_dir / 'transformer.npz', allow_pickle=False)
        y_pred = to_classes(transformer_npz['test_preds'])
        y_true = to_classes(test_actuals)
        log.info("\nSklearn classification_report for Transformer (test set):\n"
                 + classification_report(
                     y_true, y_pred,
                     target_names=['DOWN', 'UP'], digits=4))

        # Save the textual report too -- nice for putting in the slides
        with open(ROOT / 'artifacts' / 'classification_report_transformer.txt',
                  'w') as f:
            f.write("Transformer -- Test Set Classification Report\n")
            f.write("=" * 60 + "\n\n")
            f.write(classification_report(
                y_true, y_pred, target_names=['DOWN', 'UP'], digits=4))
            f.write("\n\nConfusion matrix (rows=actual, cols=predicted):\n")
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            f.write(f"             pred_DOWN  pred_UP\n")
            f.write(f"actual_DOWN  {cm[0,0]:9d}  {cm[0,1]:7d}\n")
            f.write(f"actual_UP    {cm[1,0]:9d}  {cm[1,1]:7d}\n")


if __name__ == '__main__':
    main()
