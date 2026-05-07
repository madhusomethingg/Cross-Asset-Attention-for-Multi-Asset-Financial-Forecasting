"""
Stage 3e: Threshold calibration.

Our regression models output continuous predicted returns. To convert these
to UP/DOWN classifications, we currently use threshold = 0 (UP if pred >= 0).
But predictions can be systematically biased -- e.g., a model trained on
bullish data tends to predict positive returns even on neutral days.

This script finds the threshold that maximizes Macro F1 on the *validation*
set (NOT test -- that would be data leakage), then applies that threshold
to test predictions. This is a standard, well-documented post-hoc
calibration technique.

Reference: see e.g. He & Garcia (2009), "Learning from Imbalanced Data";
this is essentially Platt scaling without the logistic regression layer.

Outputs:
    artifacts/classification_report_calibrated_test.csv
    artifacts/threshold_calibration.csv     (chosen threshold per model)
    artifacts/classification_report_transformer_calibrated.txt

Usage:
    python scripts/07_threshold_calibration.py
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


# -----------------------------------------------------------------------------
def to_classes_with_threshold(arr: np.ndarray, threshold: float) -> np.ndarray:
    """Map continuous returns to {0: DOWN, 1: UP} using the given threshold."""
    return (arr >= threshold).astype(int).reshape(-1)


def find_best_threshold(
    val_preds: np.ndarray,
    val_actuals: np.ndarray,
    metric: str = 'f1_macro',
    n_grid: int = 101,
) -> tuple[float, float]:
    """
    Search a grid of candidate thresholds; pick the one that maximises the
    chosen metric on validation. Returns (best_threshold, best_metric_value).

    Grid spans from the 5th to the 95th percentile of validation predictions
    so we don't waste effort on extreme thresholds that would label
    everything as a single class.
    """
    flat_preds = val_preds.reshape(-1)
    lo = float(np.percentile(flat_preds, 5))
    hi = float(np.percentile(flat_preds, 95))
    candidates = np.linspace(lo, hi, n_grid)
    y_true = (val_actuals.reshape(-1) >= 0).astype(int)

    best_metric = -np.inf
    best_thr    = 0.0
    for thr in candidates:
        y_pred = (flat_preds >= thr).astype(int)
        if metric == 'f1_macro':
            score = f1_score(y_true, y_pred, average='macro', zero_division=0)
        elif metric == 'accuracy':
            score = accuracy_score(y_true, y_pred)
        elif metric == 'balanced_accuracy':
            from sklearn.metrics import balanced_accuracy_score
            score = balanced_accuracy_score(y_true, y_pred)
        else:
            raise ValueError(f"unknown metric {metric}")
        if score > best_metric:
            best_metric = score
            best_thr = float(thr)
    return best_thr, float(best_metric)


def evaluate_with_threshold(
    test_preds: np.ndarray, test_actuals: np.ndarray, threshold: float,
) -> dict:
    y_pred = to_classes_with_threshold(test_preds, threshold)
    y_true = to_classes_with_threshold(test_actuals, 0.0)   # actual UP/DOWN
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        'threshold':      threshold,
        'accuracy':       accuracy_score (y_true, y_pred),
        'precision_UP':   precision_score(y_true, y_pred, zero_division=0),
        'recall_UP':      recall_score   (y_true, y_pred, zero_division=0),
        'f1_UP':          f1_score       (y_true, y_pred, zero_division=0),
        'precision_DOWN': precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        'recall_DOWN':    recall_score   (y_true, y_pred, pos_label=0, zero_division=0),
        'f1_DOWN':        f1_score       (y_true, y_pred, pos_label=0, zero_division=0),
        'f1_macro':       f1_score       (y_true, y_pred, average='macro', zero_division=0),
        'TP': int(tp), 'FP': int(fp), 'TN': int(tn), 'FN': int(fn),
    }


# -----------------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S',
    )
    log = logging.getLogger('threshold')

    pred_dir = ROOT / 'artifacts' / 'predictions'
    actuals_npz = np.load(pred_dir / '_actuals.npz', allow_pickle=False)
    test_actuals = actuals_npz['test_actuals']
    val_actuals  = actuals_npz['val_actuals']

    log.info(f"Validation: {val_actuals.shape}, Test: {test_actuals.shape}")

    rows = []
    threshold_log = []
    for name in MODEL_NAMES:
        npz_path = pred_dir / f'{name}.npz'
        if not npz_path.exists():
            continue
        d = np.load(npz_path, allow_pickle=False)

        # 1. Find optimal threshold on validation (NEVER on test)
        thr, val_f1 = find_best_threshold(
            d['val_preds'], val_actuals, metric='f1_macro',
        )

        # 2. Evaluate test set with that threshold
        m = evaluate_with_threshold(d['test_preds'], test_actuals, thr)
        m['model'] = name
        m['val_f1_macro'] = val_f1
        rows.append(m)

        # 3. Bookkeeping
        threshold_log.append({
            'model': name,
            'chosen_threshold': thr,
            'val_f1_macro_at_threshold': val_f1,
            'val_pred_mean':   float(d['val_preds'].mean()),
            'val_pred_std':    float(d['val_preds'].std()),
        })

    df = pd.DataFrame(rows).set_index('model')[
        ['threshold', 'accuracy',
         'precision_UP', 'recall_UP', 'f1_UP',
         'precision_DOWN', 'recall_DOWN', 'f1_DOWN',
         'f1_macro', 'val_f1_macro',
         'TP', 'FP', 'TN', 'FN']
    ]
    log.info("\nClassification report after threshold calibration "
             "(thresholds chosen on VAL, applied to TEST):\n"
             + df.round(4).to_string())

    df.to_csv(ROOT / 'artifacts' / 'classification_report_calibrated_test.csv')

    pd.DataFrame(threshold_log).set_index('model').to_csv(
        ROOT / 'artifacts' / 'threshold_calibration.csv'
    )

    # Detailed sklearn report for the Transformer specifically
    if 'transformer' in df.index:
        thr = df.loc['transformer', 'threshold']
        d = np.load(pred_dir / 'transformer.npz', allow_pickle=False)
        y_true = (test_actuals.reshape(-1) >= 0).astype(int)
        y_pred = (d['test_preds'].reshape(-1) >= thr).astype(int)
        log.info(f"\nTransformer | calibrated threshold = {thr:.6f}\n"
                 + classification_report(y_true, y_pred,
                                         target_names=['DOWN', 'UP'],
                                         digits=4))
        with open(ROOT / 'artifacts' /
                  'classification_report_transformer_calibrated.txt', 'w') as f:
            f.write(f"Transformer -- Calibrated Test Set Classification Report\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Calibrated threshold: {thr:.6f}\n")
            f.write(f"(threshold chosen on validation set to maximise macro F1)\n\n")
            f.write(classification_report(
                y_true, y_pred, target_names=['DOWN', 'UP'], digits=4))
            f.write("\n\nConfusion matrix (rows=actual, cols=predicted):\n")
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            f.write(f"             pred_DOWN  pred_UP\n")
            f.write(f"actual_DOWN  {cm[0,0]:9d}  {cm[0,1]:7d}\n")
            f.write(f"actual_UP    {cm[1,0]:9d}  {cm[1,1]:7d}\n")
        log.info("Saved calibrated artifacts.")


if __name__ == '__main__':
    main()
