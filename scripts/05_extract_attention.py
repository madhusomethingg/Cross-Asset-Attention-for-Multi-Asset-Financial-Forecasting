"""
Stage 3c: Extract cross-asset attention from the trained Transformer and
compute all the comparison artifacts the demo needs.

Inputs:
    artifacts/data/                 (DataBundle)
    artifacts/models/transformer.pt (trained weights from stage 2)

Outputs (under artifacts/attention/):
    test_attention.npz          (T, N, N) per-day attention (last timestep)
    test_corr_rolling.npz       (T, N, N) per-day rolling Pearson correlation
    test_attention_metrics.npz  per-day concentration, regime-shift, similarity
    top3_attended.json          per-day, per-asset top-3 attended assets
                                (for human-readable displays in the app)

Usage:
    python scripts/05_extract_attention.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

from src.data import load_bundle
from src.dataset import make_loaders
from src.train import get_device
from src.models.transformer import CrossAssetTransformer
from src.attention import (
    extract_attention,
    rolling_correlation,
    attention_concentration,
    regime_shift_score,
    correlation_shift_score,
    matrix_similarity_series,
    top_k_attended,
)


def _infer_transformer_hparams(state: dict) -> dict:
    """
    Read shapes off a saved CrossAssetTransformer state_dict to recover the
    architecture hyperparameters. This makes the loader robust to any
    accidental drift between the YAML config and the trained checkpoint.
    """
    d_model = state['input_proj.weight'].shape[0]
    d_ff    = state['temporal_blocks.0.ff.0.weight'].shape[0]
    # Count layers by counting blocks in either ModuleList
    n_layers = len({k.split('.')[1] for k in state
                    if k.startswith('temporal_blocks.')})
    return dict(d_model=d_model, d_ff=d_ff, n_layers=n_layers)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S',
    )
    log = logging.getLogger('attention')

    cfg = yaml.safe_load(open(ROOT / 'configs' / 'config.yaml'))
    L = cfg['data']['lookback']

    bundle = load_bundle(ROOT / 'artifacts' / 'data')
    N = bundle.n_assets
    F = bundle.n_features

    # ---- Rebuild data loaders (test only — that's what the demo shows)
    _, _, test_dl = make_loaders(
        bundle, lookback=L,
        batch_size=cfg['training']['batch_size'],
    )

    # ---- Rebuild Transformer architecture and load weights
    device = get_device()
    log.info(f"Device: {device}")

    state_path = ROOT / 'artifacts' / 'models' / 'transformer.pt'
    state = torch.load(state_path, map_location=device)

    # Infer model hyperparameters from the checkpoint itself so this script
    # is robust to config changes between training and attention extraction.
    inferred = _infer_transformer_hparams(state)
    log.info(f"Inferred from checkpoint: {inferred}")

    model_cfg = dict(cfg['models']['transformer'])
    model_cfg.update(inferred)        # checkpoint dims override config

    model = CrossAssetTransformer(
        n_features=F, n_assets=N,
        **model_cfg,
    ).to(device)
    model.load_state_dict(state)
    log.info(f"Loaded transformer weights from {state_path}")

    # ---- Extract per-day attention (last timestep only -> (T, N, N))
    attn = extract_attention(model, test_dl, device=device,
                             use_last_timestep_only=True)
    log.info(f"Test attention shape: {attn.shape}")

    # ---- Build rolling Pearson correlation matrix on test log returns
    # We use the log_ret feature column (from normalized X) for stability
    # Actually we want the *raw* log returns for correlation; pull them from
    # the saved actuals (which were unscaled targets = next-day log returns).
    actuals_npz = np.load(
        ROOT / 'artifacts' / 'predictions' / '_actuals.npz',
        allow_pickle=False,
    )
    test_actuals = actuals_npz['test_actuals']            # (T, N) realised next-day log rets
    test_dates   = actuals_npz['test_dates']
    # Sanity: the number of test predictions and the number of attention rows
    # should match (both come from the same DataLoader)
    assert attn.shape[0] == test_actuals.shape[0], (
        f"Attention has {attn.shape[0]} rows, actuals has {test_actuals.shape[0]}"
    )
    corr = rolling_correlation(test_actuals, window=60)   # (T, N, N)

    # ---- Quantitative attention metrics
    concentration = attention_concentration(attn)         # (T,)
    att_shift     = regime_shift_score(attn)              # (T,)
    cor_shift     = correlation_shift_score(corr)         # (T,)
    sim           = matrix_similarity_series(attn, corr)  # (T,)

    # ---- Top-3 attended per asset, per day (light JSON, for app readability)
    top3 = top_k_attended(attn, asset_names=list(bundle.tickers), k=3)
    # top3 shape: list of T entries; each entry is a list of N rows;
    # each row is a list of (name, weight) tuples of length 3.

    # ---- Save everything
    out_dir = ROOT / 'artifacts' / 'attention'
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / 'test_attention.npz',
                        attn=attn.astype(np.float32),
                        test_dates=test_dates,
                        tickers=np.array(bundle.tickers))
    np.savez_compressed(out_dir / 'test_corr_rolling.npz',
                        corr=corr.astype(np.float32),
                        test_dates=test_dates,
                        tickers=np.array(bundle.tickers))
    np.savez_compressed(out_dir / 'test_attention_metrics.npz',
                        concentration=concentration.astype(np.float32),
                        att_shift=att_shift.astype(np.float32),
                        cor_shift=cor_shift.astype(np.float32),
                        sim=sim.astype(np.float32),
                        test_dates=test_dates,
                        tickers=np.array(bundle.tickers))

    # JSON of top-3 attended (small enough to keep human-readable)
    top3_serializable = []
    for t_idx in range(len(top3)):
        day_obj = {
            'date': str(test_dates[t_idx]),
            'top3_per_asset': {
                bundle.tickers[i]: [
                    {'asset': name, 'weight': float(w)}
                    for (name, w) in top3[t_idx][i]
                ]
                for i in range(N)
            }
        }
        top3_serializable.append(day_obj)
    with open(out_dir / 'top3_attended.json', 'w') as f:
        json.dump(top3_serializable, f, indent=1)

    # ---- Quick sanity log
    log.info("Attention summary:")
    log.info(f"  concentration : mean={np.nanmean(concentration):.3f} bits "
             f"(theoretical max log2(N)={np.log2(N):.3f})")
    log.info(f"  attn shift   : mean={np.nanmean(att_shift):.4f}, "
             f"std={np.nanstd(att_shift):.4f}")
    log.info(f"  corr shift   : mean={np.nanmean(cor_shift):.4f}, "
             f"std={np.nanstd(cor_shift):.4f}")
    log.info(f"  attn-vs-corr similarity : "
             f"mean={np.nanmean(sim):.3f} (Spearman rho)")
    log.info(f"All attention artifacts saved to {out_dir}")


if __name__ == '__main__':
    main()
