"""
Stage 1: Build the data bundle.

Run from the project root:
    python scripts/01_build_data.py

This downloads OHLCV data from yfinance, computes features, aligns assets,
splits chronologically (70/15/15), normalizes using train-only statistics,
and saves everything to artifacts/data/.

Expected runtime: ~1 minute (mostly yfinance download).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import yaml

# Make `src` importable when running this script from the project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import build_data_bundle, save_bundle


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S',
    )
    log = logging.getLogger('build_data')

    cfg_path = ROOT / 'configs' / 'config.yaml'
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg['data']
    log.info(f"Building data bundle for {len(data_cfg['tickers'])} tickers")

    bundle = build_data_bundle(
        tickers=data_cfg['tickers'],
        start_date=data_cfg['start_date'],
        end_date=data_cfg['end_date'],
        train_frac=data_cfg['train_frac'],
        val_frac=data_cfg['val_frac'],
    )

    print()
    print(bundle.summary())

    out_dir = ROOT / 'artifacts' / 'data'
    save_bundle(bundle, out_dir)
    log.info(f"Done. Saved to {out_dir}")


if __name__ == '__main__':
    main()
