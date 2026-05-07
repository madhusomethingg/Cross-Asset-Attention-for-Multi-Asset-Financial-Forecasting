"""
PyTorch Dataset + DataLoader for the multi-asset panel.

Each sample is a sliding window of `lookback` days:
  X: (L, N, F)  -- features for L days across N assets
  y: (N,)       -- next-day log return for each asset (i.e., day L+1)

The dataset accepts the (T, N, F) array produced by `src.data` and emits
tensor batches of shape (B, L, N, F) and (B, N).
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class MultiAssetDataset(Dataset):
    """
    Sliding-window dataset over a (T, N, F) feature panel and (T, N) targets.

    Sample i contains:
        X[i] = features[i : i + L]    -> (L, N, F)
        y[i] = targets[i + L - 1]      -> (N,)

    Note: targets in the underlying array are already shifted to the next-day
    return at the time of feature engineering, so y[t] is "tomorrow's return
    given features available at the close of day t". Therefore for window
    ending at index `i + L - 1`, the prediction target is targets[i + L - 1].
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, lookback: int):
        if X.ndim != 3:
            raise ValueError(f"X must be (T, N, F), got {X.shape}")
        if y.ndim != 2:
            raise ValueError(f"y must be (T, N), got {y.shape}")
        if len(X) != len(y):
            raise ValueError(f"X and y must have the same T, got {len(X)} vs {len(y)}")
        if len(X) < lookback + 1:
            raise ValueError(
                f"Need at least lookback+1={lookback+1} rows, got {len(X)}"
            )
        self.X = X
        self.y = y
        self.L = lookback

    def __len__(self) -> int:
        # Each window of length L ends at index L-1 .. T-1, giving T-L+1 windows,
        # but the last valid window is the one whose target index is the last
        # row that has a non-NaN target. Since targets are already shifted, the
        # last row's target is NaN (no "next day"), so we exclude it.
        return len(self.X) - self.L

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.X[idx : idx + self.L]).float()      # (L, N, F)
        y = torch.from_numpy(self.y[idx + self.L - 1]).float()        # (N,)
        return x, y


def make_loaders(
    bundle,
    lookback: int,
    batch_size: int,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders from a DataBundle.

    Returns: (train_loader, val_loader, test_loader)
    """
    train_ds = MultiAssetDataset(bundle.X_train, bundle.y_train, lookback)
    val_ds   = MultiAssetDataset(bundle.X_val,   bundle.y_val,   lookback)
    test_ds  = MultiAssetDataset(bundle.X_test,  bundle.y_test,  lookback)

    # Note: shuffle=True for train ONLY. Validation/test must be ordered so
    # we can map predictions back to dates for the backtest.
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, drop_last=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, drop_last=False,
    )
    return train_loader, val_loader, test_loader
