"""
Attention extraction and analysis for the Cross-Asset Transformer.

The novel contribution of this project is to compare the model's *learned*
cross-asset attention with the classical *rolling Pearson correlation*.
Both are (N x N) matrices that try to capture cross-asset relationships,
but they differ in important ways:

  - Rolling correlation is computed from past-window returns. It is data-driven
    but model-free and tends to lag regime changes (it has to wait for the
    new regime to fill the window).

  - Cross-asset attention is computed by the trained model from the most
    recent feature snapshot. It is data-driven AND model-driven; in
    principle it can react instantaneously to new information because each
    softmax is computed fresh per timestep.

We carefully phrase claims as:
    "attention reflects how the model allocates importance across assets"
    "attention can reveal changing dependencies"

We do NOT claim:
    "attention shows true relationships"
    "attention = correlation"

This module provides:
  - extract_attention()        : run the model on every test window
  - rolling_correlation()      : classical baseline matrix per day
  - attention_to_dependency()  : symmetrise attention into a comparable shape
  - attention_concentration()  : entropy-based dispersion score per day
  - top_k_attended()           : which assets does asset i look at most today
  - regime_shift_score()       : how much did attention change vs yesterday
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# 1. Extract attention from a trained Transformer
# -----------------------------------------------------------------------------
@torch.no_grad()
def extract_attention(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_last_timestep_only: bool = True,
) -> np.ndarray:
    """
    Run the trained Transformer over a loader and collect cross-asset attention
    for every sample.

    Returns:
        attn: (T_samples, N, N)  if use_last_timestep_only
              (T_samples, L, N, N) otherwise

    For the demo we only need the final-timestep attention (it captures
    what the model attended to when making the next-day prediction).
    """
    model.eval()
    chunks = []
    for xb, _ in loader:
        xb = xb.to(device)
        _ = model(xb, return_attn=True)
        a = model.last_cross_attn                # (B, L, N, N)
        if a is None:
            raise RuntimeError("Model did not produce attention. "
                               "Did you pass return_attn=True?")
        if use_last_timestep_only:
            a = a[:, -1, :, :]                   # (B, N, N)
        chunks.append(a.cpu().numpy())
    out = np.concatenate(chunks, axis=0)
    logger.info(f"Extracted attention with shape {out.shape}")
    return out


# -----------------------------------------------------------------------------
# 2. Rolling correlation baseline
# -----------------------------------------------------------------------------
def rolling_correlation(
    returns: np.ndarray,           # (T, N) log returns aligned per day
    window: int = 60,
) -> np.ndarray:
    """
    For each day t (with t >= window-1), compute the (N x N) Pearson
    correlation matrix using returns[t-window+1 : t+1].

    Returns: (T, N, N) array.  First (window-1) entries are NaN.
    """
    T, N = returns.shape
    out = np.full((T, N, N), np.nan, dtype=np.float32)
    for t in range(window - 1, T):
        w = returns[t - window + 1 : t + 1]      # (window, N)
        # numpy corrcoef expects (N, T_window)
        c = np.corrcoef(w.T)
        # corrcoef can produce NaNs if any column is constant; replace with 0
        c = np.nan_to_num(c, nan=0.0)
        out[t] = c
    return out


# -----------------------------------------------------------------------------
# 3. Attention -> dependency matrix
# -----------------------------------------------------------------------------
def symmetrise_attention(attn: np.ndarray) -> np.ndarray:
    """
    Attention is row-stochastic (rows sum to 1) and asymmetric:
    attn[i, j] = how much asset i looks at asset j.

    To compare against a Pearson correlation matrix (symmetric, in [-1, 1]),
    we symmetrise by averaging A and A^T. This loses some information but
    makes the side-by-side comparison clean.

    Input:  (..., N, N)  any leading dims
    Output: same shape, symmetrised.
    """
    return 0.5 * (attn + np.swapaxes(attn, -1, -2))


def attention_to_dependency(attn: np.ndarray, zero_diagonal: bool = True
                            ) -> np.ndarray:
    """
    Convert a stack of attention matrices into a 'dependency' matrix that's
    visually comparable to a correlation matrix.

      1) Symmetrise (A + A^T) / 2
      2) Optionally zero out the diagonal (each asset's self-attention is
         usually large and dominates the heatmap)
      3) Re-scale each per-day matrix to [0, 1] for visual stability

    Returns: same shape as input
    """
    out = symmetrise_attention(attn).copy()
    if zero_diagonal:
        N = out.shape[-1]
        idx = np.arange(N)
        out[..., idx, idx] = 0.0
    # Per-matrix rescale to [0, 1] for nicer heatmap rendering
    flat_min = out.reshape(*out.shape[:-2], -1).min(axis=-1, keepdims=True)
    flat_max = out.reshape(*out.shape[:-2], -1).max(axis=-1, keepdims=True)
    denom    = (flat_max - flat_min)[..., None]
    flat_min = flat_min[..., None]
    out = (out - flat_min) / (denom + 1e-12)
    return out


# -----------------------------------------------------------------------------
# 4. Quantitative attention metrics
# -----------------------------------------------------------------------------
def attention_concentration(attn: np.ndarray) -> np.ndarray:
    """
    Per-day mean entropy of attention rows. LOWER entropy = more concentrated.

    We compute entropy in bits and average over rows.

    Input:  (T, N, N) row-stochastic attention
    Output: (T,) mean per-row entropy in bits, range [0, log2(N)]
    """
    T, N, _ = attn.shape
    eps = 1e-12
    # entropy per row
    row_entropy = -np.sum(attn * np.log2(attn + eps), axis=-1)   # (T, N)
    return row_entropy.mean(axis=-1)                              # (T,)


def top_k_attended(attn: np.ndarray, asset_names: list[str],
                   k: int = 3) -> list[list[tuple[str, float]]]:
    """
    For a single attention matrix or a stack, return for each row (asset i)
    the top-k other assets that asset i attended to most strongly.

    Input:  (..., N, N)
    Output: nested list with the same leading shape, each leaf is
            [(asset_name, weight), ...] of length k.

    Practical use: per-day sentence in the demo such as
      "On 2023-03-13, BTC-USD attended most to ETH-USD, QQQ, SPY."
    """
    if attn.ndim == 2:
        return _topk_single(attn, asset_names, k)
    if attn.ndim == 3:
        return [_topk_single(a, asset_names, k) for a in attn]
    raise ValueError(f"unsupported attn shape: {attn.shape}")


def _topk_single(a: np.ndarray, names: list[str], k: int):
    N = a.shape[0]
    out = []
    for i in range(N):
        row = a[i].copy()
        row[i] = -np.inf       # exclude self
        idxs = np.argsort(row)[::-1][:k]
        out.append([(names[j], float(a[i, j])) for j in idxs])
    return out


def regime_shift_score(attn: np.ndarray) -> np.ndarray:
    """
    How much did attention change from day t-1 to day t?
    Frobenius norm of (A_t - A_{t-1}).

    Input:  (T, N, N)
    Output: (T,) with first entry = NaN
    """
    T = attn.shape[0]
    out = np.full(T, np.nan)
    for t in range(1, T):
        out[t] = float(np.linalg.norm(attn[t] - attn[t - 1]))
    return out


def correlation_shift_score(corr: np.ndarray) -> np.ndarray:
    """Same as regime_shift_score but for the rolling correlation matrices."""
    T = corr.shape[0]
    out = np.full(T, np.nan)
    for t in range(1, T):
        if not (np.isfinite(corr[t]).all() and np.isfinite(corr[t - 1]).all()):
            continue
        out[t] = float(np.linalg.norm(corr[t] - corr[t - 1]))
    return out


# -----------------------------------------------------------------------------
# 5. Comparison: how similar are attention and rolling correlation?
# -----------------------------------------------------------------------------
def matrix_similarity_series(
    attn: np.ndarray,
    corr: np.ndarray,
) -> np.ndarray:
    """
    Per-day Spearman correlation between the off-diagonal entries of the
    (symmetrised) attention matrix and the rolling Pearson correlation.

    A rising series means attention is becoming more correlation-like;
    falling means the model is finding structure that correlation misses.

    Input:  attn (T, N, N), corr (T, N, N)
    Output: (T,) Spearman rho per day  (NaN where corr is NaN)
    """
    T, N, _ = attn.shape
    iu = np.triu_indices(N, k=1)
    out = np.full(T, np.nan)
    sym_a = symmetrise_attention(attn)
    for t in range(T):
        if not np.isfinite(corr[t]).all():
            continue
        a_off = sym_a[t][iu]
        c_off = corr[t][iu]
        if np.std(a_off) < 1e-10 or np.std(c_off) < 1e-10:
            continue
        rho, _ = stats.spearmanr(a_off, c_off)
        out[t] = rho
    return out
