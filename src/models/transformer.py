"""
Cross-Asset Transformer for multi-asset return forecasting.

Architecture:
  Input:  (B, L, N, F)   -- B batches, L lookback days, N assets, F features
  Output: (B, N)          -- next-day log return per asset

The key design choice is HOW we tokenize. We have two axes (time L and assets N)
and self-attention is O(seq_len^2). We use a *factorised* design that runs:

    1. TEMPORAL attention (along L, independently per asset)
       -- learns "what matters in the recent history of asset n"
    2. CROSS-ASSET attention (along N, independently per timestep)
       -- learns "which assets are relevant to asset n right now"

This is much cheaper than fully-flat (L*N) attention and -- importantly --
gives us a *clean cross-asset attention matrix* of shape (N, N) per timestep,
which is exactly what we need for the novelty tab in the demo.

The model exposes `last_cross_attn` after every forward pass: the cross-asset
attention weights from the *final* cross-asset block, averaged over heads.
Shape: (B, L, N, N). For the demo we typically take the last timestep,
giving (B, N, N) -- one (N, N) attention matrix per sample / per day.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Sinusoidal positional encoding (standard, fixed)
# -----------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer('pe', pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., L, d_model)
        L = x.size(-2)
        return x + self.pe[:, :L, :]


# -----------------------------------------------------------------------------
# Transformer block with explicit attention-weight return
# -----------------------------------------------------------------------------
class AttnBlock(nn.Module):
    """
    Pre-LN Transformer encoder block. Built on nn.MultiheadAttention with
    `need_weights=True` so we can extract attention weights for the demo.
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, return_weights: bool = False):
        """
        x: (B*, S, d_model)  -- B* is whatever batch dim the caller bundled

        Returns:
            x_out: (B*, S, d_model)
            attn:  (B*, S, S) averaged over heads if return_weights, else None
        """
        h = self.ln1(x)
        # average_attn_weights=True -> we get a single (B, S, S) matrix
        attn_out, attn_w = self.attn(
            h, h, h,
            need_weights=return_weights,
            average_attn_weights=True,
        )
        x = x + self.drop(attn_out)
        x = x + self.drop(self.ff(self.ln2(x)))
        return x, attn_w


# -----------------------------------------------------------------------------
# Main model
# -----------------------------------------------------------------------------
class CrossAssetTransformer(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_assets: int,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
        max_lookback: int = 128,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_assets   = n_assets
        self.d_model    = d_model

        # --- Input projection: F -> d_model
        self.input_proj = nn.Linear(n_features, d_model)

        # --- Positional encoding (added along time)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_lookback)

        # --- Asset embedding (added along the asset axis, learned)
        self.asset_emb = nn.Embedding(n_assets, d_model)

        # --- Stacked (temporal block, cross-asset block) pairs
        self.temporal_blocks = nn.ModuleList([
            AttnBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.cross_blocks = nn.ModuleList([
            AttnBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])

        # --- Output head: per-asset scalar
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        # --- Holder for the most recent cross-asset attention matrix
        # (set during forward when return_attn=True)
        self.last_cross_attn: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, return_attn: bool = False) -> torch.Tensor:
        """
        x: (B, L, N, F)
        Returns predictions: (B, N)

        If return_attn=True, also stores the cross-asset attention matrix
        from the final cross-asset block in `self.last_cross_attn`,
        with shape (B, L, N, N).
        """
        B, L, N, F_in = x.shape
        if N != self.n_assets:
            raise ValueError(
                f"Expected {self.n_assets} assets, got {N}"
            )

        # ---- Project features
        x = self.input_proj(x)                          # (B, L, N, d)

        # ---- Add asset embedding (broadcast over time)
        asset_ids = torch.arange(N, device=x.device)
        ae = self.asset_emb(asset_ids)                  # (N, d)
        x = x + ae.view(1, 1, N, -1)                    # (B, L, N, d)

        # ---- Add positional encoding (broadcast over assets)
        # Reshape to (B*N, L, d) so PE adds along time
        x = x.permute(0, 2, 1, 3).reshape(B * N, L, -1) # (B*N, L, d)
        x = self.pos_enc(x)
        x = x.reshape(B, N, L, -1).permute(0, 2, 1, 3)  # back to (B, L, N, d)

        last_cross_w = None
        for t_block, c_block in zip(self.temporal_blocks, self.cross_blocks):
            # ---- Temporal attention: each asset attends within its own L-history
            xt = x.permute(0, 2, 1, 3).reshape(B * N, L, -1)   # (B*N, L, d)
            xt, _ = t_block(xt, return_weights=False)
            x = xt.reshape(B, N, L, -1).permute(0, 2, 1, 3)    # (B, L, N, d)

            # ---- Cross-asset attention: at each timestep, assets attend to each other
            xc = x.reshape(B * L, N, -1)                       # (B*L, N, d)
            xc, attn_w = c_block(xc, return_weights=return_attn)
            x = xc.reshape(B, L, N, -1)
            if return_attn and attn_w is not None:
                last_cross_w = attn_w.reshape(B, L, N, N)      # save final layer

        if return_attn:
            self.last_cross_attn = last_cross_w
        else:
            self.last_cross_attn = None

        # ---- Pool over time (use the last timestep -- standard for forecasting)
        x_last = x[:, -1, :, :]                                # (B, N, d)
        x_last = self.norm(x_last)
        out = self.head(x_last).squeeze(-1)                    # (B, N)
        return out

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
