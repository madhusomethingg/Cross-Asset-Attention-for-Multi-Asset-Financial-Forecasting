"""
GRU baseline -- same interface as the LSTM, with simpler gated recurrent units.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class GRUForecaster(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_assets: int,
        hidden_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_assets   = n_assets
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor, return_attn: bool = False) -> torch.Tensor:
        B, L, N, F_in = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B * N, L, F_in)
        out, _ = self.gru(x)
        last = out[:, -1, :]
        pred = self.head(last).squeeze(-1)
        return pred.reshape(B, N)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
