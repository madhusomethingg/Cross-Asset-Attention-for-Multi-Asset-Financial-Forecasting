"""
Temporal Convolutional Network (TCN) baseline.

Follows Bai, Kolter, Koltun (2018): stacked dilated causal 1-D convolutions
with residual connections. Captures long-range dependencies through dilation
without recurrence.

We apply the TCN independently per asset (same setup as LSTM/GRU) so that the
comparison isolates the effect of cross-asset attention in the Transformer.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """1-D causal convolution: pads on the left so output[t] depends only on input[<=t]."""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class TCNBlock(nn.Module):
    """Two stacked dilated causal convs with a residual connection."""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int,
                 dilation: int, dropout: float):
        super().__init__()
        self.conv1 = CausalConv1d(in_ch, out_ch, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_ch, out_ch, kernel_size, dilation)
        self.drop = nn.Dropout(dropout)
        self.res = (nn.Conv1d(in_ch, out_ch, 1)
                    if in_ch != out_ch else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.res(x)
        h = F.relu(self.conv1(x))
        h = self.drop(h)
        h = F.relu(self.conv2(h))
        h = self.drop(h)
        return F.relu(h + residual)


class TCNForecaster(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_assets: int,
        channels: list[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        if channels is None:
            channels = [64, 64, 64, 64]
        self.n_features = n_features
        self.n_assets   = n_assets

        layers = []
        in_ch = n_features
        for i, out_ch in enumerate(channels):
            dilation = 2 ** i           # 1, 2, 4, 8 -> receptive field ~2^L
            layers.append(TCNBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch
        self.tcn = nn.Sequential(*layers)

        last_ch = channels[-1]
        self.head = nn.Sequential(
            nn.Linear(last_ch, last_ch // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(last_ch // 2, 1),
        )

    def forward(self, x: torch.Tensor, return_attn: bool = False) -> torch.Tensor:
        # x: (B, L, N, F) -> per-asset TCN. Reshape to (B*N, F, L) for Conv1d.
        B, L, N, F_in = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B * N, F_in, L)
        h = self.tcn(x)                       # (B*N, C, L)
        last = h[:, :, -1]                    # (B*N, C)
        pred = self.head(last).squeeze(-1)    # (B*N,)
        return pred.reshape(B, N)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
