"""
Loss functions for direction-aware financial forecasting.

The standard MSE loss treats all errors equally and only cares about magnitude.
For trading, we care more about *sign correctness* (UP vs DOWN). And our test
data is class-imbalanced (66% UP, 33% DOWN). Both issues are addressable
through the loss function:

  1. DirectionAwareLoss
     L = MSE  +  lambda_dir * sign_penalty
     where sign_penalty = ReLU(-pred * actual)
     i.e., we add a penalty whenever pred and actual have opposite signs.
     This makes the model directly care about getting the direction right.

  2. WeightedMSE
     L = w * (pred - actual)^2
     where w is larger for DOWN samples (the minority class).
     This forces the model to take downward moves seriously.

  3. CombinedLoss
     Mixes both: direction-aware AND class-weighted.

All losses are differentiable and drop-in replacements for nn.MSELoss().
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DirectionAwareLoss(nn.Module):
    """
    Hybrid loss: standard MSE plus a sign-mismatch penalty.

    L = MSE(pred, actual) + lambda_dir * mean(ReLU(-pred * actual))

    The sign-mismatch term is positive whenever pred and actual disagree on
    sign (their product is negative); ReLU keeps it differentiable. This loss
    smoothly biases gradients toward sign correctness while still training
    the magnitude through MSE.
    """
    def __init__(self, lambda_dir: float = 5.0):
        super().__init__()
        self.lambda_dir = lambda_dir

    def forward(self, pred: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
        mse = F.mse_loss(pred, actual)
        # sign mismatch penalty: positive when pred * actual < 0
        sign_penalty = F.relu(-pred * actual).mean()
        return mse + self.lambda_dir * sign_penalty


class WeightedMSE(nn.Module):
    """
    Class-weighted MSE. We give DOWN-day samples a higher weight to fight
    the natural bullish bias in the training data.

    weight = down_weight if actual < 0 else 1.0
    L = mean( weight * (pred - actual)^2 )
    """
    def __init__(self, down_weight: float = 2.0):
        super().__init__()
        self.down_weight = down_weight

    def forward(self, pred: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
        weight = torch.where(
            actual < 0,
            torch.full_like(actual, self.down_weight),
            torch.ones_like(actual),
        )
        return (weight * (pred - actual) ** 2).mean()


class CombinedLoss(nn.Module):
    """
    Combines DirectionAwareLoss and WeightedMSE.

    L = weighted_MSE  +  lambda_dir * sign_penalty
    """
    def __init__(self, lambda_dir: float = 5.0, down_weight: float = 2.0):
        super().__init__()
        self.lambda_dir = lambda_dir
        self.down_weight = down_weight

    def forward(self, pred: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
        weight = torch.where(
            actual < 0,
            torch.full_like(actual, self.down_weight),
            torch.ones_like(actual),
        )
        wmse = (weight * (pred - actual) ** 2).mean()
        sign_penalty = F.relu(-pred * actual).mean()
        return wmse + self.lambda_dir * sign_penalty


def make_loss(name: str = 'mse', **kwargs) -> nn.Module:
    """Factory: build the requested loss by string name.

    Extra kwargs not relevant to the chosen loss are silently ignored, so a
    config block can define lambda_dir + down_weight and any of the four
    loss types will accept it.
    """
    name = name.lower()
    if name == 'mse':
        return nn.MSELoss()
    if name == 'direction':
        return DirectionAwareLoss(
            lambda_dir=kwargs.get('lambda_dir', 5.0),
        )
    if name == 'weighted':
        return WeightedMSE(
            down_weight=kwargs.get('down_weight', 2.0),
        )
    if name == 'combined':
        return CombinedLoss(
            lambda_dir=kwargs.get('lambda_dir', 5.0),
            down_weight=kwargs.get('down_weight', 2.0),
        )
    raise ValueError(f"unknown loss: {name}")
