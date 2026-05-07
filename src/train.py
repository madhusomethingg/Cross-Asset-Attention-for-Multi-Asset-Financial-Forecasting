"""
Generic training loop for the deep learning models.
Used uniformly by Transformer / LSTM / GRU / TCN so the comparison is fair.

Features:
  - Device auto-select (MPS on Mac, CUDA on Linux/Windows, CPU fallback)
  - AdamW + cosine annealing
  - Gradient clipping
  - Early stopping on validation loss
  - Returns predictions on val + test for downstream evaluation
"""
from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Device selection
# -----------------------------------------------------------------------------
def get_device() -> torch.device:
    """
    Prefer MPS (Apple Silicon), then CUDA, then CPU.
    Honest note: MPS is occasionally flaky for some ops in older PyTorch
    versions; we set fall-back on missing ops via the env var, done by caller.
    """
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


# -----------------------------------------------------------------------------
# Result container
# -----------------------------------------------------------------------------
@dataclass
class TrainResult:
    model: nn.Module
    best_val_loss: float
    val_preds: np.ndarray         # (n_val_samples, N)
    test_preds: np.ndarray        # (n_test_samples, N)
    train_losses: list[float]
    val_losses: list[float]
    epochs_trained: int
    seconds: float


# -----------------------------------------------------------------------------
# Inference helper
# -----------------------------------------------------------------------------
@torch.no_grad()
def predict_loader(model: nn.Module, loader: DataLoader,
                   device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Run model over a loader; return (preds, targets), each (n_samples, N)."""
    model.eval()
    all_preds, all_y = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        out = model(xb)
        all_preds.append(out.cpu().numpy())
        all_y.append(yb.numpy())
    return np.concatenate(all_preds, axis=0), np.concatenate(all_y, axis=0)


# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    test_loader:  DataLoader,
    *,
    max_epochs: int = 60,
    lr: float = 3e-4,
    weight_decay: float = 1e-5,
    grad_clip: float = 1.0,
    early_stopping_patience: int = 10,
    device: torch.device | None = None,
    model_name: str = 'model',
    loss_fn: nn.Module | None = None,
) -> TrainResult:
    """
    Generic training loop.

    `loss_fn` defaults to nn.MSELoss(). Pass a custom callable to use any
    other differentiable loss (e.g. DirectionAwareLoss, CombinedLoss from
    src.losses). The validation loss is always reported as plain MSE so
    early stopping decisions remain consistent across loss choices.
    """
    if device is None:
        device = get_device()
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"[{model_name}] device={device}  params={n_params:,}")

    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = CosineAnnealingLR(opt, T_max=max_epochs)
    train_loss_fn = loss_fn if loss_fn is not None else nn.MSELoss()
    val_loss_fn   = nn.MSELoss()   # validation always tracks plain MSE
    if loss_fn is not None:
        logger.info(f"[{model_name}] using custom loss: {type(loss_fn).__name__}")

    best_val = float('inf')
    best_state = copy.deepcopy(model.state_dict())
    patience = 0
    train_losses, val_losses = [], []

    t0 = time.time()
    for epoch in range(1, max_epochs + 1):
        # ---- Train
        model.train()
        running, n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = train_loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            running += loss.item() * xb.size(0)
            n += xb.size(0)
        sched.step()
        train_loss = running / n
        train_losses.append(train_loss)

        # ---- Validate (always plain MSE for fair comparison + early stopping)
        model.eval()
        v_running, v_n = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                v_running += val_loss_fn(pred, yb).item() * xb.size(0)
                v_n += xb.size(0)
        val_loss = v_running / v_n
        val_losses.append(val_loss)

        msg = (f"[{model_name}] epoch {epoch:3d}  "
               f"train {train_loss:.6f}  val {val_loss:.6f}")
        if val_loss < best_val - 1e-7:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
            msg += "  *"
        else:
            patience += 1
        logger.info(msg)
        if patience >= early_stopping_patience:
            logger.info(f"[{model_name}] early stopping at epoch {epoch}")
            break

    elapsed = time.time() - t0
    model.load_state_dict(best_state)
    val_preds, _   = predict_loader(model, val_loader,  device)
    test_preds, _  = predict_loader(model, test_loader, device)

    return TrainResult(
        model=model,
        best_val_loss=best_val,
        val_preds=val_preds,
        test_preds=test_preds,
        train_losses=train_losses,
        val_losses=val_losses,
        epochs_trained=len(train_losses),
        seconds=elapsed,
    )
