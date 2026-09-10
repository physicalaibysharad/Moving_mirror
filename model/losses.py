"""Loss and evaluation metrics.

The training objective is the element-wise MSE used in the manuscript, taken
over both output channels (Re beta, Im beta) and the whole (omega, omega')
grid.

Two diagnostics are reported alongside it, because element-wise MSE on raw beta
is dominated by the largest entries -- |beta| spans about twelve orders of
magnitude across the grid, so a model that predicted zero everywhere would
already score a small MSE.  The relative L2 error is scale-free and exposes
that failure mode; the per-family breakdown shows whether one family is
carrying the result.
"""

from __future__ import annotations

import torch


def elementwise_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error over every channel and grid point."""
    return torch.mean((pred - target) ** 2)


def relative_l2(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-30) -> torch.Tensor:
    """Per-sample ``||pred - target|| / ||target||``, averaged over the batch.

    Scale-free, so unlike the MSE it cannot be made small by predicting zero.
    """
    dims = tuple(range(1, pred.ndim))
    num = torch.linalg.vector_norm(pred - target, dim=dims)
    den = torch.linalg.vector_norm(target, dim=dims).clamp_min(eps)
    return torch.mean(num / den)


def zero_baseline_mse(target: torch.Tensor) -> torch.Tensor:
    """The MSE a model would score by predicting zero everywhere.

    Any useful model must come in well below this number.
    """
    return torch.mean(target**2)


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader, device: torch.device) -> dict[str, float]:
    """Mean MSE, relative L2 and zero-baseline MSE over a loader."""
    model.eval()
    totals = {"mse": 0.0, "rel_l2": 0.0, "zero_mse": 0.0}
    n_batches = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        totals["mse"] += float(elementwise_mse(pred, y))
        totals["rel_l2"] += float(relative_l2(pred, y))
        totals["zero_mse"] += float(zero_baseline_mse(y))
        n_batches += 1
    if n_batches == 0:
        raise ValueError("empty loader")
    return {k: v / n_batches for k, v in totals.items()}
