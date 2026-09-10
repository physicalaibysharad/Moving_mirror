"""Optimiser and learning-rate schedule configuration.

Kept separate from the architecture so the schedule can be changed without
touching the model definition.

The schedule is a two-phase geometric (log-linear) warmup-then-decay:

    epoch 0            -> lr_start  (1e-4)
    warmup_fraction * n -> lr_peak  (1e-2)
    last epoch (n - 1)  -> lr_end   (1e-6)

Each phase interpolates geometrically, i.e. ``lr(epoch) = a * r**epoch`` for
some ``a, r > 0``, climbing from ``lr_start`` to ``lr_peak`` on the way up and
falling from ``lr_peak`` to ``lr_end`` on the way down. A curve of that form
has ``d^2(lr)/d(epoch)^2 = a * r**epoch * (ln r)^2 >= 0`` everywhere, so both
legs are convex (bow upward) regardless of whether ``r`` is above or below 1
-- the warmup accelerates into the peak and the decay decelerates out of it,
rather than moving in straight lines.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import torch


@dataclass
class OptimizerConfig:
    """Adam settings."""

    name: str = "adam"
    lr: float = 1.0e-4
    weight_decay: float = 1.0e-4
    betas: tuple[float, float] = (0.9, 0.999)

    def build(self, parameters) -> torch.optim.Optimizer:
        if self.name.lower() != "adam":
            raise ValueError(f"unsupported optimizer {self.name!r}; only 'adam' is implemented")
        return torch.optim.Adam(
            parameters, lr=self.lr, weight_decay=self.weight_decay, betas=tuple(self.betas)
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WarmupDecayConfig:
    """Two-phase geometric warmup-then-decay learning-rate schedule.

    Parameters
    ----------
    lr_start
        Learning rate at epoch 0. Must equal ``OptimizerConfig.lr``, since
        that is what the optimiser is actually constructed with.
    lr_peak
        Learning rate reached at the end of the warmup phase.
    lr_end
        Learning rate reached at the last epoch.
    warmup_fraction
        Where the peak sits, as a fraction of the run. The peak epoch is
        ``round(warmup_fraction * (n_epochs - 1))``, clamped to leave at
        least one epoch in each phase.
    step_every_epoch
        Must be True: the schedule is indexed by epoch, not by optimiser
        step.
    """

    lr_start: float = 1.0e-4
    lr_peak: float = 1.0e-2
    lr_end: float = 1.0e-6
    warmup_fraction: float = 0.15
    step_every_epoch: bool = True

    def __post_init__(self) -> None:
        if not (self.lr_start > 0 and self.lr_peak > 0 and self.lr_end > 0):
            raise ValueError("lr_start, lr_peak and lr_end must all be positive")
        if not 0.0 < self.warmup_fraction < 1.0:
            raise ValueError(f"warmup_fraction must lie in (0, 1), got {self.warmup_fraction}")
        if not self.step_every_epoch:
            raise ValueError("WarmupDecayConfig is indexed by epoch; step_every_epoch must be True")

    def _peak_epoch(self, n_epochs: int) -> int:
        if n_epochs < 3:
            raise ValueError("need at least 3 epochs for a warmup + decay schedule")
        peak = round(self.warmup_fraction * (n_epochs - 1))
        return min(max(peak, 1), n_epochs - 2)

    def lr_at(self, epoch: int, n_epochs: int) -> float:
        """The learning rate the schedule reaches at ``epoch`` (for logging).

        ``epoch`` is 0-indexed and clamped to ``[0, n_epochs - 1]``.
        """
        epoch = min(max(epoch, 0), n_epochs - 1)
        peak = self._peak_epoch(n_epochs)
        if epoch <= peak:
            frac = epoch / peak
            return self.lr_start * (self.lr_peak / self.lr_start) ** frac
        frac = (epoch - peak) / (n_epochs - 1 - peak)
        return self.lr_peak * (self.lr_end / self.lr_peak) ** frac

    def build(self, optimizer: torch.optim.Optimizer, n_epochs: int) -> torch.optim.lr_scheduler.LambdaLR:
        base_lr = optimizer.param_groups[0]["lr"]
        if abs(base_lr - self.lr_start) > 1e-12 * max(1.0, self.lr_start):
            raise ValueError(
                f"optimizer was built with lr={base_lr:g}, but the schedule starts at "
                f"lr_start={self.lr_start:g}. Set OptimizerConfig.lr = lr_start."
            )
        n_epochs_ref = n_epochs  # captured for the closure
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda step: self.lr_at(step, n_epochs_ref) / self.lr_start
        )

    def clamp(self, optimizer: torch.optim.Optimizer) -> None:
        """Floor the learning rate at ``lr_end`` (guards against float drift)."""
        for group in optimizer.param_groups:
            group["lr"] = max(group["lr"], self.lr_end)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_OPTIMIZER_CONFIG = OptimizerConfig()
DEFAULT_DECAY_CONFIG = WarmupDecayConfig()
