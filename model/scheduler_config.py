"""Optimiser and exponential learning-rate decay configuration.

Kept separate from the architecture so the decay schedule can be changed
without touching the model definition.

The schedule is ``torch.optim.lr_scheduler.ExponentialLR``: after every epoch
the learning rate is multiplied by ``gamma``, so

    lr(epoch) = lr_initial * gamma ** epoch

With the defaults below (lr 1e-3, gamma 0.9, 20 epochs as in the manuscript)
the rate falls to 1e-3 * 0.9**20 = 1.2e-4 by the final epoch.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import torch


@dataclass
class OptimizerConfig:
    """Adam settings."""

    name: str = "adam"
    lr: float = 1.0e-3
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
class ExponentialDecayConfig:
    """Exponential learning-rate decay.

    Parameters
    ----------
    gamma
        Multiplicative factor applied to the learning rate at each step.
        Must satisfy ``0 < gamma <= 1``.
    step_every_epoch
        If True the scheduler advances once per epoch (the usual choice, and
        what the ``lr(epoch)`` formula above assumes).  If False it advances
        once per optimiser step, which decays far faster for the same gamma.
    min_lr
        Floor below which the learning rate is not allowed to fall.  ``None``
        disables the floor.
    """

    gamma: float = 0.9
    step_every_epoch: bool = True
    min_lr: float | None = 1.0e-6

    def __post_init__(self) -> None:
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError(f"gamma must satisfy 0 < gamma <= 1, got {self.gamma}")
        if self.min_lr is not None and self.min_lr < 0.0:
            raise ValueError(f"min_lr must be non-negative, got {self.min_lr}")

    def build(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.ExponentialLR:
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.gamma)

    def clamp(self, optimizer: torch.optim.Optimizer) -> None:
        """Apply the ``min_lr`` floor after a scheduler step."""
        if self.min_lr is None:
            return
        for group in optimizer.param_groups:
            group["lr"] = max(group["lr"], self.min_lr)

    def lr_at(self, epoch: int, lr_initial: float) -> float:
        """The learning rate the schedule reaches at ``epoch`` (for logging)."""
        lr = lr_initial * self.gamma**epoch
        return lr if self.min_lr is None else max(lr, self.min_lr)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_OPTIMIZER_CONFIG = OptimizerConfig()
DEFAULT_DECAY_CONFIG = ExponentialDecayConfig()
