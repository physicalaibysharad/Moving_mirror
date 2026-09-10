"""Input standardisation, fitted on the training split only.

The trajectories of the two families live on different scales -- good2016 has
z in about [-10, 0], glw2021 in about [-15, 15] -- so the z channels are
standardised before they reach the model.  Statistics are computed per time
index over the training split alone, so no validation sample influences them.

The target (Re beta, Im beta) is left in physical units by default, so the
reported element-wise MSE is directly comparable with the value quoted in the
manuscript.  ``Standardizer`` can be applied to the target as well if a run
wants that; ``train.py`` exposes it as a flag.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class Standardizer:
    """Affine rescaling ``(x - mean) / std`` with a broadcastable mean/std."""

    mean: torch.Tensor
    std: torch.Tensor
    eps: float = 1.0e-8

    @classmethod
    def fit(cls, x: torch.Tensor, dim: tuple[int, ...] = (0,), eps: float = 1.0e-8) -> "Standardizer":
        """Fit over ``dim``, keeping the remaining axes as per-feature statistics."""
        mean = x.mean(dim=dim, keepdim=True)
        std = x.std(dim=dim, keepdim=True)
        # A constant feature carries no information; leaving std at 0 would
        # produce inf, so it is pinned to 1 and the feature becomes all zeros.
        std = torch.where(std > eps, std, torch.ones_like(std))
        return cls(mean=mean, std=std, eps=eps)

    @classmethod
    def identity(cls) -> "Standardizer":
        return cls(mean=torch.zeros(1), std=torch.ones(1))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x.device, x.dtype)) / self.std.to(x.device, x.dtype)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Undo :meth:`encode`, to return predictions to physical units."""
        return x * self.std.to(x.device, x.dtype) + self.mean.to(x.device, x.dtype)

    def state_dict(self) -> dict[str, Any]:
        return {"mean": self.mean, "std": self.std, "eps": self.eps}

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "Standardizer":
        return cls(mean=state["mean"], std=state["std"], eps=state.get("eps", 1.0e-8))
