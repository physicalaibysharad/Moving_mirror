"""Convert generated moving-mirror samples into FNO input/target tensors.

The operator is 2-D and lives on the (omega, omega') grid, because that is the
grid the target beta is defined on.  The trajectory z(t) is 1-D, so it is
carried in as channels that are **constant across that grid**; the FNO then
learns a beta field conditioned on the trajectory.

Input channels, in order:

===========================  =========================================
``n_z_features``             z(t), one channel per time sample, each
                             broadcast over the whole (omega, omega') grid
``len(families)``            one-hot family indicator (also broadcast)
``2``                        the omega and omega' coordinate grids
===========================  =========================================

Target channels: ``(Re beta, Im beta)``.

Time is normalised per family.  The two families were generated on different
windows -- good2016 on t in [-6, 12], glw2021 on t in [-21, 21] -- so each
family's own window is mapped onto [0, 1] before z is read off.  Since the time
grid is shared by every sample within a family, this is a relabelling of the
axis and discards nothing; the one-hot family channel tells the model which
window a sample came from.

Beta is stored on disk as complex128 and is kept at that precision here:
``TensorSpec.dtype`` defaults to ``torch.float64``, matching the double
precision the samples were generated in.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any

import numpy as np
import torch

FAMILIES: tuple[str, ...] = ("good2016_lambertw", "glw2021_casimir")


@dataclass
class TensorSpec:
    """How a sample dict is packed into tensors."""

    n_z_features: int = 128
    families: tuple[str, ...] = FAMILIES
    family_channels: bool = True
    coord_channels: bool = True
    normalize_time: bool = True
    dtype: torch.dtype = torch.float64

    def __post_init__(self) -> None:
        self.families = tuple(self.families)
        if self.n_z_features < 2:
            raise ValueError(f"n_z_features must be at least 2, got {self.n_z_features}")

    @property
    def in_channels(self) -> int:
        n = self.n_z_features
        if self.family_channels:
            n += len(self.families)
        if self.coord_channels:
            n += 2
        return n

    @property
    def out_channels(self) -> int:
        return 2

    def channel_names(self) -> list[str]:
        """Human-readable channel labels, for debugging and logging."""
        names = [f"z[{i}]" for i in range(self.n_z_features)]
        if self.family_channels:
            names += [f"is_{fam}" for fam in self.families]
        if self.coord_channels:
            names += ["omega", "omega_prime"]
        return names

    def family_index(self, family: str) -> int:
        try:
            return self.families.index(family)
        except ValueError:
            raise KeyError(
                f"unknown family {family!r}; TensorSpec knows {self.families}"
            ) from None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["dtype"] = str(self.dtype)
        d["in_channels"] = self.in_channels
        return d


# --------------------------------------------------------------------------- #
# trajectory -> feature vector
# --------------------------------------------------------------------------- #

def resample_trajectory(t: np.ndarray, z: np.ndarray, n_out: int, normalize_time: bool = True) -> np.ndarray:
    """Read ``z`` off at ``n_out`` evenly spaced points of its own time window.

    With ``n_out`` equal to the native grid size this is the identity (up to
    floating-point round-off), so the default 128 keeps the full trajectory.
    """
    t = np.asarray(t, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    if t.shape != z.shape:
        raise ValueError(f"t and z must have the same shape, got {t.shape} and {z.shape}")
    if not np.all(np.diff(t) > 0):
        raise ValueError("time grid must be strictly increasing")

    if normalize_time:
        span = t[-1] - t[0]
        if span <= 0:
            raise ValueError("degenerate time window")
        u = (t - t[0]) / span
    else:
        u = t

    if n_out == t.size and normalize_time:
        return z.copy()

    u_out = np.linspace(u[0], u[-1], n_out)
    return np.interp(u_out, u, z)


# --------------------------------------------------------------------------- #
# sample -> tensors
# --------------------------------------------------------------------------- #

def pack_input(sample: dict[str, Any], spec: TensorSpec) -> torch.Tensor:
    """Build the ``(C, n_omega, n_omega_prime)`` input tensor for one sample."""
    omega = np.asarray(sample["omega"], dtype=np.float64)
    omega_prime = np.asarray(sample["omega_prime"], dtype=np.float64)
    n_w, n_wp = omega.size, omega_prime.size

    z_feat = resample_trajectory(
        np.asarray(sample["t"]), np.asarray(sample["z"]),
        spec.n_z_features, spec.normalize_time,
    )

    planes = [np.broadcast_to(v, (n_w, n_wp)) for v in z_feat]

    if spec.family_channels:
        idx = spec.family_index(sample["family"])
        for k in range(len(spec.families)):
            planes.append(np.full((n_w, n_wp), 1.0 if k == idx else 0.0))

    if spec.coord_channels:
        planes.append(np.broadcast_to(_unit_scale(omega)[:, None], (n_w, n_wp)))
        planes.append(np.broadcast_to(_unit_scale(omega_prime)[None, :], (n_w, n_wp)))

    stacked = np.stack([np.ascontiguousarray(p, dtype=np.float64) for p in planes], axis=0)
    if stacked.shape[0] != spec.in_channels:
        raise AssertionError(
            f"packed {stacked.shape[0]} channels but TensorSpec declares {spec.in_channels}"
        )
    return torch.from_numpy(stacked).to(spec.dtype)


def pack_target(sample: dict[str, Any], spec: TensorSpec) -> torch.Tensor:
    """Build the ``(2, n_omega, n_omega_prime)`` target: real and imaginary beta."""
    beta = sample["beta"]
    beta = beta.numpy() if isinstance(beta, torch.Tensor) else np.asarray(beta)
    if not np.all(np.isfinite(beta)):
        raise ValueError("non-finite beta in sample; the generators should have caught this")
    target = np.stack([beta.real, beta.imag], axis=0)
    return torch.from_numpy(np.ascontiguousarray(target)).to(spec.dtype)


def pack_sample(sample: dict[str, Any], spec: TensorSpec) -> tuple[torch.Tensor, torch.Tensor]:
    """``(input, target)`` for one sample."""
    return pack_input(sample, spec), pack_target(sample, spec)


def unpack_prediction(pred: torch.Tensor) -> torch.Tensor:
    """Turn a ``(..., 2, n_w, n_wp)`` prediction back into complex beta."""
    if pred.shape[-3] != 2:
        raise ValueError(f"expected 2 channels (Re, Im), got shape {tuple(pred.shape)}")
    re, im = pred[..., 0, :, :], pred[..., 1, :, :]
    return torch.complex(re, im)


def _unit_scale(x: np.ndarray) -> np.ndarray:
    """Map a monotone grid onto [0, 1]."""
    lo, hi = float(x[0]), float(x[-1])
    if hi <= lo:
        raise ValueError(f"non-increasing grid: [{lo}, {hi}]")
    return (np.asarray(x, dtype=np.float64) - lo) / (hi - lo)
