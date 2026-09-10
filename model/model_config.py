"""Architecture configuration for the FNO that maps z(t) to beta.

The defaults reproduce the FNO described in the Experiments section of
``Moving_mirror_manuscript/moving_mirror.tex``: 4 layers, Fourier modes
(12, 12) and 64 hidden channels.

The operator is 2-D: it acts on the (omega, omega') grid on which beta is
defined.  The trajectory enters as a set of channels that are constant across
that grid (see :mod:`fno_tensors`), so the input channel count is derived from
the tensor-packing configuration rather than set by hand.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any


@dataclass
class FNOConfig:
    """Hyper-parameters passed straight to ``neuralop.models.FNO``."""

    # --- from the manuscript -------------------------------------------- #
    n_modes: tuple[int, int] = (12, 12)
    hidden_channels: int = 64
    n_layers: int = 4

    # --- channel counts -------------------------------------------------- #
    # in_channels is filled in from the TensorSpec (n_z_features + families +
    # 2 coordinate channels); out_channels is 2 for (Re beta, Im beta).
    in_channels: int | None = None
    out_channels: int = 2

    # --- neuralop defaults, exposed so they are recorded in every run ---- #
    lifting_channel_ratio: int = 2
    projection_channel_ratio: int = 2
    positional_embedding: str | None = "grid"
    norm: str | None = None
    use_channel_mlp: bool = True
    channel_mlp_dropout: float = 0.0
    channel_mlp_expansion: float = 0.5
    factorization: str | None = None
    rank: float = 1.0

    def __post_init__(self) -> None:
        n_modes = tuple(self.n_modes)
        if len(n_modes) != 2:
            raise ValueError(f"n_modes must be 2-D for a (omega, omega') grid, got {n_modes}")
        self.n_modes = n_modes
        if self.out_channels != 2:
            raise ValueError(
                "out_channels must be 2: the target is (Re beta, Im beta). "
                f"Got {self.out_channels}."
            )

    def validate_against_grid(self, n_omega: int, n_omega_prime: int) -> None:
        """Fourier modes must stay below the Nyquist limit of the output grid."""
        for modes, n, name in zip(self.n_modes, (n_omega, n_omega_prime), ("omega", "omega'")):
            if modes > n // 2 + 1:
                raise ValueError(
                    f"n_modes={modes} along {name} exceeds the Nyquist limit "
                    f"{n // 2 + 1} for a {n}-point grid."
                )

    def to_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for the ``FNO`` constructor."""
        if self.in_channels is None:
            raise ValueError(
                "in_channels is unset; call build_model() with a TensorSpec, "
                "which derives it from the packed input channels."
            )
        return asdict(self)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_FNO_CONFIG = FNOConfig()
