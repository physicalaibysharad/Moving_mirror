"""Construct the FNO from a configuration.

The ``neuralop`` package is vendored at the repository root but is not
necessarily pip-installed, so the vendored copy is added to ``sys.path`` as a
fallback before importing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from fno_tensors import TensorSpec
from model_config import FNOConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
VENDORED_NEURALOP = REPO_ROOT / "neuraloperator"


def _import_fno():
    try:
        from neuralop.models import FNO  # noqa: PLC0415
    except ImportError:
        if not VENDORED_NEURALOP.exists():
            raise ImportError(
                "neuralop is not installed and no vendored copy was found at "
                f"{VENDORED_NEURALOP}. Install it with:\n"
                f"  pip install -e {VENDORED_NEURALOP}"
            ) from None
        sys.path.insert(0, str(VENDORED_NEURALOP))
        from neuralop.models import FNO  # noqa: PLC0415
    return FNO


def build_model(
    config: FNOConfig,
    spec: TensorSpec,
    n_omega: int,
    n_omega_prime: int,
) -> torch.nn.Module:
    """Instantiate the FNO, deriving ``in_channels`` from the tensor packing."""
    config.validate_against_grid(n_omega, n_omega_prime)

    if config.in_channels is None:
        config.in_channels = spec.in_channels
    elif config.in_channels != spec.in_channels:
        raise ValueError(
            f"FNOConfig.in_channels={config.in_channels} disagrees with the packed "
            f"input, which has {spec.in_channels} channels."
        )

    FNO = _import_fno()
    return FNO(**config.to_kwargs())


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def resolve_device(requested: str) -> torch.device:
    """``"auto"`` picks CUDA when available, otherwise CPU."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device 'cuda' requested but no CUDA device is available")
    return device
