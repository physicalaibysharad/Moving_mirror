"""On-disk layout for the generated moving-mirror samples.

One directory per trajectory family, one ``.pt`` file per parameter set, plus a
``manifest.json`` recording the generation config and every sample's
parameters.  Each ``.pt`` holds a plain dict:

===============  =========================  ==============================
key              dtype / shape              meaning
===============  =========================  ==============================
``family``       str                        e.g. ``"good2016_lambertw"``
``paper``        str                        arXiv id and title
``index``        int                        position in the manifest
``params``       dict of float              the sampled parameters
``t``            float64 ``(n_t,)``         time grid (shared by all samples)
``z``            float64 ``(n_t,)``         mirror position z(t) (papers' x)
``omega``        float64 ``(n_w,)``         out-frequency grid
``omega_prime``  float64 ``(n_wp,)``        in-frequency grid
``beta``         complex128 ``(n_w, n_wp)`` ``beta[i, j] = beta_{w_i, w'_j}``
``meta``         dict                       grid config, formula reference
===============  =========================  ==============================

``beta`` is stored complex128 because the good2016 coefficients span ~30 orders
of magnitude across the grid; narrowing to complex64 is left to whatever script
later packs these into model tensors.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def sample_filename(index: int, params: dict[str, float]) -> str:
    """``sample_0007_s0.8123_kappa1.0451.pt`` -- index first so sorting is stable."""
    tags = "_".join(f"{k}{v:.4g}" for k, v in params.items())
    return f"sample_{index:04d}_{tags}.pt"


def save_sample(
    out_dir: Path,
    index: int,
    family: str,
    paper: str,
    params: dict[str, float],
    t: np.ndarray,
    z: np.ndarray,
    omega: np.ndarray,
    omega_prime: np.ndarray,
    beta: np.ndarray,
    meta: dict[str, Any],
) -> str:
    """Write one parameter set to ``out_dir`` and return its filename."""
    if not np.all(np.isfinite(z)):
        raise ValueError(f"non-finite trajectory for {family} sample {index}, params={params}")
    if not np.all(np.isfinite(beta)):
        raise ValueError(f"non-finite beta for {family} sample {index}, params={params}")

    payload = {
        "family": family,
        "paper": paper,
        "index": index,
        "params": {k: float(v) for k, v in params.items()},
        "t": torch.from_numpy(np.ascontiguousarray(t, dtype=np.float64)),
        "z": torch.from_numpy(np.ascontiguousarray(z, dtype=np.float64)),
        "omega": torch.from_numpy(np.ascontiguousarray(omega, dtype=np.float64)),
        "omega_prime": torch.from_numpy(np.ascontiguousarray(omega_prime, dtype=np.float64)),
        "beta": torch.from_numpy(np.ascontiguousarray(beta, dtype=np.complex128)),
        "meta": meta,
    }
    name = sample_filename(index, params)
    torch.save(payload, out_dir / name)
    return name


def write_manifest(out_dir: Path, config: dict[str, Any], entries: list[dict[str, Any]]) -> None:
    """Record the generation config and the per-sample parameters."""
    manifest = {
        "family": config["family"],
        "paper": config["paper"],
        "n_samples": len(entries),
        "config": config,
        "samples": entries,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def load_sample(path: Path) -> dict[str, Any]:
    """Read one sample back.  Convenience for downstream/packing scripts."""
    return torch.load(path, weights_only=False)


def prepare_out_dir(out_dir: Path, overwrite: bool) -> None:
    """Create ``out_dir``, refusing to mix new samples in with stale ones."""
    out_dir = Path(out_dir)
    existing = sorted(out_dir.glob("sample_*.pt")) if out_dir.exists() else []
    if existing and not overwrite:
        raise SystemExit(
            f"{out_dir} already holds {len(existing)} sample(s). "
            f"Pass --overwrite to replace them."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in existing:
        stale.unlink()
