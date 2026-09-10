"""Dataset and 80:20 split over the generated moving-mirror samples.

Samples are read once from ``data/files/<family>/sample_*.pt``.  Only the
compact per-sample quantities are held in memory -- the trajectory features and
the (2, n_omega, n_omega') target -- and the constant-across-the-grid input
channels are materialised in ``__getitem__``.  Storing the full input tensor
for all 1000 samples would cost roughly 470 MB; this costs about 8 MB.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from fno_tensors import FAMILIES, TensorSpec, pack_target, resample_trajectory
from normalization import Standardizer

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FILES_ROOT = REPO_ROOT / "data" / "files"


@dataclass
class SplitConfig:
    """How the samples are divided into training and validation sets."""

    val_fraction: float = 0.2          # the manuscript's 80:20 split
    seed: int = 0
    stratify_by_family: bool = True    # keep both families in both splits

    def __post_init__(self) -> None:
        if not 0.0 < self.val_fraction < 1.0:
            raise ValueError(f"val_fraction must lie in (0, 1), got {self.val_fraction}")


class MirrorDataset(Dataset):
    """Trajectory -> Bogoliubov coefficient pairs."""

    def __init__(
        self,
        files_root: Path | str = DEFAULT_FILES_ROOT,
        spec: TensorSpec | None = None,
        families: Iterable[str] = FAMILIES,
        limit_per_family: int | None = None,
    ) -> None:
        self.spec = spec or TensorSpec()
        self.files_root = Path(files_root)
        if not self.files_root.exists():
            raise FileNotFoundError(
                f"{self.files_root} does not exist. Generate the data first:\n"
                f"  cd data/scripts && python gen_good2016.py && python gen_glw2021.py"
            )

        z_features: list[np.ndarray] = []
        targets: list[torch.Tensor] = []
        self.family_index: list[int] = []
        self.paths: list[Path] = []
        self.params: list[dict[str, float]] = []
        self.omega: np.ndarray | None = None
        self.omega_prime: np.ndarray | None = None

        for family in families:
            family_dir = self.files_root / family
            paths = sorted(family_dir.glob("sample_*.pt"))
            if not paths:
                raise FileNotFoundError(f"no samples under {family_dir}")
            if limit_per_family is not None:
                paths = paths[:limit_per_family]

            for path in paths:
                sample = torch.load(path, weights_only=False)
                self._check_grid(sample)
                z_features.append(
                    resample_trajectory(
                        np.asarray(sample["t"]), np.asarray(sample["z"]),
                        self.spec.n_z_features, self.spec.normalize_time,
                    )
                )
                targets.append(pack_target(sample, self.spec))
                self.family_index.append(self.spec.family_index(sample["family"]))
                self.paths.append(path)
                self.params.append(dict(sample["params"]))

        self.z_features = torch.from_numpy(np.stack(z_features)).to(self.spec.dtype)
        self.targets = torch.stack(targets)
        self.family_index_t = torch.tensor(self.family_index, dtype=torch.long)
        self.n_omega = self.targets.shape[-2]
        self.n_omega_prime = self.targets.shape[-1]
        self._z_norm: Standardizer | None = None
        self._coord_planes = self._build_coord_planes()

    # -- grids ------------------------------------------------------------ #

    def _check_grid(self, sample: dict[str, Any]) -> None:
        """Both families must share the (omega, omega') grid; time may differ."""
        omega = np.asarray(sample["omega"], dtype=np.float64)
        omega_prime = np.asarray(sample["omega_prime"], dtype=np.float64)
        if self.omega is None:
            self.omega, self.omega_prime = omega, omega_prime
            return
        if not (np.allclose(omega, self.omega) and np.allclose(omega_prime, self.omega_prime)):
            raise ValueError(
                "samples disagree on the (omega, omega') grid; the model assumes one "
                "shared frequency grid across the whole dataset."
            )

    def _build_coord_planes(self) -> torch.Tensor | None:
        if not self.spec.coord_channels:
            return None
        w = _unit(self.omega)[:, None].repeat(1, self.n_omega_prime)
        wp = _unit(self.omega_prime)[None, :].repeat(self.n_omega, 1)
        return torch.stack([w, wp]).to(self.spec.dtype)

    # -- normalisation ---------------------------------------------------- #

    def fit_input_normalizer(self, indices: Iterable[int]) -> Standardizer:
        """Fit the z standardiser on the given (training) indices only."""
        idx = torch.as_tensor(list(indices), dtype=torch.long)
        self._z_norm = Standardizer.fit(self.z_features[idx], dim=(0,))
        return self._z_norm

    def set_input_normalizer(self, norm: Standardizer | None) -> None:
        self._z_norm = norm

    # -- Dataset ---------------------------------------------------------- #

    def __len__(self) -> int:
        return self.targets.shape[0]

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.z_features[i]
        if self._z_norm is not None:
            z = self._z_norm.encode(z.unsqueeze(0)).squeeze(0)

        planes = [z[:, None, None].expand(-1, self.n_omega, self.n_omega_prime)]

        if self.spec.family_channels:
            onehot = torch.zeros(len(self.spec.families), dtype=self.spec.dtype)
            onehot[self.family_index[i]] = 1.0
            planes.append(onehot[:, None, None].expand(-1, self.n_omega, self.n_omega_prime))

        if self._coord_planes is not None:
            planes.append(self._coord_planes)

        return torch.cat(planes, dim=0), self.targets[i]


def make_splits(dataset: MirrorDataset, config: SplitConfig) -> tuple[list[int], list[int]]:
    """Return ``(train_indices, val_indices)``.

    With ``stratify_by_family`` the split is applied within each family, so the
    validation set holds the same family proportions as the full dataset.
    """
    rng = np.random.default_rng(config.seed)
    families = np.asarray(dataset.family_index)

    groups = [np.arange(len(dataset))] if not config.stratify_by_family else [
        np.flatnonzero(families == k) for k in range(len(dataset.spec.families))
    ]

    train_idx: list[int] = []
    val_idx: list[int] = []
    for group in groups:
        if group.size == 0:
            continue
        shuffled = rng.permutation(group)
        n_val = int(round(config.val_fraction * shuffled.size))
        n_val = min(max(n_val, 1), shuffled.size - 1)  # never empty, never everything
        val_idx.extend(shuffled[:n_val].tolist())
        train_idx.extend(shuffled[n_val:].tolist())

    if set(train_idx) & set(val_idx):
        raise AssertionError("training and validation splits overlap")
    return sorted(train_idx), sorted(val_idx)


def make_loaders(
    dataset: MirrorDataset,
    train_idx: list[int],
    val_idx: list[int],
    batch_size: int,
    num_workers: int = 0,
    seed: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """DataLoaders over the two splits, with a seeded shuffle for training."""
    generator = torch.Generator().manual_seed(seed)
    train = DataLoader(
        torch.utils.data.Subset(dataset, train_idx),
        batch_size=batch_size, shuffle=True, num_workers=num_workers, generator=generator,
    )
    val = DataLoader(
        torch.utils.data.Subset(dataset, val_idx),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )
    return train, val


def _unit(x: np.ndarray) -> torch.Tensor:
    lo, hi = float(x[0]), float(x[-1])
    return torch.from_numpy((np.asarray(x, dtype=np.float64) - lo) / (hi - lo))
