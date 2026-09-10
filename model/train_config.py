"""Training-loop configuration: the run itself, not the architecture.

Defaults follow the manuscript: 20 epochs, an 80:20 train/validation split and
element-wise MSE as the objective.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from dataset import SplitConfig
from fno_tensors import TensorSpec
from model_config import FNOConfig
from scheduler_config import OptimizerConfig, WarmupDecayConfig


@dataclass
class TrainConfig:
    """Everything one training run needs."""

    epochs: int = 500
    batch_size: int = 16
    device: str = "auto"
    seed: int = 0
    num_workers: int = 0

    normalize_inputs: bool = True
    normalize_targets: bool = False   # off: MSE stays in physical beta units
    grad_clip: float | None = 1.0

    out_dir: Path = Path(__file__).resolve().parent.parent / "outputs" / "runs"
    run_name: str = "fno_beta"
    save_checkpoint: bool = True
    log_every: int = 1

    split: SplitConfig = field(default_factory=SplitConfig)
    spec: TensorSpec = field(default_factory=TensorSpec)
    model: FNOConfig = field(default_factory=FNOConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    decay: WarmupDecayConfig = field(default_factory=WarmupDecayConfig)

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        self.out_dir = Path(self.out_dir)

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serialisable record of the full configuration."""
        d = asdict(self)
        d["out_dir"] = str(self.out_dir)
        d["spec"] = self.spec.to_dict()
        d["model"] = self.model.to_dict()
        return d


DEFAULT_TRAIN_CONFIG = TrainConfig()
