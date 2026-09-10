"""Train an FNO to map a mirror trajectory z(t) to its Bogoliubov coefficient.

    python train.py                       # manuscript defaults: 20 epochs
    python train.py --epochs 5 --limit 40 # quick smoke run
    python train.py --help                # every knob

The configuration is assembled from the dataclasses in ``model_config.py``,
``scheduler_config.py`` and ``train_config.py``; command-line flags override
their defaults, and the resolved configuration is written next to the
checkpoint so a run can be reproduced.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from build import build_model, count_parameters, resolve_device
from dataset import DEFAULT_FILES_ROOT, MirrorDataset, make_loaders, make_splits
from losses import elementwise_mse, evaluate, relative_l2, zero_baseline_mse
from normalization import Standardizer
from train_config import TrainConfig


def train(config: TrainConfig, files_root: Path, limit: int | None = None) -> dict:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = resolve_device(config.device)

    # -- data ---------------------------------------------------------- #
    dataset = MirrorDataset(files_root=files_root, spec=config.spec, limit_per_family=limit)
    train_idx, val_idx = make_splits(dataset, config.split)

    if config.normalize_inputs:
        # fitted on the training split only, so validation stays untouched
        dataset.fit_input_normalizer(train_idx)

    target_norm: Standardizer | None = None
    if config.normalize_targets:
        target_norm = Standardizer.fit(dataset.targets[train_idx], dim=(0, 2, 3))

    train_loader, val_loader = make_loaders(
        dataset, train_idx, val_idx, config.batch_size, config.num_workers, config.seed
    )

    print(f"dataset: {len(dataset)} samples "
          f"({len(train_idx)} train / {len(val_idx)} val), "
          f"grid {dataset.n_omega}x{dataset.n_omega_prime}")
    print(f"input channels: {config.spec.in_channels}  (z {config.spec.n_z_features}"
          f" + family {len(config.spec.families) if config.spec.family_channels else 0}"
          f" + coords {2 if config.spec.coord_channels else 0})")

    # -- model --------------------------------------------------------- #
    model = build_model(config.model, config.spec, dataset.n_omega, dataset.n_omega_prime).to(device)
    optimizer = config.optimizer.build(model.parameters())
    scheduler = config.decay.build(optimizer)

    print(f"model: FNO n_modes={config.model.n_modes} hidden={config.model.hidden_channels} "
          f"layers={config.model.n_layers} -> {count_parameters(model):,} parameters")
    print(f"device: {device} | lr {config.optimizer.lr:g} decaying by "
          f"gamma={config.decay.gamma} per epoch "
          f"(final {config.decay.lr_at(config.epochs - 1, config.optimizer.lr):.3g})\n")

    # A model predicting zero everywhere scores this MSE; the run must beat it.
    baseline = float(zero_baseline_mse(dataset.targets[val_idx]))

    history: list[dict] = []
    best_val = float("inf")
    best_epoch = -1
    out_dir = config.out_dir / config.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(config.epochs):
        model.train()
        epoch_start = time.time()
        running_mse = running_rel = 0.0
        n_batches = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            if target_norm is not None:
                y = target_norm.encode(y)

            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = elementwise_mse(pred, y)
            loss.backward()
            if config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            running_mse += float(loss.detach())
            running_rel += float(relative_l2(pred.detach(), y))
            n_batches += 1

        if config.decay.step_every_epoch:
            scheduler.step()
            config.decay.clamp(optimizer)

        val = evaluate(model, val_loader, device)
        record = {
            "epoch": epoch + 1,
            "train_mse": running_mse / n_batches,
            "train_rel_l2": running_rel / n_batches,
            "val_mse": val["mse"],
            "val_rel_l2": val["rel_l2"],
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - epoch_start,
        }
        history.append(record)

        if (epoch + 1) % config.log_every == 0:
            print(f"epoch {record['epoch']:3d}/{config.epochs} | "
                  f"train MSE {record['train_mse']:.4e} | val MSE {record['val_mse']:.4e} | "
                  f"val rel-L2 {record['val_rel_l2']:.4f} | "
                  f"lr {record['lr']:.2e} | {record['seconds']:.1f}s")

        if record["val_mse"] < best_val:
            best_val, best_epoch = record["val_mse"], epoch + 1
            if config.save_checkpoint:
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state": model.state_dict(),
                        "config": config.to_dict(),
                        "input_norm": dataset._z_norm.state_dict() if dataset._z_norm else None,
                        "target_norm": target_norm.state_dict() if target_norm else None,
                    },
                    out_dir / "best.pt",
                )

    summary = {
        "best_val_mse": best_val,
        "best_epoch": best_epoch,
        "final_train_mse": history[-1]["train_mse"],
        "final_val_mse": history[-1]["val_mse"],
        "final_val_rel_l2": history[-1]["val_rel_l2"],
        "zero_baseline_val_mse": baseline,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_parameters": count_parameters(model),
    }

    (out_dir / "history.json").write_text(
        json.dumps({"config": config.to_dict(), "history": history, "summary": summary}, indent=2) + "\n"
    )

    print(f"\nbest val MSE {best_val:.4e} at epoch {best_epoch}")
    ratio = baseline / best_val if best_val > 0 else float("inf")
    print(f"zero-prediction baseline val MSE {baseline:.4e} "
          f"-- the model is {ratio:.1f}x better than predicting zero")
    print(f"final val relative L2 {summary['final_val_rel_l2']:.4f}")
    print(f"written to {out_dir}")
    return summary


def build_argparser() -> argparse.ArgumentParser:
    defaults = TrainConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=defaults.epochs)
    p.add_argument("--batch-size", type=int, default=defaults.batch_size)
    p.add_argument("--lr", type=float, default=defaults.optimizer.lr)
    p.add_argument("--weight-decay", type=float, default=defaults.optimizer.weight_decay)
    p.add_argument("--gamma", type=float, default=defaults.decay.gamma,
                   help="exponential LR decay factor per epoch")
    p.add_argument("--min-lr", type=float, default=defaults.decay.min_lr)
    p.add_argument("--modes", type=int, nargs=2, default=list(defaults.model.n_modes),
                   metavar=("M_OMEGA", "M_OMEGA_PRIME"))
    p.add_argument("--hidden-channels", type=int, default=defaults.model.hidden_channels)
    p.add_argument("--layers", type=int, default=defaults.model.n_layers)
    p.add_argument("--n-z-features", type=int, default=defaults.spec.n_z_features,
                   help="how many points of z(t) become input channels")
    p.add_argument("--val-fraction", type=float, default=defaults.split.val_fraction)
    p.add_argument("--seed", type=int, default=defaults.seed)
    p.add_argument("--device", default=defaults.device, help="auto, cpu or cuda")
    p.add_argument("--num-workers", type=int, default=defaults.num_workers)
    p.add_argument("--no-input-norm", action="store_true", help="skip z standardisation")
    p.add_argument("--normalize-targets", action="store_true",
                   help="standardise (Re beta, Im beta); MSE then reports in normalised units")
    p.add_argument("--files-root", type=Path, default=DEFAULT_FILES_ROOT)
    p.add_argument("--limit", type=int, default=None, help="samples per family, for smoke runs")
    p.add_argument("--run-name", default=defaults.run_name)
    p.add_argument("--out-dir", type=Path, default=defaults.out_dir)
    return p


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    config = TrainConfig()
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.seed = args.seed
    config.device = args.device
    config.num_workers = args.num_workers
    config.normalize_inputs = not args.no_input_norm
    config.normalize_targets = args.normalize_targets
    config.run_name = args.run_name
    config.out_dir = args.out_dir

    config.optimizer.lr = args.lr
    config.optimizer.weight_decay = args.weight_decay
    config.decay.gamma = args.gamma
    config.decay.min_lr = args.min_lr
    config.decay.__post_init__()

    config.model.n_modes = tuple(args.modes)
    config.model.hidden_channels = args.hidden_channels
    config.model.n_layers = args.layers
    config.model.__post_init__()

    config.spec.n_z_features = args.n_z_features
    config.spec.__post_init__()

    config.split.val_fraction = args.val_fraction
    config.split.seed = args.seed
    config.split.__post_init__()
    return config


def main() -> int:
    args = build_argparser().parse_args()
    train(config_from_args(args), files_root=args.files_root, limit=args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
