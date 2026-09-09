"""Generate trajectory / beta samples for the Lambert-W "speed of light" mirror.

    M. R. R. Good, "Reflecting at the Speed of Light", arXiv:1612.02459.

        z(t)  = -(s/kappa) W(e^{kappa t})                          [eq. 2]
        beta  = (1/kappa)(-i/pi) s^{i p} sqrt(a b) (i sigma)^{i p - 1} Gamma(-i p)
                with a = w/kappa, b = w'/kappa, p = a + b,         [eq. 11]
                sigma = (1/s + 1) a + (1/s - 1) b

Sampled parameters: the final drift speed ``s`` in (0, 1] and the acceleration
``kappa`` > 0.

Grid defaults follow the Experiments section of ``moving_mirror.tex``:
128 time samples, a 30 x 30 (omega, omega') grid on [0.01, 2].

The time grid is *absolute* and shared by every sample -- deliberately not
rescaled by kappa.  On a kappa-scaled grid z(t) would collapse to
``(s/kappa) * (one fixed shape)``, so s and kappa would be degenerate in the
input and no operator could separate them.  On a fixed grid kappa sets where
the knee falls and s/kappa its depth, so both are recoverable.

Run:  python gen_good2016.py [--n-samples 500] [--overwrite]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import mirror_physics as mp
import sample_io

FAMILY = "good2016_lambertw"
PAPER = "arXiv:1612.02459 -- M. R. R. Good, 'Reflecting at the Speed of Light'"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).parent.parent / "files" / FAMILY)
    ap.add_argument("--n-samples", type=int, default=500)
    ap.add_argument("--seed", type=int, default=2016)
    ap.add_argument("--overwrite", action="store_true", help="delete existing samples first")

    g = ap.add_argument_group("grids (defaults from the Experiments section)")
    g.add_argument("--n-time", type=int, default=128)
    g.add_argument("--t-min", type=float, default=-6.0, help="absolute time, not kappa-scaled")
    g.add_argument("--t-max", type=float, default=12.0)
    g.add_argument("--n-omega", type=int, default=30)
    g.add_argument("--n-omega-prime", type=int, default=30)
    g.add_argument("--omega-min", type=float, default=0.01, help="0 excluded: beta diverges there")
    g.add_argument("--omega-max", type=float, default=2.0)

    p = ap.add_argument_group("parameter sampling")
    p.add_argument("--s-min", type=float, default=0.1, help="drift speed, 0 < s <= 1")
    p.add_argument("--s-max", type=float, default=1.0)
    p.add_argument("--kappa-min", type=float, default=0.5, help="log-uniform in [min, max]")
    p.add_argument("--kappa-max", type=float, default=1.5)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    if not (0.0 < args.s_min <= args.s_max <= 1.0):
        raise SystemExit("require 0 < s-min <= s-max <= 1")
    if not (0.0 < args.kappa_min <= args.kappa_max):
        raise SystemExit("require 0 < kappa-min <= kappa-max")
    if args.omega_min <= 0.0:
        raise SystemExit("omega-min must be > 0; beta diverges at zero frequency")

    out_dir = Path(args.out_dir)
    sample_io.prepare_out_dir(out_dir, args.overwrite)

    t = np.linspace(args.t_min, args.t_max, args.n_time)
    omega = np.linspace(args.omega_min, args.omega_max, args.n_omega)
    omega_prime = np.linspace(args.omega_min, args.omega_max, args.n_omega_prime)

    rng = np.random.default_rng(args.seed)
    s_vals = rng.uniform(args.s_min, args.s_max, args.n_samples)
    kappa_vals = np.exp(rng.uniform(np.log(args.kappa_min), np.log(args.kappa_max), args.n_samples))

    meta = {
        "trajectory": "z(t) = -(s/kappa) * W(exp(kappa t))   [eq. 2]",
        "beta": "(1/kappa)(-i/pi) s^{i p} sqrt(a b) (i sigma)^{i p - 1} Gamma(-i p)   [eq. 11]",
        "beta_note": "a = omega/kappa, b = omega'/kappa, p = a + b, "
                     "sigma = (1/s + 1) a + (1/s - 1) b; kappa restored via "
                     "beta_kappa(w, w') = beta_1(w/kappa, w'/kappa) / kappa",
        "units": "hbar = c = 1; frequencies in the same units as kappa",
        "t_grid": [args.t_min, args.t_max, args.n_time],
        "omega_grid": [args.omega_min, args.omega_max, args.n_omega],
        "omega_prime_grid": [args.omega_min, args.omega_max, args.n_omega_prime],
    }

    entries = []
    for i, (s, kappa) in enumerate(zip(s_vals, kappa_vals)):
        params = {"s": float(s), "kappa": float(kappa)}
        z = mp.good2016_trajectory(t, s, kappa)
        beta = mp.good2016_beta(omega[:, None], omega_prime[None, :], s, kappa)
        name = sample_io.save_sample(
            out_dir, i, FAMILY, PAPER, params, t, z, omega, omega_prime, beta, meta
        )
        entries.append({"index": i, "file": name, "params": params})
        if (i + 1) % 50 == 0 or i + 1 == args.n_samples:
            print(f"  {i + 1}/{args.n_samples} samples written")

    config = {"family": FAMILY, "paper": PAPER, "seed": args.seed, **meta,
              "s_range": [args.s_min, args.s_max],
              "kappa_range_log_uniform": [args.kappa_min, args.kappa_max]}
    sample_io.write_manifest(out_dir, config, entries)
    print(f"Wrote {len(entries)} samples + manifest.json to {out_dir}")


if __name__ == "__main__":
    main()
