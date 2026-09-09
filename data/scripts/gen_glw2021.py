"""Generate trajectory / beta samples for the Casimir-light mirror.

    M. R. R. Good, E. V. Linder, F. Wilczek, "Finite Thermal Particle Creation
    of Casimir Light", arXiv:2108.11188.

        g v = -sinh(2 kappa z),  v = t + z   =>  t(z) = -sinh(2 kappa z)/g - z   [eq. 1]
        beta = -sqrt(w w')/(pi kappa w_p) e^{-pi w/(2 kappa)} K_{i w/kappa}(w_p/g),
               w_p = w + w'                                                      [eq. 3]

Sampled parameters: ``kappa`` > 0 and the dimensionful ``g``, drawn through the
ratio g/kappa.  The mirror is asymptotically static at both ends and thermal
for g >> kappa; the paper's fig. 1 uses g/kappa = 1e6.

There is no closed form for z(t), so t(z) -- which is strictly decreasing -- is
inverted numerically per grid point.

The time grid is absolute and shared by every sample so that g and kappa stay
distinguishable from the trajectory alone (see the note in gen_good2016.py).

The mirror leaves its near-null phase at t ~ ln(g/kappa)/(2 kappa), which spans
a factor of ~8 over the default parameter ranges.  ``--t-window auto`` (the
default) therefore sizes one shared half-width to cover the slowest sample
actually drawn, so the deceleration is inside the window for every sample; the
cost is coarser resolution for the fastest ones.  Narrow --g-over-kappa-max or
raise --n-time to buy that back.

Run:  python gen_glw2021.py [--n-samples 500] [--overwrite]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import mirror_physics as mp
import sample_io

FAMILY = "glw2021_casimir"
PAPER = "arXiv:2108.11188 -- Good, Linder & Wilczek, 'Finite Thermal Particle Creation of Casimir Light'"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).parent.parent / "files" / FAMILY)
    ap.add_argument("--n-samples", type=int, default=500)
    ap.add_argument("--seed", type=int, default=2021)
    ap.add_argument("--overwrite", action="store_true", help="delete existing samples first")
    ap.add_argument("--besselk-backend", choices=["mpmath", "quad"], default="mpmath",
                    help="mpmath is both more accurate and faster on a 30x30 grid")

    g = ap.add_argument_group("grids (defaults from the Experiments section)")
    g.add_argument("--n-time", type=int, default=128)
    g.add_argument("--t-window", choices=["auto", "fixed", "per-sample"], default="auto",
                   help="auto: one shared T covering the slowest sample drawn (default). "
                        "fixed: use --t-half-width for all. "
                        "per-sample: each sample gets its own T, which breaks the shared "
                        "domain an operator learner needs.")
    g.add_argument("--t-half-width", type=float, default=12.0, help="only for --t-window fixed")
    g.add_argument("--t-window-factor", type=float, default=1.5,
                   help="T = factor * ln(2 g/kappa)/(2 kappa) for auto and per-sample")
    g.add_argument("--n-omega", type=int, default=30)
    g.add_argument("--n-omega-prime", type=int, default=30)
    g.add_argument("--omega-min", type=float, default=0.01, help="0 excluded: beta diverges there")
    g.add_argument("--omega-max", type=float, default=2.0)

    p = ap.add_argument_group("parameter sampling")
    p.add_argument("--g-over-kappa-min", type=float, default=1e2, help="log-uniform; g >> kappa is thermal")
    p.add_argument("--g-over-kappa-max", type=float, default=1e6)
    p.add_argument("--kappa-min", type=float, default=0.5, help="log-uniform in [min, max]")
    p.add_argument("--kappa-max", type=float, default=1.5)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    if not (0.0 < args.g_over_kappa_min <= args.g_over_kappa_max):
        raise SystemExit("require 0 < g-over-kappa-min <= g-over-kappa-max")
    if not (0.0 < args.kappa_min <= args.kappa_max):
        raise SystemExit("require 0 < kappa-min <= kappa-max")
    if args.omega_min <= 0.0:
        raise SystemExit("omega-min must be > 0; beta diverges at zero frequency")

    out_dir = Path(args.out_dir)
    sample_io.prepare_out_dir(out_dir, args.overwrite)

    omega = np.linspace(args.omega_min, args.omega_max, args.n_omega)
    omega_prime = np.linspace(args.omega_min, args.omega_max, args.n_omega_prime)

    rng = np.random.default_rng(args.seed)
    kappa_vals = np.exp(rng.uniform(np.log(args.kappa_min), np.log(args.kappa_max), args.n_samples))
    ratio_vals = np.exp(
        rng.uniform(np.log(args.g_over_kappa_min), np.log(args.g_over_kappa_max), args.n_samples)
    )
    g_vals = ratio_vals * kappa_vals

    meta = {
        "trajectory": "g (t + z) = -sinh(2 kappa z), inverted numerically for z(t)   [eq. 1]",
        "beta": "-sqrt(w w')/(pi kappa w_p) exp(-pi w/(2 kappa)) K_{i w/kappa}(w_p/g)   [eq. 3]",
        "beta_note": "w_p = w + w'; beta is real-valued here (K of imaginary order is "
                     "real for real argument) but is stored complex128 so both families "
                     "share one dtype. The section 4 table's extra factor s is not in eq. 3.",
        "units": "hbar = c = 1; frequencies in the same units as kappa",
        "besselk_backend": args.besselk_backend,
        "t_window": args.t_window,
        "omega_grid": [args.omega_min, args.omega_max, args.n_omega],
        "omega_prime_grid": [args.omega_min, args.omega_max, args.n_omega_prime],
        "n_time": args.n_time,
    }

    # Half-width that puts the near-null -> static transition inside the window.
    needed = args.t_window_factor * np.log(2.0 * g_vals / kappa_vals) / (2.0 * kappa_vals)
    if args.t_window == "per-sample":
        half_widths = needed
        print("warning: --t-window per-sample gives each sample its own t grid; "
              "samples no longer share a domain.")
    else:
        shared = float(needed.max()) if args.t_window == "auto" else args.t_half_width
        half_widths = np.full(args.n_samples, shared)
        dt = 2.0 * shared / (args.n_time - 1)
        print(f"shared time window: t in [{-shared:.2f}, {shared:.2f}], dt = {dt:.3f}")
        short = float(needed.min())
        print(f"  fastest sample needs T = {short:.2f}, i.e. its transition spans "
              f"~{2 * short / dt:.0f} of {args.n_time} points")
        if shared < needed.max():
            print(f"  warning: T = {shared:.2f} truncates the slowest sample "
                  f"(needs {needed.max():.2f}); its deceleration will be cut off.")

    entries = []
    for i, (kappa, g, half) in enumerate(zip(kappa_vals, g_vals, half_widths)):
        params = {"g": float(g), "kappa": float(kappa)}
        half = float(half)
        t = np.linspace(-half, half, args.n_time)

        z = mp.glw2021_trajectory(t, g, kappa)
        beta = mp.glw2021_beta(
            omega[:, None], omega_prime[None, :], g, kappa, backend=args.besselk_backend
        )
        sample_meta = dict(meta, t_grid=[float(-half), float(half), args.n_time])
        name = sample_io.save_sample(
            out_dir, i, FAMILY, PAPER, params, t, z, omega, omega_prime, beta, sample_meta
        )
        entries.append({"index": i, "file": name, "params": params,
                        "g_over_kappa": float(g / kappa)})
        if (i + 1) % 50 == 0 or i + 1 == args.n_samples:
            print(f"  {i + 1}/{args.n_samples} samples written")

    config = {"family": FAMILY, "paper": PAPER, "seed": args.seed, **meta,
              "g_over_kappa_range_log_uniform": [args.g_over_kappa_min, args.g_over_kappa_max],
              "kappa_range_log_uniform": [args.kappa_min, args.kappa_max]}
    sample_io.write_manifest(out_dir, config, entries)
    print(f"Wrote {len(entries)} samples + manifest.json to {out_dir}")


if __name__ == "__main__":
    main()
