"""Validate the implementations in ``mirror_physics`` against the papers.

Every check reproduces a result the source papers state independently of the
formula being tested, so a pass is evidence the transcription (and the kappa
restoration in ``good2016_beta``) is right.

Run:  python verify.py [--quick] [--data]

``--data`` additionally validates whatever samples are already on disk under
``../files``: shared time grid, finite and monotonic trajectories, and stored
beta re-derived from an independent route.
"""

from __future__ import annotations

import argparse
import sys

import json
from pathlib import Path

import numpy as np
from scipy.special import digamma, gamma, lambertw

import mirror_physics as mp
import sample_io

_FAILURES: list[str] = []


def check(name: str, err: float, tol: float, detail: str = "") -> None:
    ok = np.isfinite(err) and err <= tol
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}: err = {err:.3e} (tol {tol:.1e}) {detail}")
    if not ok:
        _FAILURES.append(name)


# --------------------------------------------------------------------------- #

def check_lambert_w() -> None:
    print("\nLambert W  (W(e^a), used by good2016_trajectory)")
    a = np.linspace(-30.0, 600.0, 4001)
    ours = mp.lambert_w_exp(a)

    # scipy can only be used where exp(a) is representable
    small = a[a < 700.0][a[a < 700.0] > -700.0]
    ref = lambertw(np.exp(small)).real
    rel = np.max(np.abs(mp.lambert_w_exp(small) - ref) / ref)
    check("agrees with scipy.special.lambertw", rel, 1e-12)

    # defining relation ln w + w = a holds over the whole range, incl. a > 709
    resid = np.max(np.abs(np.log(ours) + ours - a) / np.maximum(np.abs(a), 1.0))
    check("satisfies ln W + W = a up to a = 600", resid, 1e-13, "(scipy overflows here)")


def check_good2016_trajectory() -> None:
    print("\nGood 2016 trajectory  x(t) = -(s/kappa) W(e^{kappa t})   [eq. 2]")
    # dx/dt = -s W/(1+W), so |v|/s approaches 1 only as 1/W ~ 1/(kappa t):
    # at kappa t = 5000 the residual is 2.0e-4, which sets the tolerance.
    for s, kappa in [(0.3, 1.0), (0.9, 0.7), (1.0, 2.0)]:
        t = np.linspace(5000.0 / kappa, 5000.0 / kappa + 1e-3, 3)
        x = mp.good2016_trajectory(t, s, kappa)
        v = np.gradient(x, t)[1]
        check(f"drift speed -> s  (s={s}, kappa={kappa})", abs(abs(v) - s) / s, 1e-3)

    # starts asymptotically static at x = 0
    x0 = mp.good2016_trajectory(np.array([-40.0]), 0.5, 1.0)[0]
    check("asymptotically static at x = 0", abs(x0), 1e-16)


def check_good2016_beta() -> None:
    print("\nGood 2016 beta  [eq. 11] vs published |beta|^2")
    w = np.linspace(0.05, 3.0, 37)[:, None]
    wp = np.linspace(0.05, 3.0, 41)[None, :]

    for s, kappa in [(1.0, 1.0), (0.6, 1.0), (0.35, 1.7), (0.9, 0.4)]:
        b2 = np.abs(mp.good2016_beta(w, wp, s, kappa)) ** 2
        ref = mp.good2016_beta_sq_reference(w, wp, s, kappa)
        rel = np.max(np.abs(b2 - ref) / ref)
        check(f"|beta|^2 matches eq. 12  (s={s}, kappa={kappa})", rel, 1e-11)

    # eq. (13): the s -> 1 limit
    kappa = 1.3
    b2 = np.abs(mp.good2016_beta(w, wp, 1.0, kappa)) ** 2
    eq13 = wp / (2.0 * np.pi * kappa * w * (w + wp) * np.expm1(2.0 * np.pi * (w + wp) / kappa))
    check("|beta|^2 matches eq. 13 at s = 1", np.max(np.abs(b2 - eq13) / eq13), 1e-11)

    # eq. (16): high-frequency limit w' >> w at s = 1.  Dropping w from the
    # Planck factor also needs w << kappa/(2 pi); at w = 0.002, kappa = 1 that
    # approximation is itself only good to 1.3%, which sets the tolerance.
    kappa = 1.0
    w_hi = np.array([0.0005, 0.001, 0.002])[:, None]
    wp_hi = np.array([40.0, 80.0])[None, :]
    b2 = np.abs(mp.good2016_beta(w_hi, wp_hi, 1.0, kappa)) ** 2
    eq16 = 1.0 / (2.0 * np.pi * kappa * w_hi) / np.expm1(2.0 * np.pi * wp_hi / kappa)
    check("|beta|^2 -> eq. 16 for w' >> w", np.max(np.abs(b2 / eq16 - 1.0)), 2e-2)


def check_besselk() -> None:
    # The oscillatory quadrature cancels to a ~1e-12 absolute floor that extra
    # nodes do not lower, so it is held to an absolute tolerance.  Its relative
    # error is only large where K itself is ~1e-6 (large nu, z ~ 1), and there
    # beta is negligible anyway.  mpmath is the default backend precisely
    # because it is both more accurate and, on a 30x30 grid, faster.
    print("\nK_{i nu}(z) of imaginary order  (quad backend vs mpmath reference)")
    rng = np.random.default_rng(0)
    nu = rng.uniform(0.005, 4.0, 60)
    for label, z in [
        ("z ~ 1e-6 (g >> kappa regime)", rng.uniform(5e-7, 5e-6, 60)),
        ("z ~ 1e-2", rng.uniform(5e-3, 5e-2, 60)),
        ("z ~ 1", rng.uniform(0.5, 3.0, 60)),
    ]:
        q = mp.besselk_imag_order(nu, z, backend="quad")
        m = mp.besselk_imag_order(nu, z, backend="mpmath")
        check(f"quad == mpmath, {label}", np.max(np.abs(q - m)), 1e-11)


def check_glw2021_trajectory() -> None:
    print("\nGLW 2021 trajectory  g v = -sinh(2 kappa x)   [eq. 1]")
    for g, kappa in [(1e6, 1.0), (1e3, 0.5), (1e2, 2.0)]:
        t = np.linspace(-8.0, 8.0, 121)
        x = mp.glw2021_trajectory(t, g, kappa)
        resid = np.max(np.abs(mp.glw2021_time_of_position(x, g, kappa) - t))
        check(f"inversion round-trips  (g={g:g}, kappa={kappa})", resid, 1e-10)

        x0 = mp.glw2021_trajectory(np.array([0.0]), g, kappa)[0]
        check(f"passes through x(0) = 0  (g={g:g}, kappa={kappa})", abs(x0), 1e-12)

        # analytic peak speed |dx/dt| = 1 / (1 + 2 kappa/g), attained at x = 0
        tt = np.linspace(-1e-4, 1e-4, 5)
        v = np.gradient(mp.glw2021_trajectory(tt, g, kappa), tt)[2]
        check(
            f"peak speed = 1/(1+2kappa/g)  (g={g:g}, kappa={kappa})",
            abs(abs(v) - 1.0 / (1.0 + 2.0 * kappa / g)) * (1.0 + 2.0 * kappa / g),
            1e-6,
        )


def _glw_gamma_bar(omega: np.ndarray, g: float) -> np.ndarray:
    """Paper eq. (6)/(5) at kappa = 1: Gamma-bar_omega = A/pi + B/4."""
    # H_{i w} = psi(1 + i w) + gamma  =>  A = ln(2g/w) + Re psi(1 + i w) - 1
    A = np.log(2.0 * g / omega) + np.real(digamma(1.0 + 1j * omega)) - 1.0
    csch = 1.0 / np.sinh(np.pi * omega)
    term = (omega / (2.0 * g)) ** (2j * omega) * csch / ((2.0 * omega + 1j) * gamma(1.0 + 1j * omega) ** 2)
    B = 2.0 * np.real(term)  # the second term of eq. (6) is the conjugate of the first
    return A / np.pi + B / 4.0


def check_glw2021_spectrum(quick: bool) -> None:
    print("\nGLW 2021 spectrum  N_omega = int |beta|^2 d omega'  vs eq. (5), fig. 1")
    g, kappa = 1e6, 1.0  # the paper's fig. 1 uses g/kappa = 1e6
    omegas = np.array([0.1, 0.2, 0.3]) if quick else np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5])

    # |beta|^2 falls off only as 1/omega' until K cuts it off near omega' ~ g,
    # so integrate on a log grid spanning both ends.
    n = 4000 if quick else 12000
    u = np.linspace(np.log(1e-9), np.log(1e3 * g), n)
    wp = np.exp(u)

    numeric = np.empty_like(omegas)
    for i, w in enumerate(omegas):
        integrand = np.empty_like(wp)
        for lo in range(0, wp.size, 1000):  # chunked: the quad backend is (n, 2000)
            sl = slice(lo, lo + 1000)
            b = mp.glw2021_beta(np.full(wp[sl].shape, w), wp[sl], g, kappa)
            integrand[sl] = np.abs(b) ** 2 * wp[sl]  # * omega' for the d(ln omega') measure
        numeric[i] = np.trapezoid(integrand, u)

    analytic = _glw_gamma_bar(omegas, g) / np.expm1(2.0 * np.pi * omegas)  # T = 1/(2 pi)
    for w, num, ana in zip(omegas, numeric, analytic):
        check(f"N_omega at omega={w:.2f}  (num {num:.4f} vs eq.5 {ana:.4f})", abs(num / ana - 1.0), 2e-2)


def check_generated_data(files_root: Path) -> None:
    """Validate the samples already written under ``files_root``."""
    print(f"\nGenerated data under {files_root}")
    if not files_root.exists():
        print("  (no files directory yet -- run the generators first)")
        return

    for family_dir in sorted(d for d in files_root.iterdir() if d.is_dir()):
        fam = family_dir.name
        paths = sorted(family_dir.glob("sample_*.pt"))
        manifest_path = family_dir / "manifest.json"
        if not paths:
            print(f"  ({fam}: no samples)")
            continue

        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        check(f"{fam}: manifest count matches files", abs(manifest.get("n_samples", -1) - len(paths)), 0,
              f"({len(paths)} files)")

        t_ref = None
        n_bad_finite = n_bad_grid = n_bad_mono = 0
        for path in paths:
            s = sample_io.load_sample(path)
            t, z, beta = s["t"].numpy(), s["z"].numpy(), s["beta"].numpy()
            if not (np.all(np.isfinite(z)) and np.all(np.isfinite(beta))):
                n_bad_finite += 1
            if t_ref is None:
                t_ref = t
            elif not np.allclose(t, t_ref):
                n_bad_grid += 1
            if not np.all(np.diff(z) < 0):  # both mirrors travel monotonically leftward
                n_bad_mono += 1

        check(f"{fam}: all trajectories and beta finite", n_bad_finite, 0)
        check(f"{fam}: all samples share one time grid", n_bad_grid, 0)
        check(f"{fam}: trajectories monotonically leftward", n_bad_mono, 0)

        # re-derive one sample's |beta|^2 by a route the generator did not use
        s = sample_io.load_sample(paths[len(paths) // 2])
        prm, w = s["params"], s["omega"].numpy()[:, None]
        wp = s["omega_prime"].numpy()[None, :]
        if fam.startswith("good2016"):
            ref = mp.good2016_beta_sq_reference(w, wp, prm["s"], prm["kappa"])  # eq. 12 directly
            tol = 1e-11
        else:
            ref = np.abs(mp.glw2021_beta(w, wp, prm["g"], prm["kappa"], backend="quad")) ** 2
            tol = 1e-7  # quad backend cross-check, not the mpmath one used to generate
        got = np.abs(s["beta"].numpy()) ** 2
        check(f"{fam}: stored |beta|^2 reproduced independently",
              float(np.max(np.abs(got - ref) / ref)), tol)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="coarser spectrum integral")
    ap.add_argument("--data", action="store_true", help="also validate samples on disk")
    ap.add_argument("--files-root", type=Path, default=Path(__file__).parent.parent / "files")
    args = ap.parse_args()

    check_lambert_w()
    check_good2016_trajectory()
    check_good2016_beta()
    check_besselk()
    check_glw2021_trajectory()
    check_glw2021_spectrum(args.quick)
    if args.data:
        check_generated_data(args.files_root)

    print()
    if _FAILURES:
        print(f"{len(_FAILURES)} check(s) FAILED: " + ", ".join(_FAILURES))
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
