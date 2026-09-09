"""Analytic moving-mirror trajectories and Bogoliubov beta coefficients.

Implements the two exactly-solvable mirrors listed in the literature survey
(section 4 of ``moving_mirror.tex``):

1. ``good2016`` -- M. R. R. Good, "Reflecting at the Speed of Light",
   arXiv:1612.02459.  Trajectory  x(t) = -(s/kappa) W(e^{kappa t}),  where W is
   the Lambert W function.  The mirror starts asymptotically static and coasts
   to a final drift speed 0 < s <= 1 without forming an acceleration horizon.

2. ``glw2021`` -- M. R. R. Good, E. V. Linder, F. Wilczek, "Finite Thermal
   Particle Creation of Casimir Light", arXiv:2108.11188.  Trajectory
   g v = -sinh(2 kappa x)  with v = t + x.  The mirror is asymptotically static
   on both ends and is thermal for g >> kappa.

Conventions
-----------
Both papers quote beta with kappa scaled out (kappa = 1).  For good2016 the
integral representation (its eq. 10) obeys the exact scaling

    beta_kappa(w, w') = (1/kappa) * beta_1(w/kappa, w'/kappa),

which is what restores kappa below; ``verify.py`` checks the resulting
|beta|^2 against the published eqs. (12), (13) and (16).  The glw2021 formula
(its eq. 3) already carries kappa explicitly.

All frequencies are in the same units as kappa; hbar = c = 1.
"""

from __future__ import annotations

import numpy as np
from scipy.special import loggamma
from scipy.optimize import brentq

__all__ = [
    "lambert_w_exp",
    "good2016_trajectory",
    "good2016_beta",
    "good2016_beta_sq_reference",
    "glw2021_time_of_position",
    "glw2021_trajectory",
    "glw2021_beta",
    "besselk_imag_order",
]


# --------------------------------------------------------------------------- #
# Paper 1: Good 2016 (arXiv:1612.02459) -- Lambert-W "speed of light" mirror
# --------------------------------------------------------------------------- #

def lambert_w_exp(a, tol: float = 1e-14, max_iter: int = 100):
    """Return ``W(exp(a))`` for real ``a``, without ever forming ``exp(a)``.

    ``exp(a)`` overflows for a >~ 709, so the defining relation is solved in
    log space instead.  With ``w = W(e^a)`` we have ``w e^w = e^a``, i.e.
    ``ln w + w = a``; substituting ``u = ln w`` gives

        g(u) = e^u + u - a = 0.

    ``g`` is smooth, strictly increasing and convex, so Newton's method
    converges monotonically from any point where ``g >= 0``.  ``u0 = ln a``
    (for a > 1) and ``u0 = a`` (otherwise) both satisfy that.
    """
    a = np.asarray(a, dtype=np.float64)
    u = np.where(a > 1.0, np.log(np.maximum(a, 1.0)), a)
    for _ in range(max_iter):
        eu = np.exp(u)
        step = (eu + u - a) / (eu + 1.0)
        u = u - step
        if np.max(np.abs(step)) < tol:
            break
    return np.exp(u)


def good2016_trajectory(t, s: float, kappa: float):
    """Mirror worldline ``x(t) = -(s/kappa) W(e^{kappa t})``  (paper eq. 2)."""
    return -(s / kappa) * lambert_w_exp(kappa * np.asarray(t, dtype=np.float64))


def good2016_beta(omega, omega_prime, s: float, kappa: float):
    """Bogoliubov ``beta_{omega omega'}`` for the Lambert-W mirror (eq. 11).

    With ``a = omega/kappa``, ``b = omega'/kappa``, ``p = a + b`` and
    ``sigma = (1/s + 1) a + (1/s - 1) b``:

        beta = (1/kappa) * (-i/pi) * s^{i p} * sqrt(a b)
               * (i sigma)^{i p - 1} * Gamma(-i p).

    Evaluated through its logarithm so the strong ``e^{-pi p}`` suppression at
    large frequency never underflows intermediate factors.

    Parameters
    ----------
    omega, omega_prime : array_like
        Frequency grids.  Broadcast against each other; passing shapes
        ``(n, 1)`` and ``(1, m)`` yields the ``(n, m)`` matrix with
        ``beta[i, j] = beta_{omega_i, omega'_j}``.
    s : float
        Final drift speed, ``0 < s <= 1``.
    kappa : float
        Acceleration parameter, ``kappa > 0``.

    Returns
    -------
    numpy.ndarray of complex128
    """
    if not (0.0 < s <= 1.0):
        raise ValueError(f"s must satisfy 0 < s <= 1, got {s}")
    if kappa <= 0.0:
        raise ValueError(f"kappa must be positive, got {kappa}")

    a = np.asarray(omega, dtype=np.float64) / kappa
    b = np.asarray(omega_prime, dtype=np.float64) / kappa
    if np.any(a <= 0) or np.any(b <= 0):
        raise ValueError("beta diverges at zero frequency; use omega, omega' > 0")

    p = a + b
    sigma = (1.0 / s + 1.0) * a + (1.0 / s - 1.0) * b

    # log(-i/pi) = -log(pi) - i pi/2 ;  log(i sigma) = log(sigma) + i pi/2
    log_beta = (
        -np.log(np.pi)
        - 0.5j * np.pi
        + 1j * p * np.log(s)
        + 0.5 * np.log(a * b)
        + (1j * p - 1.0) * (np.log(sigma) + 0.5j * np.pi)
        + loggamma(-1j * p)
        - np.log(kappa)
    )
    return np.exp(log_beta)


def good2016_beta_sq_reference(omega, omega_prime, s: float, kappa: float):
    """``|beta|^2`` straight from the paper's eq. (12), with kappa restored.

        |beta|^2 = 2 w w' / (pi kappa omega_s^2 omega_p (e^{2 pi omega_p/kappa} - 1))

    Used by ``verify.py`` as an independent check on :func:`good2016_beta`.
    """
    w = np.asarray(omega, dtype=np.float64)
    wp_ = np.asarray(omega_prime, dtype=np.float64)
    omega_p = w + wp_
    omega_s = (1.0 / s + 1.0) * w + (1.0 / s - 1.0) * wp_
    return (
        2.0 * w * wp_
        / (np.pi * kappa * omega_s**2 * omega_p)
        * (1.0 / np.expm1(2.0 * np.pi * omega_p / kappa))
    )


# --------------------------------------------------------------------------- #
# Paper 2: Good, Linder & Wilczek 2021 (arXiv:2108.11188) -- Casimir light
# --------------------------------------------------------------------------- #

def glw2021_time_of_position(x, g: float, kappa: float):
    """Invert of the worldline: ``t(x) = -sinh(2 kappa x)/g - x``  (eq. 1).

    Follows from ``g v = -sinh(2 kappa x)`` with ``v = t + x``.  ``t`` is
    strictly decreasing in ``x`` (``dt/dx = -(2 kappa/g) cosh(2 kappa x) - 1``),
    so the map is globally invertible.
    """
    x = np.asarray(x, dtype=np.float64)
    return -np.sinh(2.0 * kappa * x) / g - x


def glw2021_trajectory(t, g: float, kappa: float, tol: float = 1e-13):
    """Mirror worldline ``x(t)`` for ``g v = -sinh(2 kappa x)``.

    No closed form exists, so ``t(x)`` is inverted numerically.  Monotonicity
    makes bracketing safe: the bracket is grown geometrically until it spans
    the requested time, capped so ``sinh`` cannot overflow.
    """
    if g <= 0.0 or kappa <= 0.0:
        raise ValueError("g and kappa must be positive")

    t = np.atleast_1d(np.asarray(t, dtype=np.float64))
    x_cap = 300.0 / (2.0 * kappa)  # sinh(300) is the last finite double

    out = np.empty_like(t)
    for i, ti in enumerate(t):
        lo, hi = -1.0 / kappa, 1.0 / kappa
        while glw2021_time_of_position(lo, g, kappa) < ti and lo > -x_cap:
            lo = max(2.0 * lo, -x_cap)
        while glw2021_time_of_position(hi, g, kappa) > ti and hi < x_cap:
            hi = min(2.0 * hi, x_cap)
        if not (
            glw2021_time_of_position(lo, g, kappa) >= ti >= glw2021_time_of_position(hi, g, kappa)
        ):
            raise ValueError(f"t = {ti} lies outside the representable worldline")
        out[i] = brentq(
            lambda x: glw2021_time_of_position(x, g, kappa) - ti, lo, hi, xtol=tol
        )
    return out


def besselk_imag_order(nu, z, backend: str = "mpmath", n_nodes: int = 2000):
    """Modified Bessel ``K_{i nu}(z)`` of purely imaginary order, real ``z > 0``.

    ``K_{-v} = K_v`` and ``conj(K_v(z)) = K_{conj(v)}(z)`` for real ``z``, so
    ``K_{i nu}(z)`` is real-valued; the result is returned as a real array.

    backend
        ``"mpmath"`` -- ``mpmath.besselk`` element by element.  The default: on
                        a 30x30 grid it is both faster (~0.17 s vs ~0.39 s) and
                        ~1e-15 accurate.
        ``"quad"``   -- Gauss-Legendre on the integral representation
                        ``K_{i nu}(z) = int_0^inf e^{-z cosh(tau)} cos(nu tau) dtau``,
                        cut off where ``z cosh(tau) = 40`` (``e^-40 ~ 4e-18``).
                        Fully vectorised, so it wins on much larger grids, but
                        it sits on a ~1e-12 absolute round-off floor that more
                        nodes do not lower.  Kept as an independent cross-check.

    ``scipy.special.kv`` only accepts real order, which is why neither branch
    uses it.
    """
    nu_arr, z_arr = np.broadcast_arrays(
        np.asarray(nu, dtype=np.float64), np.asarray(z, dtype=np.float64)
    )
    if np.any(z_arr <= 0.0):
        raise ValueError("K_{i nu}(z) is evaluated for z > 0 only")

    if backend == "mpmath":
        import mpmath

        flat = np.empty(nu_arr.size, dtype=np.float64)
        for i, (nv, zv) in enumerate(zip(nu_arr.ravel(), z_arr.ravel())):
            flat[i] = float(mpmath.besselk(1j * mpmath.mpf(float(nv)), mpmath.mpf(float(zv))).real)
        return flat.reshape(nu_arr.shape)

    if backend != "quad":
        raise ValueError(f"unknown backend {backend!r}; use 'quad' or 'mpmath'")

    # tau_max: where the Gaussian-like envelope has decayed past 1e-17.
    tau_max = np.arccosh(np.maximum(40.0 / z_arr, 1.0 + 1e-12))
    nodes, weights = np.polynomial.legendre.leggauss(n_nodes)

    half = 0.5 * tau_max[..., None]
    tau = half * (nodes + 1.0)                       # map [-1, 1] -> [0, tau_max]
    integrand = np.exp(-z_arr[..., None] * np.cosh(tau)) * np.cos(nu_arr[..., None] * tau)
    return np.sum(integrand * weights, axis=-1) * half[..., 0]


def glw2021_beta(omega, omega_prime, g: float, kappa: float, backend: str = "mpmath"):
    """Bogoliubov ``beta_{omega omega'}`` for the Casimir-light mirror (eq. 3).

        beta = -sqrt(w w')/(pi kappa omega_p) * e^{-pi w/(2 kappa)}
               * K_{i w/kappa}(omega_p / g),     omega_p = w + w'.

    The result is real; it is returned as complex128 so both families share one
    on-disk dtype.  (The literature-survey table in section 4 carries an extra
    factor ``s`` here -- that parameter belongs to the other paper and is not
    in eq. 3.)
    """
    if g <= 0.0 or kappa <= 0.0:
        raise ValueError("g and kappa must be positive")

    w = np.asarray(omega, dtype=np.float64)
    wp_ = np.asarray(omega_prime, dtype=np.float64)
    if np.any(w <= 0) or np.any(wp_ <= 0):
        raise ValueError("beta diverges at zero frequency; use omega, omega' > 0")

    w, wp_ = np.broadcast_arrays(w, wp_)
    omega_p = w + wp_

    bessel = besselk_imag_order(w / kappa, omega_p / g, backend=backend)
    beta = (
        -np.sqrt(w * wp_) / (np.pi * kappa * omega_p)
        * np.exp(-np.pi * w / (2.0 * kappa))
        * bessel
    )
    return beta.astype(np.complex128)
