# Moving mirror — documentation

This document records the data generation work in detail: what was produced,
which equations were implemented, how each one was validated and with what
result. [README.md](README.md) gives the short overview.

The project studies the radiation produced by a moving mirror. The aim is to
learn, with a neural operator, the map from a mirror worldline `z(t)` to its
Bogoliubov coefficient `β_{ω,ω'}`. The training data comes from trajectories
that can be solved exactly in the literature, so the correct answer is known
for every sample.

## Work completed so far

Data generation is complete for the two exactly-solvable families listed in
section 4 of the manuscript. Both families have been generated, checked against
the original papers, and documented. No further processing of the data has been
carried out yet.

The first family, `good2016_lambertw`, follows Good, *Reflecting at the Speed
of Light* (arXiv:1612.02459), with the trajectory `z(t) = -(s/κ) W(e^{κt})`.
The second, `glw2021_casimir`, follows Good, Linder and Wilczek, *Finite
Thermal Particle Creation of Casimir Light* (arXiv:2108.11188), with the
worldline defined implicitly by `g(t+z) = -sinh(2κz)`.

Each family contains 500 samples, giving 1000 in total. This matches the
roughly even split described in the manuscript. Every sample is stored as a
`.pt` file holding the sampled parameters, the trajectory `z(t)` on a shared
128-point time grid, and the analytic `β` on a 30 × 30 grid in `(ω, ω')` over
the range `[0.01, 2]`. The samples are stored as raw physical quantities. They
have not been arranged into model tensors, because batching, channel layout and
normalisation are decisions that belong to a later stage.

The code in `data/scripts/` consists of four parts. `mirror_physics.py` holds
the trajectories and the analytic `β` for both papers, including a vectorised
implementation of the Bessel function `K_{iν}(z)` of imaginary order and an
evaluation of `W(e^a)` that remains accurate beyond the point where `exp(a)`
overflows. `gen_good2016.py` and `gen_glw2021.py` generate the two families;
both are seeded, and every grid and sampling range is exposed as a command-line
flag. `sample_io.py` defines the file format, the manifest and the loader.
`verify.py` checks the implementation. The file layout, the constants and their
provenance are documented in [data/README.md](data/README.md).

## Equations implemented

The following expressions are taken from the two papers and implemented in
`mirror_physics.py`. Throughout, `ħ = c = 1`, `ω_p = ω + ω'` and frequencies are
measured in the same units as `κ`.

For Good (2016), with `a = ω/κ`, `b = ω'/κ`, `p = a + b` and
`σ = (1/s + 1)a + (1/s - 1)b`:

- eq. (2), the worldline: `x(t) = -(s/κ) W(e^{κt})`.
- eq. (11), the coefficient:
  `β = (1/κ)(-i/π) s^{ip} √(ab) (iσ)^{ip-1} Γ(-ip)`.
- eq. (12), the modulus squared:
  `|β|² = 2ωω' / (π κ ω_s² ω_p (e^{2πω_p/κ} - 1))`, where
  `ω_s = (1/s + 1)ω + (1/s - 1)ω'`.
- eq. (13), the limit `s → 1`:
  `|β|² = ω' / (2π κ ω ω_p (e^{2πω_p/κ} - 1))`.
- eq. (16), the limit `ω' ≫ ω` at `s = 1`:
  `|β|² = 1 / (2π κ ω (e^{2πω'/κ} - 1))`.

For Good, Linder and Wilczek (2021):

- eq. (1), the worldline: `g v = -sinh(2κx)` with `v = t + x`, inverted as
  `t(x) = -sinh(2κx)/g - x`.
- eq. (3), the coefficient:
  `β = -√(ωω') / (π κ ω_p) · e^{-πω/(2κ)} · K_{iω/κ}(ω_p / g)`.
- eqs. (5) and (6), the particle spectrum:
  `N_ω = Γ̄_ω / (e^{2πω} - 1)` at `κ = 1`, with
  `Γ̄_ω = A/π + B/4`, `A = ln(2g/ω) + Re ψ(1 + iω) - 1`, and `B` twice the real
  part of the second term of eq. (6).

Both papers quote `β` with `κ` scaled out. For Good (2016) the underlying
integral (its eq. 10) obeys the exact scaling
`β_κ(ω, ω') = β_1(ω/κ, ω'/κ)/κ`, and this is how `κ` is restored in the code.
The expression of eq. (3) already carries `κ` explicitly.

## Validation

The script `verify.py` performs the checks below. Each one reproduces a result
that the papers derived by a route independent of the formula being tested, so
agreement is evidence about the transcription rather than a restatement of it.
Every check passes. The errors quoted are those printed by the most recent full
run of `python verify.py --data`, and are relative unless marked otherwise.

**Lambert W function**, used by the first trajectory. The implementation
computes `W(e^a)` without ever forming `e^a`. It was compared with
`scipy.special.lambertw` wherever `e^a` is representable, giving a maximum
error of 1.7e-15, and was checked against its own defining relation
`ln W + W = a` out to `a = 600`, where `e^a` overflows and scipy cannot be
used, giving 5.6e-16.

**Good (2016) trajectory, eq. (2).** The drift speed was measured numerically
at `κt = 5000` for `(s, κ) = (0.3, 1.0), (0.9, 0.7)` and `(1.0, 2.0)`, and
approaches `s` to 2.0e-4 in each case. The residual is not numerical error: the
speed approaches `s` only as `1/W ≈ 1/(κt)`, so 2.0e-4 is the expected value at
that time. The mirror was also confirmed to be asymptotically static at
`x = 0`, to 2.1e-18.

**Good (2016) coefficient, eq. (11).** The implemented `β` was squared and
compared with the paper's own closed form for `|β|²`, eq. (12), on a 37 × 41
frequency grid over `[0.05, 3]`, for `(s, κ) = (1.0, 1.0), (0.6, 1.0),
(0.35, 1.7)` and `(0.9, 0.4)`. The largest error over all four pairs is
3.2e-14. The same quantity was compared with eq. (13), the closed form in the
limit `s → 1`, at `κ = 1.3`, giving 1.3e-14. It was also compared with
eq. (16), the high-frequency form for `ω' ≫ ω`, at `ω = 0.0005, 0.001, 0.002`
and `ω' = 40, 80`, giving 1.3e-2. That last number is larger because the
limiting form itself is only accurate to about 1.3% at those frequencies, since
eq. (16) additionally requires `ω ≪ κ/2π`; the check confirms the approach to
the limit rather than an identity.

**Bessel function of imaginary order.** The vectorised Gauss–Legendre backend
for `K_{iν}(z)` was compared with mpmath at 60 random orders `ν ∈ [0.005, 4]`
in three regimes: `z ≈ 10⁻⁶` (the regime `g ≫ κ` in which the data is
generated), `z ≈ 10⁻²` and `z ≈ 1`. The absolute errors are 1.4e-12, 5.6e-13
and 1.8e-13 respectively.

**Good, Linder and Wilczek (2021) trajectory, eq. (1).** The worldline is
defined implicitly and must be inverted numerically. For
`(g, κ) = (10⁶, 1.0), (10³, 0.5)` and `(10², 2.0)` the inversion round-trips to
6.2e-13 absolute or better, the mirror passes through `x(0) = 0` exactly, and
the peak speed agrees with the analytic value `1/(1 + 2κ/g)` to 3.2e-10 or
better.

**Good, Linder and Wilczek (2021) spectrum, eqs. (5) and (6).** This is the
strongest single result, because it tests the `β` formula, the handling of `κ`
and the Bessel implementation together, against a number the paper obtained by
a completely different derivation. The coefficient of eq. (3) was integrated
numerically, `N_ω = ∫ |β|² dω'`, on a logarithmic grid of 12000 points spanning
`ω' ∈ [10⁻⁹, 10⁹]`, at `g/κ = 10⁶` as used in fig. 1 of the paper, and compared
with the closed form of eq. (5).

| `ω` | `N_ω` (numerical integral of eq. 3) | `N_ω` (eq. 5) | relative error |
|---|---|---|---|
| 0.05 | 5.1590 | 5.1590 | 2.2e-16 |
| 0.10 | 5.3868 | 5.3868 | 0.0 |
| 0.20 | 1.9728 | 1.9728 | 2.2e-16 |
| 0.30 | 0.7493 | 0.7493 | 2.7e-15 |
| 0.40 | 0.4198 | 0.4198 | 1.1e-15 |
| 0.50 | 0.1895 | 0.1895 | 5.8e-15 |

**The generated files.** With `--data`, the checks are repeated on what is
actually stored on disk. For both families the manifest count matches the 500
files present, every trajectory and every `β` entry is finite, all 500 samples
share one time grid, and every trajectory is monotonically leftward. For one
sample drawn from the middle of each family, the stored `|β|²` was re-derived
by a route the generator did not use. For `good2016_lambertw` this used
eq. (12) directly and agrees to 1.4e-14; for `glw2021_casimir` it used the
quadrature Bessel backend rather than the mpmath one used in generation, and
agrees to 3.9e-9, which is the accuracy floor of that backend rather than a
disagreement about the physics.

## Two corrections to the manuscript

Reproducing the papers' own expressions for `|β|²` revealed two errors in the
section 4 table of `moving_mirror.tex`.

First, the `glw2021` row contains a stray factor `s`. That parameter is the
drift speed belonging to the other paper, and eq. (3) of arXiv:2108.11188 does
not contain it.

Second, the `good2016` row has the wrong exponent. The table gives
`(iω_s)^{iω_p}`, but the correct exponent is `iω_p - 1`. This is not a matter
of convention: only the `- 1` reproduces eqs. (12), (13) and (16) of the paper
at the accuracies reported above, and with the exponent as printed the
dependence on `ω_s` disappears from `|β|²` altogether.

The code follows the papers in both cases. Two further decisions, namely the
use of an absolute rather than a κ-scaled time grid and the choice of time
window for the second family, are explained in
[data/README.md](data/README.md#review-notes). Both are worth reviewing before
the data is treated as final.

## Work still to be done

The samples have not yet been converted into model inputs, which includes
batching, channel layout, normalisation and any reduction to `complex64`. The
neural-operator training has not been started.

## Regenerating the data

```bash
cd data/scripts
python gen_good2016.py --overwrite      # about 1 second
python gen_glw2021.py  --overwrite      # about 90 seconds
python verify.py --data                 # checks the formulas and the stored files
```

`verify.py` also accepts `--quick`, which uses a coarser grid for the spectrum
integral, and can be run without `--data` to check the formulas alone.
