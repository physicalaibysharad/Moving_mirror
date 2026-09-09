# Moving mirror

This project studies the radiation produced by a moving mirror. The aim is to
learn, with a neural operator, the map from a mirror worldline `z(t)` to its
Bogoliubov coefficient `β_{ω,ω'}`. The training data comes from trajectories
that can be solved exactly in the literature, so the correct answer is known
for every sample.

## Contents of the repository

- `data/` — the data generation code and the generated samples. A detailed
  description is in [data/README.md](data/README.md).
- `Moving_mirror_manuscript/` — the LaTeX manuscript and its figures.
- `Research_papers/` — reference PDFs (Fulling, Davies, Moore, Good and others).
- `neuraloperator/` — a local copy of the neural-operator library.

The directories `data/files/`, `neuraloperator/` and `Research_papers/` are not
tracked by git. The generation scripts are versioned; the samples they produce
are regenerated locally.

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
`verify.py` checks the implementation.

## Validation

All checks in `verify.py --data` pass. Each check reproduces a result that the
papers derived by an independent route, so agreement is evidence about the
implementation rather than a restatement of it.

The most informative check is the particle spectrum `N_ω` of the second family,
compared with eq. (5) and fig. 1 of arXiv:2108.11188 at `g/κ = 10⁶`. It agrees
to a relative error of 2.7e-15 or better. This single check exercises the `β`
formula, the treatment of `κ` and the Bessel implementation together. The
complete set of results is tabulated in
[data/README.md](data/README.md#validation-results).

## Two corrections to the manuscript

Reproducing the papers' own expressions for `|β|²` revealed two errors in the
section 4 table of `moving_mirror.tex`.

First, the `glw2021` row contains a stray factor `s`. That parameter is the
drift speed belonging to the other paper, and eq. (3) of arXiv:2108.11188 does
not contain it.

Second, the `good2016` row has the wrong exponent. The table gives
`(iω_s)^{iω_p}`, but the correct exponent is `iω_p - 1`. This is not a matter
of convention: only the `- 1` reproduces eqs. (12), (13) and (16) of the paper,
and with the exponent as printed the dependence on `ω_s` disappears from `|β|²`
altogether.

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
