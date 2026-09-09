# Moving-mirror data: section 4 trajectories

Trajectory / Bogoliubov-coefficient data for the two exactly-solvable mirrors in
the literature survey (section 4 of `Moving_mirror_manuscript/moving_mirror.tex`).

| Family | Paper | Trajectory | Parameters |
|---|---|---|---|
| `good2016_lambertw` | arXiv:1612.02459 — Good, *Reflecting at the Speed of Light* | `z(t) = -(s/κ) W(e^{κt})` | `s` (drift speed, 0<s≤1), `κ` |
| `glw2021_casimir` | arXiv:2108.11188 — Good, Linder & Wilczek, *Finite Thermal Particle Creation of Casimir Light* | `g(t+z) = -sinh(2κz)` | `g`, `κ` (thermal for g≫κ) |

**Status:** generated and validated — 500 samples per family, every check in
`verify.py --data` passing. Four items need your review before this is treated
as settled; see [Review notes](#review-notes) at the end.

## Layout

```
data/
  scripts/
    mirror_physics.py   trajectories + analytic beta for both papers
    sample_io.py        on-disk format, manifest, loader
    gen_good2016.py     generator for family 1
    gen_glw2021.py      generator for family 2
    verify.py           validates the formulas against the papers' own results
  files/
    good2016_lambertw/  manifest.json + 500 x sample_XXXX_s..._kappa....pt
    glw2021_casimir/    manifest.json + 500 x sample_XXXX_g..._kappa....pt
```

One directory per family, one `.pt` file per parameter set. Each file is a dict:

| key | dtype / shape | meaning |
|---|---|---|
| `family`, `paper`, `index` | str, str, int | provenance |
| `params` | dict of float | the sampled parameters |
| `t` | float64 `(128,)` | time grid, **shared by every sample in the family** |
| `z` | float64 `(128,)` | mirror position `z(t)` (the papers' `x(t)`) |
| `omega`, `omega_prime` | float64 `(30,)` | frequency grids |
| `beta` | complex128 `(30, 30)` | `beta[i,j] = β_{ω_i, ω'_j}` |
| `meta` | dict | grid config, formula reference, units |

These are raw physics samples, deliberately *not* packed into model tensors —
batching, channel layout, normalisation and any narrowing to complex64 belong to
a separate downstream script.

## Regenerating

```bash
cd scripts
python gen_good2016.py --overwrite      # ~1 s
python gen_glw2021.py  --overwrite      # ~90 s (Bessel K of imaginary order)
python verify.py --data                 # validate formulas + what is on disk
```

Both generators are seeded (`--seed`) and expose every grid and sampling range
as a flag; `--help` lists them. `prepare_out_dir` refuses to mix new samples in
with stale ones unless `--overwrite` is passed.

## Constants

Grid defaults follow the Experiments section of the manuscript: 128 time
samples, a 30×30 `(ω, ω')` grid on `[0.01, 2]` (0 excluded — β diverges there),
500 samples per family (1000 across both, matching the manuscript's ~50/50
split). Parameter ranges come from the papers: `s ∈ (0,1]` and `g/κ ∈ [10², 10⁶]`
(the paper's fig. 1 uses `g/κ = 10⁶`), with `κ` log-uniform on `[0.5, 1.5]`
around the manuscript's reference `κ = 1`. Units are `ħ = c = 1`, frequencies in
the same units as `κ`.

## Conventions

The code follows the papers, not the section 4 table, in two places where they
disagree; see [Review notes](#review-notes) items 1 and 2.

Both papers quote β with `κ` scaled out. For `good2016` the beta integral
(its eq. 10) obeys the exact scaling `β_κ(ω,ω') = β_1(ω/κ, ω'/κ)/κ`, which is
how `κ` is restored; the `glw2021` formula already carries `κ` explicitly.

`β` is real-valued for `glw2021` (`K` of imaginary order is real for real
argument) but is stored complex128 so both families share one dtype. complex128
rather than complex64 because `|β|` spans ~13 orders of magnitude across the
grid.

## Validation results

Last full run of `python verify.py --data` — all checks passing:

| Check | Measured |
|---|---|
| `good2016` \|β\|² vs eq. (12), four (s, κ) pairs | ≤ 3.2e-14 rel |
| `good2016` \|β\|² vs eq. (13), the s→1 limit | 1.3e-14 rel |
| `good2016` \|β\|² vs eq. (16), ω'≫ω limit | 1.3e-2 rel (limit is itself ~1.3% here) |
| `glw2021` `N_ω` vs eq. (5) / fig. 1 at g/κ=10⁶, six ω | ≤ 2.7e-15 rel |
| `glw2021` worldline inversion round-trip | ≤ 6.2e-13 abs |
| `K_{iν}(z)` quad backend vs mpmath | ≤ 1.4e-12 abs |
| `W(e^a)` vs scipy; `ln W + W = a` to a=600 | 1.7e-15 / 5.6e-16 |
| 1000 stored samples: finite, shared grid, monotonic | 0 failures |
| Stored \|β\|² re-derived independently | 1.4e-14 / 3.9e-9 rel |

The `N_ω` agreement is the strongest single result: it exercises the β formula,
the κ handling and the Bessel implementation at once, against a number the paper
derived by a completely different route.

## What `verify.py` checks

Each check reproduces something the papers state independently of the formula
being tested:

- `W(e^a)` against `scipy.special.lambertw`, and `ln W + W = a` out to `a = 600`
  (where `exp(a)` overflows and scipy cannot be used).
- `good2016` drift speed → `s`; `|β|²` against eqs. (12), (13) and the
  high-frequency limit (16).
- `K_{iν}(z)`: the vectorised Gauss–Legendre backend against mpmath.
- `glw2021` worldline inversion round-trip, `z(0)=0`, and peak speed
  `1/(1+2κ/g)`.
- `glw2021` spectrum `N_ω = ∫|β|²dω'` against eq. (5)/fig. 1 at `g/κ = 10⁶`
  — currently agreeing to ~1e-15.
- With `--data`: shared grids, finite and monotonic trajectories, and stored `β`
  re-derived by a route the generator did not use.

## Review notes

Four things I decided or could not resolve while building this. The first two
change the manuscript; the last two change the data.

**1. The section 4 table has a stray `s` in the `glw2021` row.** The table gives
`-s·√(ωω')/(πκω_p)·…`. Eq. (3) of arXiv:2108.11188 has no `s` — that parameter
is the drift speed from the *other* paper. The code follows the paper. Worth
fixing in `moving_mirror.tex`.

**2. The section 4 table has the wrong exponent in the `good2016` row.** The
table gives `(iω_s)^{iω_p}`; it is `(iω_s)^{iω_p - 1}`. This is not a judgement
call — only the `- 1` reproduces the paper's own `|β|²` in eqs. (12), (13) and
(16), which `verify.py` checks to ~1e-14. With the table's exponent the `ω_s`
dependence disappears from `|β|²` entirely. Also worth fixing in the manuscript.

**3. The time grid is absolute rather than κ-scaled.** On a κ-scaled grid the
`good2016` trajectory collapses to `(s/κ) × (one fixed shape)`, so `s` and `κ`
become degenerate in the input and no operator could separate them. On a fixed
grid `κ` sets where the knee falls and `s/κ` sets its depth, and both stay
recoverable. This is the choice most likely to matter for training — if a
downstream script rescales time by κ, the degeneracy comes back.

**4. The `glw2021` time window trades resolution for coverage.** The mirror
leaves its near-null phase at `t ~ ln(g/κ)/(2κ)`, which spans a factor of ~8
over the default parameter ranges. A single fixed half-width of 12 would cut off
the `g/κ=10⁶, κ=0.5` corner while it is still moving at 0.88c, so the default
`--t-window auto` sizes one shared window to the slowest sample drawn. The cost
is ~32 of 128 points spanning the fastest sample's transition; the generator
prints that number on every run. If training wants uniform resolution more than
full coverage, narrow `--g-over-kappa-max` (to 10⁴, say) and the two goals stop
competing. `--t-window fixed` and `--t-window per-sample` are also available;
per-sample gives each sample its own grid and so breaks the shared domain.

### Unresolved

The brief mentioned an "experiments folder" as a source for constant values.
There is no such folder in the repository. The constants here come from the
Experiments *section* of `Moving_mirror_manuscript/moving_mirror.tex` — 128 time
samples, 30×30 on `[0.01, 2]`, 1000 trajectories in a ~50/50 split. If a
separate experiments folder exists elsewhere, these values should be re-checked
against it; the grid and sampling ranges are all flags, so nothing needs
rewriting to change them.
