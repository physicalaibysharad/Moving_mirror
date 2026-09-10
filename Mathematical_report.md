# From mirror worldline to trained operator: the mathematics, step by step

This report follows the whole chain, from the two mirror trajectories through
to the trained model, and explains what each step does and why it is done that
way. It starts from the trajectories and takes the Bogoliubov coefficient as a
given definition; the field-theory derivation is standard and is cited in the
manuscript.

Every step is written as *what happens*, then *why it is justified*. Where the
justification rests on convention rather than on an argument, the step is
marked **Open point** and a better alternative is given. Those are
recommendations only — the code as it stands is unchanged, and the numbers
quoted throughout come from it.

Notation: `ħ = c = 1`; the mirror's position is `z(t)` (the papers write
`x(t)`); `ω` and `ω'` are the outgoing and incoming frequencies;
`ω_p = ω + ω'`; `κ` is the acceleration parameter.

---

## Step 1. The two worldlines

**What happens.** Two families of mirror motion are used, both taken from
papers that solve them exactly.

The first, from Good (2016), is given in closed form:

```
z(t) = -(s/κ) W(e^{κt})
```

where `W` is the Lambert W function and `0 < s ≤ 1`. The mirror begins
asymptotically at rest, accelerates, and coasts away at a final drift speed `s`.

The second, from Good, Linder and Wilczek (2021), is given implicitly:

```
g·v = -sinh(2κz),      where v = t + z
```

This mirror is asymptotically at rest at both ends, and radiates thermally when
`g ≫ κ`.

**Why.** The point of the dataset is that the target is known exactly. For a
neural operator we need pairs `(z(t), β)`, and if `β` had to be computed by
numerical integration the labels would carry their own error, which we could
then not distinguish from the model's error. These two families were chosen
because each has a published closed form for `β`, so the labels are exact to
machine precision. They are also complementary: one drifts away at constant
speed forever, the other returns to rest, so the dataset covers two genuinely
different asymptotic behaviours rather than two variations of one.

---

## Step 2. From a worldline to the coefficient β

**What happens.** A moving mirror reflects incoming light rays. In 1+1
dimensions a light ray is labelled by a single null coordinate, and reflection
off the mirror maps each incoming ray `v` to a unique outgoing ray `u = p(v)`,
where the function `p` is fixed entirely by the worldline. The Bogoliubov
coefficient is an overlap integral of the schematic form

```
β_{ω,ω'} ∝ ∫ dv √(ω/ω') e^{-iω'v - iω p(v)}
```

Its magnitude `|β_{ω,ω'}|²` is the number of particles produced in outgoing
mode `ω` from incoming mode `ω'`. If the mirror never moved, `p(v) = v` and `β`
would vanish; `β ≠ 0` is exactly the statement that motion creates particles.

**Why this is not what the code computes.** The code never evaluates that
integral. Both papers have already done it analytically, and the code
implements their results directly:

- Good (2016), eq. (11), with `a = ω/κ`, `b = ω'/κ`, `p = a + b` and
  `σ = (1/s + 1)a + (1/s - 1)b`:

  ```
  β = (1/κ)·(-i/π)·s^{ip}·√(ab)·(iσ)^{ip-1}·Γ(-ip)
  ```

- Good, Linder and Wilczek (2021), eq. (3):

  ```
  β = -√(ωω')/(π κ ω_p)·e^{-πω/(2κ)}·K_{iω/κ}(ω_p/g)
  ```

**Why this is justified.** Evaluating the oscillatory integral numerically
would be both slower and less accurate than using the closed forms. The risk in
using published formulas is transcription error, and that risk is addressed
directly: each formula is checked against *other* results in the same papers
that were derived by a different route (Step 11). Two transcription errors in
the manuscript's own summary table were caught this way.

---

## Step 3. Restoring κ

**What happens.** Both papers quote `β` with `κ` scaled out, effectively
setting `κ = 1`. For the second family the formula carries `κ` explicitly, so
nothing is needed. For the first, `κ` is restored using

```
β_κ(ω, ω') = (1/κ)·β_1(ω/κ, ω'/κ)
```

**Why this is exactly right, not an approximation.** The Lambert-W worldline is
self-similar. Writing `z_1` for the shape at `κ = 1` with the same `s`:

```
z_κ(t) = -(s/κ)W(e^{κt}) = (1/κ)·z_1(κt)
```

So changing `κ` is nothing more than viewing the *same* curve on rescaled time
and space axes, `(t, z) → (t/κ, z/κ)`. That rescaling carries straight through
the overlap integral of Step 2: substituting `v → v/κ` turns `dv` into `dv/κ`,
which gives the `1/κ` prefactor, and turns each `ω·v` into `(ω/κ)·(κv)`, which
gives the rescaled frequencies. The `√(ω/ω')` factor is a ratio, so it does not
change at all. The identity is therefore exact.

This is worth stating clearly because it also explains something about the data:
**at fixed `s`, `κ` is a pure scale parameter, not a shape parameter.** That
fact returns in Step 5.

---

## Step 4. Choosing the parameter ranges

**What happens.** For family one, `s` is drawn uniformly from `[0.1, 1.0]` and
`κ` log-uniformly from `[0.5, 1.5]`. For family two, `g/κ` is drawn
log-uniformly from `[10², 10⁶]` and `κ` log-uniformly from `[0.5, 1.5]`. 500
samples each, from fixed seeds.

**Why uniform for `s` but log-uniform for `κ` and `g/κ`.** `s` is a speed as a
fraction of `c`; it lives on a bounded interval with a physically meaningful
scale, and `s = 0.5` sits genuinely halfway between `0.1` and `1.0`. Uniform
sampling is the natural choice.

`κ` and `g/κ` are different: they are scale parameters, and what matters
physically is their ratio, not their difference. `g/κ` spans four decades, and
sampling it uniformly would put 90% of the samples in the top decade, leaving
the interesting `g/κ ~ 10²` regime almost unrepresented. Log-uniform sampling
gives each decade equal weight. The same reasoning applies to `κ`, though over
a much narrower range.

**Why these particular ranges.** `s ∈ (0, 1]` is forced by the physics — the
mirror cannot exceed the speed of light. `g/κ ≥ 10²` keeps the mirror in the
thermal regime the paper analyses, and `10⁶` is the value used in its fig. 1.
`κ ∈ [0.5, 1.5]` brackets the manuscript's reference value `κ = 1`.

---

## Step 5. The time grid

**What happens.** Trajectories are sampled at 128 points on an **absolute**
time grid — `t ∈ [-6, 12]` for family one — rather than on a grid that scales
with `κ`.

**Why this matters, and why absolute wins.** This looks like a bookkeeping
choice and is not. From Step 3, `z_κ(t) = (1/κ)z_1(κt)`. If the time grid were
also scaled by `κ`, say `t = τ/κ` with `τ` fixed, then every sampled trajectory
would read

```
z = (s/κ)·(one fixed shape in τ)
```

Every trajectory in the family would be the same curve scaled by `s/κ`. The
inputs would then depend on `s` and `κ` **only through the single combination
`s/κ`**, while the target `β` depends on `s` and `κ` separately. No model of
any kind could recover both — the information simply would not be in the input.

On an absolute grid the degeneracy is broken: `κ` controls *where along the
time axis* the acceleration knee falls, and `s/κ` controls how deep it is.
Both remain recoverable.

This is the single choice in the data pipeline most likely to break training
silently if it were changed. If a downstream script ever rescales time by `κ`,
the degeneracy comes straight back.

---

## Step 6. The time window for the second family

**What happens.** Family two has no natural fixed window. The mirror leaves its
near-null phase at roughly

```
t ≈ ln(g/κ)/(2κ)
```

which varies by about a factor of 8 across the sampled parameter range. The
generator computes the window each sample would need, then uses a single shared
window sized for the slowest sample actually drawn.

**Why a shared window.** An operator learner needs all its inputs on a common
domain. If each sample had its own time grid, the value at index 40 would mean
a different physical time in different samples, and the model would be reading
inconsistent input.

**Why sized to the slowest.** A window that is too small truncates the
trajectory while the mirror is still moving — at the `g/κ = 10⁶, κ = 0.5`
corner, a half-width of 12 would cut the mirror off while it is still travelling
at 0.88c, discarding the very part of the motion that produces the radiation.
Truncating the input while the physics is still happening is a worse error than
sampling it coarsely.

**The cost, stated plainly.** The fastest samples then get roughly 32 of their
128 points spanning the transition instead of all 128. This is a real loss of
resolution, and the generator prints the number on every run. It is the right
trade only because missing data is worse than coarse data.

---

## Step 7. The frequency grid

**What happens.** `β` is evaluated on a 30 × 30 uniform grid with `ω` and `ω'`
both running over `[0.01, 2]`.

**Why zero is excluded.** `β` diverges as either frequency approaches zero —
this is the standard infrared divergence of the massless field in 1+1
dimensions, not a numerical artefact. `0.01` is a cutoff, and any grid must have
one.

**Open point: the grid is uniform, but `β` is not.** Measured on the generated
data, for one typical sample of family one:

| quantity | value |
|---|---|
| `\|β\|` at the low-frequency corner | 7.5 |
| `\|β\|` at the high-frequency corner | 6.3e-06 |
| share of `Σ\|β\|²` in the lowest-`ω` row alone | **96.2%** |
| share in the lowest 5 × 5 block | 96.3% |

Almost the entire magnitude of the target sits in a single row of a 30-row
grid. The uniform grid spends 29 of its 30 rows resolving the part of the
function that is nearly zero, and one row on the part that carries everything.
This propagates into the loss (Step 15).

**Better mathematics.** Sample uniformly in `ln ω` instead of in `ω`:

```
ω_k = ω_min·(ω_max/ω_min)^{k/(N-1)}
```

Since `β` falls off exponentially, it varies smoothly and at a roughly constant
rate in `ln ω`, which is what a fixed grid resolves well. Importantly this
**does not break the FNO**: the spectral convolution needs equally spaced
points, and these points are equally spaced in the coordinate `ξ = ln ω`. The
model would simply be learning an operator in `ξ` rather than in `ω`. The
present grid follows the manuscript, which is a reason to keep it for
comparability, but not an argument that it resolves the function well.

---

## Step 8. Evaluating the Lambert W function

**What happens.** Family one needs `W(e^{κt})`. For `κt > 709` the quantity
`e^{κt}` overflows a double, so the code never forms it. Instead it solves the
defining relation in log space. Writing `w = W(e^a)`, the definition
`w·e^w = e^a` gives `ln w + w = a`, and substituting `u = ln w` gives

```
G(u) = e^u + u - a = 0
```

which is solved by Newton's method, returning `w = e^u`.

**Why this converges, and why no bracketing is needed.**
`G'(u) = e^u + 1 > 0` and `G''(u) = e^u > 0`, so `G` is strictly increasing and
convex. For such a function Newton's method started anywhere with `G ≥ 0`
decreases monotonically to the root and cannot overshoot — this is a standard
convexity result, not a heuristic. The two starting points used satisfy that
condition exactly:

- for `a > 1`, start at `u₀ = ln a`, giving `G = a + ln a - a = ln a ≥ 0` ✓
- otherwise start at `u₀ = a`, giving `G = e^a + a - a = e^a > 0` ✓

Convergence is quadratic. In practice the check that `ln W + W = a` holds to
5.6e-16 out to `a = 600` confirms both the method and the fact that the
overflow was genuinely avoided, since `scipy` cannot be evaluated there at all.

---

## Step 9. Evaluating β without overflow

**What happens.** The closed form for family one is a product of terms that are
individually extreme: `Γ(-ip)`, `(iσ)^{ip-1}` and `s^{ip}`. Rather than
multiplying them, the code adds their logarithms and exponentiates once at the
end, using `loggamma` in place of `Γ`.

**Why.** Both `|Γ(-ip)|` and `|(iσ)^{ip}|` decay like `e^{-πp/2}`, so their
product decays like `e^{-πp}`. At `κ = 0.5` and `ω_p = 4` this is already
`e^{-8π} ≈ 10^{-11}`, and it falls much faster on wider grids or at smaller `κ`.
The final answer is representable, but forming the factors separately would
push intermediate values toward the edge of the floating-point range for no
benefit. Working in the logarithm keeps every intermediate quantity in a
comfortable range, and turns a product into a sum, which is also numerically
better conditioned.

---

## Step 10. Two more numerical steps

**Inverting the implicit worldline.** Family two gives `t` as a function of
`z`, not the reverse:

```
t(z) = -sinh(2κz)/g - z
```

There is no closed-form inverse, so it is inverted numerically. This is safe
because

```
dt/dz = -(2κ/g)·cosh(2κz) - 1 < 0
```

strictly, for all `z` — the function is strictly decreasing, so the inverse
exists and is unique everywhere. The code brackets the root by doubling an
interval outward until it straddles the target time, then uses Brent's method.
Bracketing plus strict monotonicity means the root is guaranteed to be found;
Brent's method then converges superlinearly. The bracket is capped at
`z = 300/(2κ)` so that `sinh` cannot overflow. The inversion is checked by
round-tripping, which closes to 6.2e-13.

**Bessel K of imaginary order.** Family two needs `K_{iν}(z)`, which `scipy`
does not provide — its `kv` accepts only real order. Two independent
implementations are used:

- `mpmath`, evaluated element by element at high precision (the default: on a
  30 × 30 grid it is both faster and more accurate);
- Gauss–Legendre quadrature on the integral representation

  ```
  K_{iν}(z) = ∫₀^∞ e^{-z·cosh(τ)}·cos(ν τ) dτ
  ```

  truncated where `z·cosh(τ) = 40`, since `e^{-40} ≈ 4·10^{-18}` is below
  double precision relative to the peak, so the discarded tail cannot affect
  the result.

**Why keep both.** They are not redundant. The quadrature is fully vectorised
and wins on much larger grids, but more importantly it provides an independent
route to the same number: agreement between two methods that share no code is
evidence neither is wrong. Their difference is ~1e-12 absolute. That floor
comes from cancellation in the oscillatory `cos(ντ)` factor and does not improve
with more nodes, which is why the quadrature is held to an absolute rather than
relative tolerance.

Also worth noting: `K_{iν}(z)` is **real** for real `z`, because `K_{-v} = K_v`
and `conj(K_v(z)) = K_{conj(v)}(z)`. So `β` for family two is real-valued. This
has a consequence for training, in Step 14.

---

## Step 11. Validation

**What happens.** Every formula is checked against a result the source papers
derived by a different route.

**Why this is the only check worth making.** Verifying a formula against itself
proves nothing. What the checks do instead is reproduce independent published
results: `|β|²` against the papers' own eqs. (12), (13) and (16); the particle
spectrum `N_ω = ∫|β|²dω'` against eq. (5) and fig. 1. That last one is the
strongest, because the paper obtained `N_ω` by a completely different
derivation, and reproducing it exercises the `β` formula, the `κ` handling and
the Bessel implementation simultaneously. It agrees to 5.8e-15 or better.

The full table of results is in [Documentation.md](Documentation.md#validation).
This is also how the two errors in the manuscript's summary table were found.

---

## Step 12. Storage

**What happens.** `β` is stored as `complex128`; trajectories as `float64`.

**Why.** `|β|` spans about twelve orders of magnitude across the grid.
`complex64` carries roughly 7 significant digits with a smallest normal value
near `10^{-38}`, so it would hold the values but lose precision on the small
ones. Storage is cheap here — 1000 samples is a few tens of megabytes — so
there is no reason to economise before knowing what the downstream model needs.
Narrowing is left to the packing step, where it can be reconsidered.

---

## Step 13. Packing into tensors

**What happens.** Each sample becomes an input tensor of shape
`(132, 30, 30)` and a target of shape `(2, 30, 30)`. The input channels are:

- **128 channels** carrying `z(t)`, one per time sample, each held constant
  across the whole `(ω, ω')` grid;
- **2 channels** one-hot encoding which family the sample came from;
- **2 channels** holding `ω` and `ω'`, each rescaled to `[0, 1]`.

Time is normalised per family: each family's own window is mapped to `[0, 1]`.

**Why the time normalisation loses nothing.** The two families were generated
on different windows, `[-6, 12]` and `[-21, 21]`. Within a family every sample
shares one grid, so mapping that grid to `[0, 1]` is the same affine relabelling
applied to every sample in the family — it is renaming the axis, not resampling
the data. No information is destroyed, and the family channel tells the model
which of the two windows it is looking at. Note this is *not* the same as the
`κ`-rescaling forbidden in Step 5: that one differed from sample to sample,
which is precisely why it destroyed information.

**Why the trajectory has to be broadcast at all.** There is a genuine shape
mismatch: the input `z(t)` is a function of one variable, the output `β` is a
function of two. The operator is defined on the `(ω, ω')` grid because that is
where the output lives, so the trajectory has to be injected as a condition.
Holding it constant across the grid is the simplest way to do that.

**Open point: 128 constant channels is a crude way to inject 128 numbers.** A
channel that is constant across the grid has no spatial structure of its own.
All the spatial structure in the model's first layer must therefore be
manufactured by the pointwise lifting network acting on those constants
*together with* the two coordinate channels. The trajectory information is
present, but it is presented in a form the spectral convolution cannot use
directly, and 128 copies of the same constant across 900 grid points is a
redundant encoding — it inflates the input tensor by a factor of 900 relative
to the information it carries.

**Better mathematics.** Two options, both standard:

1. **A branch/trunk split (DeepONet-style).** Encode `z(t)` with a small
   network into `k` coefficients, and let a trunk network provide `k` basis
   functions over `(ω, ω')`; the output is their inner product. This matches
   the actual structure of the problem — a function of `t` conditioning a
   function of `(ω, ω')` — instead of forcing it into a same-domain shape.
2. **Compress `z(t)` first.** The trajectories are smooth and come from a
   two-parameter family, so their intrinsic dimension is about 2, not 128. A
   handful of POD or Fourier coefficients would carry nearly all of the
   information, and the model would receive perhaps 6 channels rather than 128.

Neither is a correctness fix — the current encoding demonstrably works — but
both are better matched to the mathematics.

**Minor redundancy.** The library's FNO appends its own coordinate grid
internally when `positional_embedding="grid"`, so the model actually receives
134 channels: the 132 packed here plus 2 more that duplicate the coordinate
channels. Harmless, but one of the two could be dropped.

---

## Step 14. The target

**What happens.** `β` is split into two real channels, `(Re β, Im β)`, and kept
at `float64` (the pipeline now runs end to end in double precision, matching
the `complex128` the samples are generated in — see
[model/README.md](model/README.md)).

**float32 would also have been safe here**, for what it is worth: the real and
imaginary parts are bounded by `|β| ≲ 8`, and `float32` denormals reach about
`10^{-45}`, far below the smallest entries (`~10^{-12}`). Precision, not range,
would have been what is lost, and about 7 significant digits would have been
ample for a model whose error is many orders of magnitude larger — though this
would **not** have been safe for a log-magnitude target, where the small
values become large negative numbers that must be resolved accurately.

**Why splitting into real and imaginary parts is reasonable — checked, not
assumed.** A concern with this representation is that if the phase of `β` wound
rapidly across the grid, `Re β` and `Im β` would oscillate quickly and a
spectral model keeping only 12 modes would struggle. Measured on the data, the
total phase sweep across a full row is about 4.3 radians — under 0.7 of a
cycle — and `Re β` changes sign once across 30 points. The phase is smooth on
this grid, so the concern does not apply and the representation is well suited
to a low-mode spectral model.

**Open point: for half the dataset, one output channel is identically zero.**
As noted in Step 10, `β` for family two is exactly real. So for all 500
`glw2021` samples, the entire `Im β` channel is zero, and predicting zero there
is exactly correct. Measured over the dataset, the imaginary channel holds 33%
of the total target magnitude, all of it from family one. A third of the
model's output budget is therefore spent on a quantity that is free to get
right for half the inputs — which also flatters the reported loss.

**Better mathematics.** Either predict `|β|` (real and meaningful for both
families) with phase as a separate output only where it exists, or give the two
families separate output heads, or weight the imaginary channel by family so
the trivially-zero half does not enter the average. The cleanest is the first:
`|β|²` is the physically observable quantity, and the phase is convention-
dependent in a way the magnitude is not.

---

## Step 15. Splitting, training, and reading the result

**Splitting.** 80% train, 20% validation, drawn with a fixed seed and
stratified by family, so both families appear in the same proportion in both
splits. Without stratification a random draw could leave the families unevenly
represented and make the validation score depend on the draw rather than on the
model.

**Training.** A Fourier Neural Operator — 4 layers, 12 × 12 modes, 64 hidden
channels — is trained for 500 epochs with Adam, minimising element-wise mean
squared error. The learning rate follows a two-phase geometric warmup-then-
decay (`1e-4 → 1e-2` over the first 15% of the run, then `1e-2 → 1e-6` by the
last epoch), rather than a flat exponential decay from the start. The details
are in [model/README.md](model/README.md).

**Reading the loss, which needs care.** The quantity minimised is still
element-wise MSE on raw `β`. Combined with Step 7, this means:

> Roughly 96% of the target's squared magnitude lies in the lowest-frequency
> row of the grid. The MSE is therefore, to a good approximation, a measure of
> how well the model fits that one row.

And because most of the grid is nearly zero, **a model that predicted zero
everywhere would already score a small MSE.** So the raw MSE cannot be read on
its own — which is why it is no longer what gets *reported*. `train.py` now
reports only the **relative L² error**
(`‖pred − target‖ / ‖target‖`), for both train and validation, at every epoch;
it divides by the size of each target and so cannot be made small by
predicting zero, and it is what the best checkpoint is now selected on. This
implements the "better mathematics" this section used to recommend.

An earlier run — 20 epochs, `float32`, flat exponential decay — scored an MSE
of 1.23e-03 against a zero-prediction baseline of 2.14e-02 (17.4× better than
predicting nothing) and a relative L² of 0.227 (≈23%), with the MSE agreeing
closely with the manuscript's reported 1.4930e-03. Those numbers are from that
earlier configuration, not the current 500-epoch `float64` pipeline described
above, and have not yet been reproduced under it.

**Still open.** Relative L² is now what training selects on and what gets
reported, but the training *objective* is still absolute MSE — so the model is
still optimised toward the large, low-frequency entries and the exponentially
small ones are still only a secondary beneficiary. Making the loss itself
scale-free (an MSE on `log|β|`, say) rather than just the reported metric would
close that gap; this is the same underlying issue as the uniform grid in
Step 7 — both treat a function that varies exponentially as though it varied
linearly.

---

## Summary of open points

| Step | Issue | Recommended change |
|---|---|---|
| 7 | Uniform `ω` grid, but 96% of `\|β\|²` sits in one row | Sample uniformly in `ln ω`; still uniform for the FNO |
| 13 | 128 constant channels is a redundant encoding of the trajectory | Branch/trunk split, or compress `z(t)` to a few coefficients |
| 14 | `Im β` is identically zero for half the dataset | Predict `\|β\|`, or use per-family output heads |
| 15 | Absolute MSE is dominated by the largest entries | Now reported as relative L²; training objective is still absolute MSE — MSE on `log\|β\|` would close that gap |

The first and last are the same mathematical point seen twice: `β` varies
exponentially, and both the grid and the loss treat it as though it varied
linearly. Fixing either would likely matter more than any change to the
architecture.

Everything else in the chain rests on an argument rather than a convention: the
`κ`-restoration identity is exact, the Newton and Brent iterations have
convergence guarantees from monotonicity and convexity, the quadrature cutoff
is below machine precision, and the formulas themselves are checked against
independent published results.
