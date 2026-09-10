# Moving mirror

Learning the radiation of a moving mirror with a neural operator: the map from
a mirror worldline `z(t)` to its Bogoliubov coefficient `β_{ω,ω'}`. The
training data comes from mirror trajectories that can be solved exactly in the
literature, so the correct answer is known for every sample.

## Repository

- `data/` — data generation code and the generated samples
  ([data/README.md](data/README.md)).
- `Moving_mirror_manuscript/` — the LaTeX manuscript and its figures.
- `Research_papers/` — reference PDFs.
- `neuraloperator/` — a local copy of the neural-operator library.

`data/files/`, `neuraloperator/` and `Research_papers/` are not tracked by git.
The generation scripts are versioned; the samples are regenerated locally.

## Status

Data generation is complete for the two exactly-solvable families of section 4
of the manuscript, and validated against the source papers. Nothing downstream
of the raw samples has been built yet.

| Family | Paper | Trajectory | Samples |
|---|---|---|---|
| `good2016_lambertw` | Good, *Reflecting at the Speed of Light* (arXiv:1612.02459) | `z(t) = -(s/κ) W(e^{κt})` | 500 |
| `glw2021_casimir` | Good, Linder & Wilczek, *Finite Thermal Particle Creation of Casimir Light* (arXiv:2108.11188) | `g(t+z) = -sinh(2κz)` | 500 |

Each sample holds the trajectory `z(t)` on a shared 128-point time grid and the
analytic `β` on a 30 × 30 grid in `(ω, ω')`, stored as raw physical quantities
rather than as model tensors.

Every check in `verify.py` passes. The implemented coefficients reproduce the
papers' own published expressions to a relative error of order 10⁻¹⁴, and the
particle spectrum of the second family reproduces eq. (5) of arXiv:2108.11188
to order 10⁻¹⁵.

Still to be done: packing the samples into model inputs, and the neural-operator
training itself.

## Quickstart

```bash
cd data/scripts
python gen_good2016.py --overwrite      # about 1 second
python gen_glw2021.py  --overwrite      # about 90 seconds
python verify.py --data                 # checks the formulas and the stored files
```

## Documentation

[Documentation.md](Documentation.md) gives the full account: the equations
implemented and their numbers in the source papers, every validation check with
its measured result, and two errors found in the section 4 table of the
manuscript.

[Mathematical_report.md](Mathematical_report.md) walks through the mathematics
end to end — worldline, coefficient, grids, numerical methods, tensor packing,
loss — justifying each step and flagging the four where better mathematics is
available.

[model/README.md](model/README.md) covers the training pipeline, and
[data/README.md](data/README.md) documents the file format, the constants, and
the open review points on the generated data.
