# FNO training: z(t) -> beta

Learns the operator that maps a mirror worldline `z(t)` to its Bogoliubov
coefficient `β_{ω,ω'}`, using the samples generated under `data/files/`.

## Files

| File | Role |
|---|---|
| `model_config.py` | FNO architecture (`FNOConfig`) |
| `scheduler_config.py` | Adam settings and the exponential LR decay (`OptimizerConfig`, `ExponentialDecayConfig`) |
| `train_config.py` | The run itself: epochs, batch size, split, device (`TrainConfig`) |
| `fno_tensors.py` | Sample dict -> FNO input/target tensors (`TensorSpec`, `pack_input`, `pack_target`) |
| `dataset.py` | Dataset, 80:20 stratified split, dataloaders |
| `normalization.py` | Input standardisation fitted on the training split |
| `build.py` | Model factory; resolves the vendored `neuralop` |
| `losses.py` | Element-wise MSE, relative L2, zero-prediction baseline |
| `train.py` | Entry point |

## Running

```bash
cd model
python train.py                          # manuscript defaults, 20 epochs
python train.py --epochs 5 --limit 40    # quick smoke run
python train.py --help                   # every flag
```

`neuralop` need not be installed: `build.py` falls back to the vendored copy at
`neuraloperator/`. Runs are written to `model/runs/<run_name>/` as `best.pt`
and `history.json`, each carrying the full resolved configuration.

## How the input is built

The target `β` lives on a 2-D `(ω, ω')` grid, but `z(t)` is a 1-D function of
time. The operator is therefore 2-D on the `(ω, ω')` grid, and the trajectory
enters as channels that are constant across it. The FNO learns a `β` field
conditioned on the trajectory. Input channels, in order:

- 128 channels carrying `z(t)`, one per time sample, each broadcast over the grid,
- 2 one-hot channels naming the family,
- 2 coordinate channels holding `ω` and `ω'`, each scaled to `[0, 1]`.

That is 132 input channels. The output is 2 channels, `(Re β, Im β)`.

Time is normalised per family. The two families were generated on different
windows — good2016 on `t ∈ [-6, 12]`, glw2021 on `t ∈ [-21, 21]` — so each
family's window is mapped onto `[0, 1]`. Because the time grid is shared by
every sample within a family, this is a relabelling of the axis and discards
nothing; the family channel tells the model which window a sample came from.

The `z` channels are standardised using statistics fitted on the training split
alone, since the two families differ in scale (`z ∈ [-10, 0]` against
`z ∈ [-15, 15]`). The target is left in physical units by default, so the
reported MSE is directly comparable with the manuscript.

## Defaults

From the Experiments section of the manuscript: 4 layers, modes `(12, 12)`,
64 hidden channels, 20 epochs, 80:20 split, element-wise MSE. Adam at
`lr = 1e-3` with `weight_decay = 1e-4`, decayed by `gamma = 0.9` per epoch, so
the rate reaches `1.2e-4` by epoch 20. The split is stratified by family, so
both families appear in the same proportion in training and validation.

## Reading the loss

Element-wise MSE on raw `β` is dominated by the largest entries: `|β|` spans
about twelve orders of magnitude across the grid, so **a model predicting zero
everywhere already scores a small MSE**. Every run therefore reports two extra
numbers:

- the zero-prediction baseline MSE, which the model must beat by a wide margin,
- the relative L2 error, which is scale-free and cannot be gamed this way.

A run at the defaults reaches val MSE `1.23e-03` (the manuscript quotes
`1.4930e-03`), against a zero baseline of `2.14e-02` — 17.4x better than
predicting zero — at a relative L2 of `0.227`. The MSE agrees with the
manuscript, while the relative L2 shows there is real headroom left: the model
captures the large entries well and the exponentially small ones much less so.
Predicting `log|β|` instead of `(Re β, Im β)` is the natural next thing to try
if that tail matters.
