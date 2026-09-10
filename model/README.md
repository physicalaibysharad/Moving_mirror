# FNO training: z(t) -> beta

Learns the operator that maps a mirror worldline `z(t)` to its Bogoliubov
coefficient `β_{ω,ω'}`, using the samples generated under `data/files/`.

## Files

| File | Role |
|---|---|
| `model_config.py` | FNO architecture (`FNOConfig`) |
| `scheduler_config.py` | Adam settings and the warmup-then-decay LR schedule (`OptimizerConfig`, `WarmupDecayConfig`) |
| `train_config.py` | The run itself: epochs, batch size, split, device (`TrainConfig`) |
| `fno_tensors.py` | Sample dict -> FNO input/target tensors (`TensorSpec`, `pack_input`, `pack_target`) |
| `dataset.py` | Dataset, 80:20 stratified split, dataloaders |
| `normalization.py` | Input standardisation fitted on the training split |
| `build.py` | Model factory; resolves the vendored `neuralop` |
| `losses.py` | Element-wise MSE (training objective) and relative L2 (reported metric) |
| `train.py` | Entry point |

## Running

```bash
cd model
python train.py                          # defaults, 500 epochs
python train.py --epochs 5 --limit 40    # quick smoke run
python train.py --help                   # every flag
```

`neuralop` need not be installed: `build.py` falls back to the vendored copy at
`neuraloperator/`. Runs are written to `outputs/runs/<run_name>/` (at the repo
root, not under `model/`) as `best.pt` and `history.json`, each carrying the
full resolved configuration. See [outputs/README.md](../outputs/README.md).

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
training-objective MSE stays comparable with the manuscript's.

## Defaults

Architecture from the Experiments section of the manuscript: 4 layers, modes
`(12, 12)`, 64 hidden channels, 80:20 split. Adam with `weight_decay = 1e-4`.

The learning rate follows a two-phase geometric warmup-then-decay: `1e-4` at
epoch 0, up to `1e-2` by `15%` of the way through the run (`warmup_fraction`),
back down to `1e-6` at the last epoch — `500` by default. Each phase is
geometric (`lr = a * r**epoch`), so the lr-vs-epoch curve has positive
curvature on both legs — accelerating into the peak, decelerating out of it —
rather than moving in straight lines. Override with `--lr-start`, `--lr-peak`,
`--lr-end` and `--warmup-fraction`.

The split is stratified by family, so both families appear in the same
proportion in training and validation.

All calculations run in `float64`/`complex128`, matching the precision the
samples are generated in (`TensorSpec.dtype`). `train.py` sets
`torch.set_default_dtype` before the model is built, which is also what every
`nn.Linear`/`nn.Conv` layer the FNO creates picks up automatically; the
vendored `neuralop`'s spectral-convolution weights and FFT buffers, which
otherwise hardcode `complex64`, are patched in
`neuraloperator/neuralop/layers/spectral_convolution.py` to follow the same
global default instead. Double precision costs several times the runtime and
twice the memory of `float32` on GPU — worth knowing if training gets slow.

## Reading the loss

The training objective (backpropagated each step) is element-wise MSE on raw
`β`. It is not what gets reported, because `|β|` spans about twelve orders of
magnitude across the grid, so **a model predicting zero everywhere already
scores a small MSE** — it would look deceptively good. Every run instead
reports **relative L2** (`||pred - target|| / ||target||`, scale-free and not
gameable this way) for both train and validation, at every epoch and in the
run summary. The best checkpoint (`best.pt`) is the one with the lowest
validation relative L2, not the lowest MSE.
