# outputs/

Everything a training run produces. Nothing here is source; the whole
directory is reproducible from `data/` and `model/`.

```
outputs/
  runs/<run_name>/best.pt        best-validation checkpoint (weights + resolved config + normaliser state)
  runs/<run_name>/history.json   per-epoch losses, learning rate, timings, and the run summary
  plots/                         figures: loss curves, predicted vs. true beta, error maps
```

`model/train.py` writes to `outputs/runs/<run_name>/` by default; override with
`--out-dir` and `--run-name`. Contents are git-ignored except this file.
