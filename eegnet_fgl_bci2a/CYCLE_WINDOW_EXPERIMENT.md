# Mu-cycle-indexed short-window experiment

This experiment tests whether the chosen 2 s observation is a useful
accuracy/time tradeoff on BCI IV-2a and IV-2b. It does not assume that a
particular number of oscillations is biologically optimal. A nominal 10 Hz
mu reference converts 5, 10, 15, 20, 25, 30, and 40 cycles to 0.5, 1,
1.5, 2, 2.5, 3, and 4 s. The summary also records nominal cycles at a
20 Hz beta reference and at the lower edge of each tested band.

All runs reuse `train_student_riemann_guided.py` with a Compact EEGNet
student. `ce` sets both teacher losses to zero, and `riemann_spatial` uses
the 4 s OAS/TangentSpace/LogReg Riemann teacher with Riemann KD and spatial
structure weights 0.1 and 0.2. No neural EEGNet teacher is used here.
Each band and window gets its own student training run. Data is split by
session; normalization fits on the training split. The training-session
validation split selects the checkpoint. The held-out test session only
provides the final student metrics; the diagnostic 4 s Riemann teacher
test evaluation is disabled in this runner.

This is an **offline, cue-aligned observation-window comparison**. MOABB's
offline bandpass preprocessing does not establish a causal, real-time
end-to-end response latency. Report a nominal observation endpoint and
do not equate it with measured online latency.

On the Linux server, run from `eegnet_fgl_bci2a`. The existing BNCI cache
is under `data/MNE-bnci-data`, so use `--data_dir data`. The runner sets
`MNE_DATA` and `MNE_DATASETS_BNCI_PATH` for each training subprocess.
Pass `--require_cached_data` to stop before training if any requested
subject's train/evaluation MAT file is missing.

Smoke test, subject 1, seed 42, 1 s and 2 s, three bands:

```bash
CUDA_VISIBLE_DEVICES=0 python run_cycle_window_ablation.py \
  --datasets bci2a bci2b --subjects 1 --seeds 42 \
  --bands full mu beta --methods ce \
  --mu_cycles 10 20 --data_dir data --require_cached_data --device cuda \
  --output_dir results/cycle_window_smoke
```

First complete the fixed 8-32 Hz CE time curve. It comprises 630 jobs
(2 datasets x 9 subjects x 5 seeds x 7 windows), so budget accordingly:

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u run_cycle_window_ablation.py \
  --datasets bci2a bci2b --subjects 1 2 3 4 5 6 7 8 9 \
  --seeds 42 2024 3407 7 123 --bands full --methods ce \
  --mu_cycles 5 10 15 20 25 30 40 \
  --data_dir data --require_cached_data --epochs 300 --patience 100 --device cuda \
  --output_dir results/cycle_window_main \
  > cycle_window_main.out 2>&1 &
```

The command can be repeated after interruption. Completed subject/seed/
window rows are skipped. To expand the same root with the teacher-guided
arm, use the identical command with `--methods ce riemann_spatial`.
To run the narrow bands too, use `--bands full mu beta`. Both expansions
cost substantial additional training time. The runner refuses to mix
different training hyperparameters within one arm's result directory.

Outputs under `--output_dir`:

- `subject_results.csv`: one held-out test row per completed run.
- `seed_summary.csv`: mean across subjects for each seed; `complete=1`
  means all requested subjects were present.
- `summary.csv`: mean and sample SD across **complete** seeds only.
- `dataset/band/method/logs/`: one log per training run.
- `dataset/band/method/run_config.json`: settings used for resume checks.

For a report, plot `mean_accuracy` against `window_s` separately for
each dataset and band. Compare 2 s with neighboring 1.5/2.5/3 s windows
and with 4 s. The cycle labels are descriptive; any window selection
must be made in training-session validation, not after inspecting test
results.
