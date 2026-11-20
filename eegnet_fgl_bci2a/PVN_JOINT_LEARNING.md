# Staged PVN + GRU + Q joint-learning experiment

This is a paper-guided reconstruction, not verified author-code equivalence.
It does not use teachers or KD. The disjoint/frozen diagnostic is unchanged.

## What changes relative to the disjoint diagnostic

- Use up to the full 4 s trial (25 incoming 40-point slices), not 1.92 s.
- Train the PVN for 300 epochs, then freeze it. Train the projection, GRU and
  belief decoder for 300 epochs with the random-HALT prefix objective.
- During RL, Q TD loss updates only the Q head. A separate supervised belief
  learner updates the projection, GRU and decoder (25 rounds x 20 epochs).
- By default train on 230 trials, hold out 58 for diagnostics, and leave the
  separate 288-trial evaluation session untouched until the test stage.
  There is no extra classifier172/policy58 partition. This validation holdout
  is a local diagnostic adaptation. `--full_training_session` instead uses all
  288 training trials as in the paper, with no validation diagnostics. Do not
  mix these protocols or choose between them using test scores.
- Report MOABB zero-phase 1-48 Hz times as nominal offline observation time,
  not demonstrated causal online latency. The loader already returns microvolts.

Paper ambiguities remain: initial hidden/action values, exact FIR coefficients,
differentiable voting loss and distributed synchronization/epoch size. Existing
local reconstruction choices are documented in PVN_LBRL_REPRODUCTION.md.
The default exploration is the printed `paper_literal` formula, with zero
warm-up. The known alternative `--epsilon_schedule r2d2_standard
--replay_warmup_episodes 1024` must be a separately named controlled variant;
do not silently substitute it or claim it is the authors' implementation.

## Server: start with pretraining only

Use the same settings and output directory in all three stages, changing only
`--stage`. These commands use the validation-holdout diagnostic protocol.

```bash
cd /workspace/zfq/eegnetfgl/eegnet_fgl_bci2a
mkdir -p logs
CUDA_VISIBLE_DEVICES=3 nohup /opt/conda/envs/eeg_fgl/bin/python -u \
  run_pvn_joint_learning.py --stage pretrain --subject 1 --seed 42 \
  --data_dir data --device cuda \
  --output_dir results/pvn_joint_s1_seed42_literal \
  > logs/pvn_joint_s1_seed42_pretrain.out 2>&1 &
echo "PID: $!"
```

Inspect `pre_rl_prefix_accuracy.csv`: validation should show useful late-prefix
classification before committing to RL. Do not use test results as this gate.
Pretraining is saved and reused, including RNG states, rather than refitted.

```bash
CUDA_VISIBLE_DEVICES=3 nohup /opt/conda/envs/eeg_fgl/bin/python -u \
  run_pvn_joint_learning.py --stage pilot --subject 1 --seed 42 \
  --data_dir data --device cuda \
  --output_dir results/pvn_joint_s1_seed42_literal \
  > logs/pvn_joint_s1_seed42_pilot.out 2>&1 &
echo "PID: $!"
```

Pilot performs 50,000 Q updates and the supervised belief rounds, asserts
that GRU weights actually changed, writes `post_rl_prefix_accuracy.csv`, and
saves the final iterate in `locked_model.pth`. No validation-best selection and
no test scoring. A pilot cannot resume mid-Q: a crash requires a separate new
run directory (copy the compatible pretraining checkpoint and manifest only,
not a partial pilot's output). Do not launch two stages concurrently.

Only after reviewing the diagnostics, score the locked model once:

```bash
CUDA_VISIBLE_DEVICES=3 /opt/conda/envs/eeg_fgl/bin/python -u \
  run_pvn_joint_learning.py --stage test --subject 1 --seed 42 \
  --data_dir data --device cuda \
  --output_dir results/pvn_joint_s1_seed42_literal
```

`test_results.csv` compares learned Q stopping against the same locked GRU
classifier at the last prefix. This fixed baseline is NOT whole-window PVN
and NOT the Compact EEGNet baseline from the KD paper.

Synthetic checks (no dataset or experiment training):

```bash
python -m pytest tests/test_pvn_joint_learning.py -q
```
