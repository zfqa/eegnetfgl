# PVN-LBRL paper-guided reproduction

Reference: Luo et al., *Pattern Recognition* 179 (2026), 113503,
DOI 10.1016/j.patcog.2026.113503. The author implementation was not publicly
available when this code was written. This is **not a bitwise or source-code
reproduction** and must not be called a verified reproduction of the reported
accuracy, average prediction length, or ITR.

## Implemented from the paper

- BCI IV-2a session split, train-session-only stratified validation, 1-48 Hz,
  4 s MI trial, and voltage scaling by 1e6. Evaluation session is final test.
- PVN: spatial convolution (16 filters), same-padded temporal convolution
  (24 filters), two layer-normalization modules, 48-channel 1x1 convolution,
  and class-logit convolution. Temporal mean of per-sample class probabilities
  is the differentiable vote used for CE pretraining.
- The incoming EEG slice is encoded once. Flattened class-time features are
  projected to 256 dimensions, concatenated with previous action and reward,
  and accumulated by a 256-unit GRU cell.
- Belief decoder: linear layer, class-time reshape, 1x1 Bf-Conv, softmax,
  temporal average. Dueling Q network: independent value and advantage MLPs;
  actions are HALT=0 and WAIT=1.
- PVN whole-trial CE pretraining, then frozen-PVN per-slice belief NLL
  pretraining, then epsilon-greedy prioritized recurrent episode replay with
  TD learning. Eight epsilon policies are simulated in one process. Training
  rewards use **training-session labels only**. Policy checkpoints are chosen
  by validation ITR. Test labels are read only for final metrics.
- At inference, each trial is processed slice by slice and stops at HALT or
  the last complete slice. Neither teachers nor future slices are accessed.

## Reproduction choices and limitations

The article does not publish exact FIR coefficients, its point-voting
differentiable CE surrogate, every replay/synchronization setting, random
seeds, or source code. This implementation uses 33-tap trainable FIR-initialized
temporal kernels, channel-wise LN, full-episode prioritized replay with
importance weights, Double-DQN targets, and one-process actor emulation. It is
**R2D2-style, not the paper's exact distributed R2D2 implementation**. The
replay stores frozen PVN features, as in the paper's Algorithm 1. Encoder
precomputation is valid because each slice is encoded independently; the
test-time path still encodes only the current slice.

Paper comparisons at 60 and 80 samples use `floor(1000 / slice_samples)`
complete slices; the incomplete trial tail is not observed by the policy.
The default MOABB bandpass uses offline zero-phase filtering; consequently,
its early-halt lengths are **nominal offline observation times**, not a
verified causal online system latency. For a leakage-free control, run the
separate `--preprocessing_mode causal_mat` arm and do not mix it with the
paper-aligned zero-phase results.

The independent PVN-LBRL run has no teacher distillation. A later
PVN-LBRL-plus-teachers experiment would be a new method, not a reproduction
of the published PVN-LBRL results. Do not compare its accuracy directly with
8-32 Hz Compact EEGNet results because preprocessing differs.

## Run on the Linux server

Run from `eegnet_fgl_bci2a`; `data` should contain `MNE-bnci-data`.
Functional smoke test (not a scientific result):

```bash
CUDA_VISIBLE_DEVICES=3 python -u run_pvn_lbrl_ablation.py \
  --subjects 1 --seeds 42 --slice_samples 40 --gamma 0.1 \
  --data_dir data --epochs_pvn 1 --epochs_belief 1 --rl_steps 4 \
  --replay_batch_size 1 --validate_every 2 --device cuda \
  --output_dir results/pvn_lbrl_smoke
```

First scientific pilot, one subject and one seed:

```bash
CUDA_VISIBLE_DEVICES=3 nohup python -u run_pvn_lbrl_ablation.py \
  --subjects 1 --seeds 42 --slice_samples 40 --gamma 0.1 \
  --data_dir data --epochs_pvn 300 --epochs_belief 300 \
  --rl_steps 50000 --device cuda \
  --output_dir results/pvn_lbrl_pilot_s1 > pvn_lbrl_pilot_s1.out 2>&1 &
```

After reviewing that pilot, rerun 9 subjects and repeated seeds in a new
output directory. The runner writes `subject_results.csv`, `seed_summary.csv`,
`summary.csv`, model checkpoints, and `run_config.json`. Re-running an
identical command skips completed subject/seed rows; it rejects a changed
configuration in the same output directory.
