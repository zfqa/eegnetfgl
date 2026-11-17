# PVN-LBRL BCI IV-2a method reproduction

Reference: Luo et al., *Pattern Recognition* 179 (2026), 113503,
DOI 10.1016/j.patcog.2026.113503. This runner targets the IV-2a PVN-LBRL
arm, not the paper's PhysioNet experiments or external baselines. The author
implementation is not available to verify equivalence. This is a reconstruction
of the published method, not a verified reproduction of the reported results.

## Step-by-step correspondence

| Paper | Implementation |
| --- | --- |
| Sec. 4.1 | Within-subject session split, 1-48 Hz, and 4 s trials in microvolts. The default fits on the entire training session and evaluates once on the separate test session. |
| Sec. 3.3, Fig. 2, Eq. 4 | Spatial projection (16 filters), shared SAME temporal filters (24 filters) applied to each spatial component independently, layer normalization over their joint features, spatial mixing into 48 channels, second layer normalization, per-point class logits, then hard temporal voting for offline prediction. |
| Sec. 3.3, Fig. 1, Eq. 5 | One slice at a time: PVN, linear projection, previous action/reward, 256-unit GRU, belief decoder with temporal average, and dueling Q head. |
| Sec. 3.4, Eqs. 6-8, Sec. 4.2 | PVN cross-entropy pretraining (300 epochs), then frozen PVN and uniformly sampled HALT/WAIT rollouts for belief NLL (300 epochs). The HALT slice is included; later slices are excluded from the objective. Both stages use AdamW at 1e-3. |
| Algorithm 1, Sec. 4.2 | Eight epsilon-greedy actor policies, prioritized replay capped at one million transitions, and separate uniform slice-label replay for the belief learner. Q receives 50,000 optimizer updates and belief receives 25 rounds of 20 epochs, both with AdamW at 1e-4. |
| Sec. 3.1, Eqs. 1-3 | HALT emits the belief argmax and has reward +1/-1. WAIT has reward 0 before the limit. At the last complete slice, WAIT terminates with -1 as the equation's `else` case and does not bootstrap. The evaluator uses the last belief as a fallback if the policy never chooses HALT. |
| Secs. 4.3, 4.5-4.7 | Accuracy, mean observation time and ITR (Eq. 9), plus local F1/kappa diagnostics. The Table 4 settings use gamma 0.1 with slice sizes 40/60/80/100; the conservative analysis uses 100 samples and gamma 0.997. |

## Reproduction choices and limitations

The MOABB array loader already converts volts to microvolts. The causal BNCI
MAT loader keeps the original microvolt units. The runner therefore defaults
to `--scale 1` for both paths. Previous default runs multiplied these arrays
by an additional `1e6`; pass `--scale 1e6` only to reproduce that legacy
setting. Keep corrected results in a separate output directory.

The article does not specify the exact FIR bank, differentiable voting
objective, actor/target synchronization intervals, online epoch size, or
random seeds. This implementation uses trainable 33-tap FIR initialization,
the temporal mean of point probabilities for differentiable CE, and locally
documented schedules exposed in the runner. An online belief epoch draws
approximately one batch pass over the available slice samples, capped at
the number of full-prefix training slices.

The Q learner uses the cited R2D2 paper's transformed five-step Double-DQN
target and max/mean TD-error priority, with the PVN-LBRL paper's alpha 0.9
and beta 0.6. Instead of Atari-length replay sequences, it replays complete
EEG episodes from their initial recurrent state. Actors collect in batches
using a periodically refreshed CPU snapshot, concurrently with learner
updates after the initial collection. This is not the authors' distributed
implementation. Cached encoder features are computed per slice with the
PVN frozen; test-time encoding only processes the current slice. No
minimum observation time or additional WAIT reward has been introduced.

### Exploration ambiguity and controlled correction

The PVN-LBRL article literally prints `epsilon_i = 0.4 ** (7 + i)` for
one-based actor index `i`. With eight actors this is approximately
`6.55e-4 ... 1.07e-6`. In the first complete local run this produced almost
no exploration: 50,000 episodes contained only 50,794 slices and the policy
halted every test trial at 0.16 s. The cited R2D2 method instead uses
`epsilon_i = 0.4 ** (1 + 7 i / (N - 1))` for zero-based `i`, spanning
`0.4 ... 6.55e-4` with eight actors.

The runner therefore exposes both auditable modes:

- `--epsilon_schedule paper_literal`: the formula printed by PVN-LBRL;
- `--epsilon_schedule r2d2_standard`: the schedule from the cited R2D2
  algorithm and the default controlled correction.

Because PVN-LBRL does not report when its Q learner begins relative to actor
collection, the corrected mode also defaults to
`--replay_warmup_episodes 1024`. Warm-up uses uniformly random actions
(`--replay_warmup_epsilon 1.0`) so that a random initial Q-head preference
cannot fill replay almost exclusively with one-slice HALT episodes. This is an
explicit implementation choice, not a claim about the unavailable author
code. Set the episode count to zero for an immediate learner start, or set the
warm-up epsilon to the ordinary actor schedule only for a collapse-control
experiment. Logs report warm-up/recent mean slices, first-slice halt rate,
greedy first-slice HALT rate, and the mean Q(WAIT)-Q(HALT) margin.

The final iterate is used, with no validation-ITR checkpoint selection.
`--no-full_training_session` is available for an independent hold-out
diagnostic, not for a direct paper-protocol comparison. This version has a
different PVN architecture; old checkpoints and result directories cannot
be reused.

Paper comparisons at 60 and 80 samples use `floor(1000 / slice_samples)`
complete slices; the incomplete trial tail is not observed by the policy.
The default MOABB bandpass uses offline zero-phase filtering; consequently,
its early-halt lengths are **nominal offline observation times**, not a
verified causal online system latency. The paper specifies the passband, not
an exact filter implementation. For a causal control, run the separate
`--preprocessing_mode causal_mat` arm and do not pool the results.

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
  --belief_updates 1 --online_belief_epochs 1 \
  --epsilon_schedule r2d2_standard --replay_warmup_episodes 16 \
  --replay_batch_size 4 --validate_every 2 --device cuda \
  --output_dir results/pvn_lbrl_algorithm1_smoke
```

First scientific pilot, one subject and one seed:

```bash
CUDA_VISIBLE_DEVICES=3 nohup python -u run_pvn_lbrl_ablation.py \
  --subjects 1 --seeds 42 --slice_samples 40 --gamma 0.1 \
  --data_dir data --epochs_pvn 300 --epochs_belief 300 \
  --rl_steps 50000 --belief_updates 25 --online_belief_epochs 20 \
  --epsilon_schedule r2d2_standard --replay_warmup_episodes 1024 \
  --replay_warmup_epsilon 1.0 --device cuda \
  --output_dir results/pvn_lbrl_r2d2_corrected_pilot_s1 \
  > pvn_lbrl_r2d2_corrected_pilot_s1.out 2>&1 &
```

After reviewing that pilot, rerun 9 subjects and repeated seeds in a new
output directory for each slice size and gamma. Use `--slice_samples 100
--gamma 0.997` for the conservative setting. The paper mentions 10
independent repetitions for statistical comparisons; its exact seeds are
not reported, so explicitly record the local seed list. The runner writes
`subject_results.csv`, `seed_summary.csv`,
`summary.csv`, model checkpoints, and `run_config.json`. Re-running an
identical command skips completed subject/seed rows; it rejects a changed
configuration in the same output directory.

After a run, compare train/test prefix accuracy and stopping behavior without
using evaluation-session labels for fitting:

```bash
python -u diagnose_pvn_policy.py \
  --checkpoint results/pvn_lbrl_balanced_warmup_s1/subject_1_seed_42_final.pth \
  --data_dir data --device cuda --splits train test \
  --output_dir results/pvn_lbrl_balanced_warmup_s1/diagnostics
```

To locate a first-slice generalization gap, run an independent audit with a
validation split inside the training session. This reports the full-window PVN
vote and the GRU fixed-prefix accuracy before RL and after 500 Q-only updates.
The belief learner is disabled to isolate the Q phase; the held-out test session
is not scored and no final policy checkpoint is emitted.

```bash
CUDA_VISIBLE_DEVICES=0 python -u run_pvn_lbrl_ablation.py \
  --subjects 1 --seeds 42 --audit_prefixes --no-full_training_session \
  --slice_samples 40 --gamma 0.1 --rl_steps 500 --belief_updates 0 \
  --data_dir data --device cuda --output_dir results/pvn_lbrl_prefix_audit_s1_q500
```

Use `--rl_steps 0` with a separate output directory to stop after pretraining.
The audit writes `subject_1_seed_42_prefix_audit.csv` and prints the first and
last prefix accuracies. Use the validation split, not the prior test scores, to
compare changes to the encoder, belief learner, or stopping policy.

For a supervised A/B comparison without RL, add `--compare_prefix_training`.
Both arms share one frozen PVN, the initial belief-model weights, batch order,
and the number of supervised prefixes per trial. The balanced arm samples
those prefix lengths uniformly without replacement. This arm is a local
modification, not part of the paper-guided reproduction.

```bash
CUDA_VISIBLE_DEVICES=0 python -u run_pvn_lbrl_ablation.py \
  --subjects 1 --seeds 42 --audit_prefixes --compare_prefix_training \
  --no-full_training_session --slice_samples 40 --rl_steps 0 --belief_updates 0 \
  --data_dir data --device cuda --output_dir results/pvn_lbrl_balanced_prefix_ab_s1
```

The predeclared primary diagnostic is validation accuracy at the 12th slice
(1.92 s), the longest complete prefix at or below 2 s for 40-sample slices.
The comparison CSV also records `pvn_matched_prefix_accuracy` at that row:
the same frozen full-window-pretrained PVN directly votes on the cropped
480-sample input, without a GRU or later samples supplied to the encoder.
This is an inference-only control, not a separately trained 1.92 s PVN.
The same row records `pvn_segmented_prefix_accuracy`: pool point-level votes
from the exact slice logits supplied to the GRU, excluding all later slices.
The contiguous/segmented comparison isolates the difference between encoding
one long prefix and encoding individual slices. The segmented vote and GRU
share the same slice features but aggregate them differently.

Use `--seeds 42 2024 3407 7 123 --scale 1` with a fresh output directory
to repeat the corrected-scale experiment. The default MOABB zero-phase filter
is unchanged, so observation times remain nominal offline times.

For a cumulative PVN baseline without GRU or RL, replace
`--compare_prefix_training` with `--audit_pvn_prefixes` and use a new
output directory. This fits the PVN once per subject/seed, then applies the
same frozen PVN directly to each cumulative prefix up to 2 s. At 40 samples
per slice, the output `subject_*_seed_*_pvn_prefix_curve.csv` contains 12
points per split, including 0.48, 0.96, 1.44, and 1.92 s.
The audit scores training/validation only and does not select an exit threshold.
