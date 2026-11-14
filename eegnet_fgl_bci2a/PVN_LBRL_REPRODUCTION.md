# PVN-LBRL BCI IV-2a method reproduction

Reference: Luo et al., *Pattern Recognition* 179 (2026), 113503,
DOI 10.1016/j.patcog.2026.113503. This runner targets the IV-2a PVN-LBRL
arm, not the paper's PhysioNet experiments or external baselines. The author
implementation is not available to verify equivalence. This is a reconstruction
of the published method, not a verified reproduction of the reported results.

## Step-by-step correspondence

| Paper | Implementation |
| --- | --- |
| Sec. 4.1 | Within-subject session split, 1-48 Hz, 4 s trials and voltage scaling by 1e6. The default fits on the entire training session and evaluates once on the separate test session. |
| Sec. 3.3, Fig. 2, Eq. 4 | Spatial projection (16 filters), shared SAME temporal filters (24 filters) applied to each spatial component independently, layer normalization over their joint features, spatial mixing into 48 channels, second layer normalization, per-point class logits, then hard temporal voting for offline prediction. |
| Sec. 3.3, Fig. 1, Eq. 5 | One slice at a time: PVN, linear projection, previous action/reward, 256-unit GRU, belief decoder with temporal average, and dueling Q head. |
| Sec. 3.4, Eqs. 6-8, Sec. 4.2 | PVN cross-entropy pretraining (300 epochs), then frozen PVN and uniformly sampled HALT/WAIT rollouts for belief NLL (300 epochs). The HALT slice is included; later slices are excluded from the objective. Both stages use AdamW at 1e-3. |
| Algorithm 1, Sec. 4.2 | Eight epsilon-greedy actor policies, exploration rates `0.4 ** (7 + i)` for one-based `i`, prioritized replay capped at one million transitions, and separate uniform slice-label replay for the belief learner. Q receives 50,000 optimizer updates and belief receives 25 rounds of 20 epochs, both with AdamW at 1e-4. |
| Sec. 3.1, Eqs. 1-3 | HALT emits the belief argmax and has reward +1/-1. WAIT has reward 0 for an observed slice. The last complete slice terminates the episode; a last-slice WAIT is not relabeled as HALT and does not bootstrap. The evaluator uses the last belief as a fallback if the policy never chooses HALT. |
| Secs. 4.3, 4.5-4.7 | Accuracy, mean observation time and ITR (Eq. 9), plus local F1/kappa diagnostics. The Table 4 settings use gamma 0.1 with slice sizes 40/60/80/100; the conservative analysis uses 100 samples and gamma 0.997. |

## Reproduction choices and limitations

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
  --replay_batch_size 4 --validate_every 2 --device cuda \
  --output_dir results/pvn_lbrl_algorithm1_smoke
```

First scientific pilot, one subject and one seed:

```bash
CUDA_VISIBLE_DEVICES=3 nohup python -u run_pvn_lbrl_ablation.py \
  --subjects 1 --seeds 42 --slice_samples 40 --gamma 0.1 \
  --data_dir data --epochs_pvn 300 --epochs_belief 300 \
  --rl_steps 50000 --belief_updates 25 --online_belief_epochs 20 --device cuda \
  --output_dir results/pvn_lbrl_algorithm1_pilot_s1 > pvn_lbrl_algorithm1_pilot_s1.out 2>&1 &
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
