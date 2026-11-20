# Frozen-PVN prefix-sampling A/B audit

This diagnostic reuses only the PVN from the staged pretraining checkpoint.
The source's trained GRU is deliberately discarded. A fresh GRU/projection/
decoder/Q initialization is shared by both arms. Q is never trained or used
to stop. No evaluation-session input or labels are scored.

## Controls

- Source must be joint-learning pretraining: IV-2a, MOABB zero-phase 1-48Hz,
  microvolts scale1, 40-point slices, train230/validation58 (no extra policy split).
- Same PVN, GRU initialization, shuffle/rollout RNG and optimizer settings.
- A uses random-HALT supervision masks. B redistributes exactly the same
  number of supervised targets per trial to uniformly sampled prefix positions.
  B uses an independent permutation generator so it does not alter A's shuffle
  or rollout RNG stream. Per-batch/per-trial target budgets are asserted equal.
- Balanced masks need not select contiguous positions: the GRU still consumes
  all earlier slices; the mask only chooses which prefix predictions enter loss.
- Shared AdamW1e-3, 300 epochs; no validation-best epoch/window selection.
- Frozen PVN weights are asserted unchanged. Observed mask counts are saved.
- Every prefix from0.16 to4s records GRU accuracy, cumulative point-vote accuracy
  from the exact independent-slice logits supplied to GRU, and contiguous-prefix
  PVN voting accuracy. The two PVN controls must coincide across arms.
- Summary endpoints are predeclared1.92s and4s, not the best validation window.
- This is a new controlled adaptation, not the paper's unmodified algorithm.
  The prior result informed this diagnostic; it is exploratory. Do not tune
  against the already viewed evaluation session. Times remain nominal offline.

## Server

Required new file: run_pvn_prefix_sampling_ab.py. Also update
pvn_lbrl_training.py (optional supervision-count recording, defaults unchanged).
Keep the source directory untouched. This command refits two GRUs, not PVN.

```bash
(
  set -e
  cd /workspace/zfq/eegnetfgl/eegnet_fgl_bci2a
  mkdir -p logs
  CUDA_VISIBLE_DEVICES=3 nohup /opt/conda/envs/eeg_fgl/bin/python -u \
    run_pvn_prefix_sampling_ab.py \
    --source_dir results/pvn_joint_s1_seed42_literal \
    --output_dir results/pvn_prefix_sampling_ab_s1_seed42 \
    --data_dir data --device cuda --epochs 300 --batch_size 32 \
    > logs/pvn_prefix_sampling_ab_s1_seed42.out 2>&1 &
  echo "PID: $!"
)
```

Progress: `tail -n40 -F logs/pvn_prefix_sampling_ab_s1_seed42.out`.
It prints an ARM line, then curves after each300-epoch arm; there is no per-epoch
log. Outputs are prefix_curves.csv, supervision_counts.csv,
validation_summary.csv, shared_initial.pth and one checkpoint per arm, plus
run_config.json with source settings and a SHA256 of the source checkpoint.
The output directory must be new/empty; no partial resume or overwriting.

```bash
/opt/conda/envs/eeg_fgl/bin/python - <<'PY'
import pandas as pd
root = 'results/pvn_prefix_sampling_ab_s1_seed42/'
print(pd.read_csv(root + 'validation_summary.csv').to_string(index=False))
print('\nActual supervision counts:')
print(pd.read_csv(root + 'supervision_counts.csv').pivot(
    index='observation_s', columns='arm', values='supervised_targets').to_string())
PY
```

If B improves but remains below segmented PVN, aggregation/overfitting is still
an issue. If segmented PVN itself is much worse than contiguous PVN, isolate
slice encoding/context loss before attributing all losses to GRU or sampling.
Do not proceed to Q just because training accuracy is high.
