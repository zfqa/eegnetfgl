# Frozen classifier, disjoint policy-training diagnostic

Exploratory adaptation, NOT strict PVN-LBRL reproduction or teacher distillation.
The existing RL runner, original models, teachers and losses are unchanged.

## Contract

- IV-2a session split, MOABB zero-phase, 1-48Hz, scale1 microvolts.
- Training session: stratified80/20 gives230/58. Stratify230 again, holding out
  25% for policy training: classifier172 / policy58 / validation58. The test
  session remains288 separate trials. Saved split indices are relative to230.
- Pretrain PVN only on classifier172 full4s trials, then balanced-prefix GRU
  using only their first480 samples. Fixed300/300 epochs, no policy-set fitting.
- Freeze PVN, projection, GRU and decoder during Q updates; assert unchanged.
- Three policies share this exact classifier and observation opportunities:
  every40 points starting0.16s, ending1.92s (12 slices). This is NOT the previous
  cumulative-PVN baseline; this model encodes slices independently, then uses GRU.
- Threshold: max belief probability (uncalibrated), candidates .5,.6,.7,.8,.9,
  .95,.99 and fixed fallback. Minimize validation time with accuracy >= fixed;
  ties prefer accuracy, then larger threshold.
- Q: existing transformed five-step Double-DQN / prioritized replay; no belief
  updates. AdamW1e-4, default500 Q updates, random256-episode warmup, eight
  R2D2 epsilon values in rotation. Serial diagnostic, not distributed R2D2.
- Last slice forces HALT in both collection and inference. Reward+1/-1 by
  classification correctness. Earlier WAIT reward0, default gamma0.1.
- Validate step0 and every100 Q updates. Prefer an accuracy-feasible checkpoint
  with minimum time. If none meets fixed validation accuracy, retain highest
  accuracy Q (not a claim of test noninferiority). Exact ties retain earlier step.
- Stages are separate: pretrain for prefix diagnostics, pilot for Q fitting,
  test for independent scoring of saved locked rules. Test labels never select
  threshold, reward, epoch, checkpoint. Repeated test-guided changes are exploratory.
- Zero-phase filtering is retained by request: report nominal offline observation
  time, not causal online latency. Full4s PVN pretraining is training-only access.

## Server

Use the same parameters and directory in all three stages; only stage changes.

```bash
CUDA_VISIBLE_DEVICES=3 python -u run_pvn_disjoint_policy.py --stage pretrain \
  --subject 1 --seed 42 --data_dir data --device cuda \
  --output_dir results/pvn_disjoint_s1_seed42
```

Inspect pre_rl_prefix_accuracy.csv, especially unseen policy and validation
prefixes. Do not launch Q if the classifier remains near random throughout.

```bash
CUDA_VISIBLE_DEVICES=3 python -u run_pvn_disjoint_policy.py --stage pilot \
  --subject 1 --seed 42 --data_dir data --device cuda \
  --output_dir results/pvn_disjoint_s1_seed42
```

Pilot reuses pretrained_classifier.pth. It saves validation_results.csv,
q_validation_history.csv, threshold_validation.csv and locked_model.pth.
No test scoring occurs. Approve test only after reviewing validation results:

```bash
CUDA_VISIBLE_DEVICES=3 python -u run_pvn_disjoint_policy.py --stage test \
  --subject 1 --seed 42 --data_dir data --device cuda \
  --output_dir results/pvn_disjoint_s1_seed42
```

Test stage loads locked weights/rules without fitting. Outputs test_results.csv
(accuracy, macroF1, kappa, confusion matrices, mean observation time) and
per-trial predictions/times for fixed, threshold and Q. No performance claimed
until the server run is completed.
