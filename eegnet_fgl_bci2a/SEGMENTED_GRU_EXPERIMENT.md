# Fixed 2 s EEGNet-GRU evidence accumulation experiment

This experiment adapts the latent belief update in Luo et al., *Pattern
Recognition* 179 (2026), 113503, DOI: 10.1016/j.patcog.2026.113503.
It is **not** a full reproduction of PVN-LBRL. The encoder is the existing
Compact EEGNet instead of PVN, and there is no Q-network, R2D2 training, or
learned WAIT/HALT policy. Accordingly, these results must not be compared
with the paper's online ITR or 0.22 s decision length.

At 250 Hz, the fixed 0-2 s student input contains 500 samples. We split it
into ten consecutive, non-overlapping 50-sample slices (0.20 s each), as
requested for this ablation. The same Compact EEGNet feature extractor
encodes every slice. Its features are projected to 256 dimensions and
concatenated with the previous action and reward before the GRU belief
update. Because this arm always observes the full 2 s, previous actions
are WAIT and rewards are zero. A belief decoder yields a categorical
distribution at every slice by softmax and temporal averaging. Only the
tenth belief is used for final classification and dual-teacher KD.

Training follows the paper's offline stage ordering: first pretrain the
slice encoder with CE, then freeze it and pretrain the GRU plus belief
decoder with CE at all ten slices. The final CE or dual-KD phase updates
the GRU and decoder, while the slice encoder remains frozen. Unlike the
paper, encoder pretraining uses 50-sample slices because Compact EEGNet's
feature dimension is fixed for the input length. This is an adaptation,
not a claim of identical PVN pretraining.

All four arms use the same session split, 8-32 Hz preprocessing, training-
only normalization, seed, and 2 s student input. The dual-KD arms use
4 s EEGNet and OAS/TangentSpace/LogReg teachers with weights 0.4+0.4.
The EEGNet teacher checkpoint is loaded by subject and seed. The
evaluation session is used only for final student testing. MOABB's
zero-phase filtering is offline and does not establish causal online
latency; use `--preprocessing_mode causal_mat` for a separate leakage-free
control rather than mixing those results into this matrix.

Run from `eegnet_fgl_bci2a` on the Linux server. Use `--data_dir data` if
the cached MAT files are under `data/MNE-bnci-data`. Ensure all 45
pretrained teacher checkpoints exist under the selected teacher root.
The runner stops before training if a checkpoint is missing.

```bash
CUDA_VISIBLE_DEVICES=3 nohup python -u run_segmented_gru_ablation.py \
  --subjects 1 2 3 4 5 6 7 8 9 \
  --seeds 42 2024 3407 7 123 \
  --variants compact compact_gru --methods ce dual \
  --teacher_root results/unified_teachers/bci2a \
  --data_dir data --device cuda \
  --epochs 300 --patience 100 \
  --gru_encoder_pretrain_epochs 300 --gru_belief_pretrain_epochs 300 \
  --output_dir results/segmented_gru_ablation \
  > segmented_gru_ablation.out 2>&1 &
```

For a short functional test before the full matrix, add `--subjects 1
--seeds 42 --epochs 1 --gru_encoder_pretrain_epochs 1
--gru_belief_pretrain_epochs 1` and use a separate output directory.
Those results have no scientific interpretation. The runner resumes
completed rows and writes `subject_results.csv`, `seed_summary.csv`, and
`summary.csv` under the output directory. Each job also has its own log.
Do not combine a changed architecture or pretraining schedule into the
same output directory.

The 256-hidden-unit GRU student has about 453,516 parameters for IV-2a,
versus about 2,420 for the original Compact 2 s student. Report this
cost when discussing accuracy or deployment. Teacher networks still do
not run at test time.
