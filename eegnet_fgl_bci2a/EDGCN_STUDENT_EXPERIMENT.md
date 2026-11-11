# Two-second EDGCN student ablation

This experiment replaces only the student encoder. The 4 s Compact EEGNet
teacher and 4 s OAS/TangentSpace/LogReg teacher, data split, and KD weights
remain in `train_student_dual_kd.py`. The student sees exactly 500 samples
(0-2 s at 250 Hz). The evaluation session is never used to fit the teachers,
preprocessing statistics, or student. At test time only the EDGCN student is
called.

The implementation follows the modules and equations in Shen et al.,
"EDGCN: An embedding-driven fusion framework for heterogeneity-aware motor
imagery decoding", *Information Fusion* 131 (2026), 104170,
DOI 10.1016/j.inffus.2026.104170: Ghost/local multiscale convolutions,
MRTE, PLV-based SASE, Chebyshev propagation, embedding-generated temporal
and spatial kernels, and separately trainable orthogonality weights. This is
a **2 s adaptation, not an exact reproduction** of the paper's reported
4.5 s/3 s benchmark. The paper does not fully specify every convolution,
pooling, PSD, PLV threshold or MLP constant in the main text. Here the
explicit defaults are in `models/edgcn.py` and the CLI. In particular:

- The 8-32 Hz input band is split into 8-12, 12-20, and 20-32 Hz PSD bands.
- Three temporal branches use 20/60/80-sample kernels, max-pool by 2, crop
  to equal length, then fuse with a 15-sample kernel and average-pool by 4.
- The SASE regional prior follows the standard 22-channel BNCI 2a order;
  all three 2b channels form one local region. Global PLV threshold is 0.5.
- The paper averages adjacency over a batch. This implementation instead
  keeps each trial's adjacency independent so a test prediction cannot
  change when other test trials or batch size change.
- The default temporal/spatial orthogonality coefficients are both 0.01 at
  initialization and are learned separately thereafter.

Do not compare the EDGCN paper's published accuracies directly with this
2 s experiment: observation window, input band, and protocol differ.
Also, the default MOABB zero-phase filter is offline; it does not establish
causal real-time inference. Use `--preprocessing_mode causal_mat` for a
future-leakage-safe sensitivity experiment and train fresh teachers with
matching causal preprocessing before using the dual arm.

From `eegnet_fgl_bci2a` on the server, with `data/MNE-bnci-data` present:

```bash
conda activate eeg_fgl
CUDA_VISIBLE_DEVICES=3 python -u run_edgcn_student_ablation.py \
  --dataset bci2a --subjects 1 --seeds 42 \
  --variants compact edgcn --methods ce dual \
  --data_dir data \
  --teacher_root results/teacher_window_matched_kd_offline_bci2a/teachers/tw4 \
  --epochs 3 --patience 2 --device cuda \
  --output_dir results/edgcn_smoke
```

Full 9-subject x 5-seed comparison (180 student runs):

```bash
CUDA_VISIBLE_DEVICES=3 nohup python -u run_edgcn_student_ablation.py \
  --dataset bci2a --subjects 1 2 3 4 5 6 7 8 9 \
  --seeds 42 2024 3407 7 123 \
  --variants compact edgcn --methods ce dual \
  --data_dir data \
  --teacher_root results/teacher_window_matched_kd_offline_bci2a/teachers/tw4 \
  --epochs 300 --patience 100 --batch_size 32 --device cuda \
  --output_dir results/edgcn_student_bci2a \
  > edgcn_student_bci2a.out 2>&1 &
```

The runner resumes by checking config-matched rows. `subject_results.csv`
contains one test row per run; `seed_summary.csv` averages nine subjects
within a seed; `summary.csv` averages only complete seeds. Check the
student parameter count in each training log, because EDGCN is substantially
larger than Compact EEGNet. A gain from EDGCN+KD over Compact CE alone
does not identify whether graph encoding or KD caused it; use all four arms.
The runner requires a teacher checkpoint for each subject and seed. By default
it rejects checkpoints without matching preprocessing metadata. If your 4 s
teacher checkpoints live elsewhere, point `--teacher_root` to the directory
that contains `seed_42/subject_1_teacher_best.pth` and analogous files.
