# 4s temporal-probability + graph-relation supervision for a2s Compact student

This is the proposed NEW distillation experiment, not a reproduction of
Shen et al.'s EDGCN scores (Information Fusion131,104170,2026).
The fixed2s Compact classifier is unchanged. No GRU, Q, dynamic window,
Riemann classifier or old raw-covariance Spatial loss is added.

## Architecture and source correspondence

1. Temporal teacher:4s EEG -> Ghost -> temporal conv kernels20/60/80 (10 maps
   each) -> trim to common length -> max pool -> concatenate30 maps -> fusion
   conv1x15 -> BN/ELU/pool/dropout. This follows Sec3.1/Fig2(a)'s structure.
   A NEW channel-preserving4-bin temporal pooling + linear classification head
   is trained with CE. Kernel widths are samples, NOT observation durations.
2. Graph teacher:4s full classifier using the existing local EDGCN reconstruction
   (Ghost/multiscale, MRTE, SASE, Chebyshev propagation, embedding-generated
   banks, orthogonality). Train full classification first. After freezing, read
   the spatial-embedding-driven branch's node features AFTER graph propagation
   and generated-weight mapping, BEFORE node averaging/classifier. Its class
   probabilities are NOT a student target. It is not a purely spatial model.
3. Student: fixed0-2s Compact EEGNet. For the auxiliary loss, retain electrode
   identity after temporal conv/BN and before depthwise spatial conv. Per filter
   and electrode, split the500 positions into4 bins of125, subtract each bin's
   temporal mean, and compute log(mean(square)+1e-6). These32-dimensional node
   features are a local adaptation, not a paper-prescribed student design.
4. Normalize each node's feature vector with L2 norm and form node cosine
   matrices R=ZZ^T. Minimize teacher/student off-diagonal MSE. Teacher node
   width16 and student width32 need not match, but electrode order must match.
   This is feature-cosine relation KD, NOT Pearson correlation, covariance,
   PLV matching or a Riemannian geodesic. No trainable projection can absorb
   this loss. It directly updates temporal conv/BN, not later spatial kernels.
5. Loss: CE + lambda_time*T^2*KL(p_teacher||p_student) + beta*relation_MSE.
   Default lambda_time=.4, beta=.2, T=1. These are starting settings, not proven
   optimal coefficients and not equivalent to the old double-KL weights.
   Temperature calibration fits ONLY the selected temporal teacher's validation
   logits/labels; it is separate from the KD temperature.

Unspecified/source differences are explicit: Ghost kernels5/3 and widths10/20,
pool sizes2/4, temporal head4 bins; the inherited graph frontend uses16 fused
maps and average pooling, PSD bands8-12/12-20/20-32 with half-second windows,
linear interpolation masking, per-trial PLV rather than the paper's batch
averaging, a local region mapping, lambda_max=2 rescaling, and learned positive
orthogonality coefficients. Top-k parameter-bank selection is not implemented.
These choices and the prior low EDGCN student results prevent any claim of
strict author-code equivalence or expected reproduction of published accuracy.

## Data, matching and interpretation

- IV-2a/2b, standard22/3 EEG channels,250Hz, default8-32Hz,0-4s MOABB
  zero-phase filtering. Keep this offline protocol as requested. A2s input
  after offline filtering is NOT a demonstration of causal online2s latency.
- Session split: training session80/20 train/validation, separate test session.
  Student statistics fit ONLY on the training split's first500 points; teacher
  statistics fit on training split1000 points. No per-test-trial normalization.
- Default local-cache guard checks the named subject's T/E MAT files under
  data/MNE-bnci-data/~bci/database. It refuses if these files are missing.
- Two independent supervised4s teachers. Each selects best validation accuracy,
  then lower CE NLL; no test-session teacher evaluation.
- Four matched student arms: ce, temporal, spatial, combined. The same student
  initialization, shuffle/dropout seed, optimizer, and validation rule apply.
  Temporal-only/combined share lambda_time; spatial-only/combined share beta.
  This is a factorial loss comparison, NOT an equal-total-KL-weight ablation.
- Raw target losses are logged separately. Inspect their scales; beta=.2 does
  not mean the same gradient strength as lambda_time=.2. Graph target offdiag
  mean/SD is printed to help detect collapsed all-similar node representations.
- Never justify complementarity solely by teacher complexity. The combined
  student must be compared against BOTH single-guidance arms. Teacher accuracy
  alone does not establish successful transfer of4s future information.

## Server: first train just the two teachers

Keep every option identical across stages except --stage. Use a NEW directory,
not any previous KD/PVN output. Commands below are only subject1/seed42.

```bash
(
  set -e
  cd /workspace/zfq/eegnetfgl/eegnet_fgl_bci2a
  mkdir -p logs
  CUDA_VISIBLE_DEVICES=3 nohup /opt/conda/envs/eeg_fgl/bin/python -u \
    run_temporal_graph_distillation.py --stage teachers \
    --dataset bci2a --subjects 1 --seeds 42 \
    --data_dir data --device cuda \
    --output_dir results/temporal_graph_distillation_s1 \
    > logs/temporal_graph_teachers_s1.out 2>&1 &
  echo "PID: $!"
)
```

Review s1_seed42/teacher_validation.csv first. If teacher validation is weak,
do not automatically launch the full student matrix or claim effective teachers.
Then run four student arms (no test scoring):

```bash
(
  set -e
  cd /workspace/zfq/eegnetfgl/eegnet_fgl_bci2a
  mkdir -p logs
  CUDA_VISIBLE_DEVICES=3 nohup /opt/conda/envs/eeg_fgl/bin/python -u \
    run_temporal_graph_distillation.py --stage students \
    --dataset bci2a --subjects 1 --seeds 42 \
    --data_dir data --device cuda \
    --output_dir results/temporal_graph_distillation_s1 \
    > logs/temporal_graph_students_s1.out 2>&1 &
  echo "PID: $!"
)
```

Only after reviewing student_validation.csv, score locked students:

```bash
cd /workspace/zfq/eegnetfgl/eegnet_fgl_bci2a
CUDA_VISIBLE_DEVICES=3 /opt/conda/envs/eeg_fgl/bin/python -u \
  run_temporal_graph_distillation.py --stage test \
  --dataset bci2a --subjects 1 --seeds 42 --data_dir data --device cuda \
  --output_dir results/temporal_graph_distillation_s1
```

Test instantiates ONLY Compact students, not teachers or graph computation.
Rows/predictions and subject_results.csv,seed_summary.csv,summary.csv are saved.
Summary first averages subjects within seed, then reports sample SD across
complete seeds. One seed has no estimable across-seed SD (NaN, not zero).
Use --subjects1..9 and --seeds42 2024 3407 7 123 explicitly for a later full
experiment in a new root, not by expanding a manifest after inspecting test.

## Safety and resume

Checkpoints contain protocol config, train/validation-data hash, classes,
channels and normalization. Mismatches fail instead of mixing experiments.
Only COMPLETED fits are checkpointed atomically. Repeating a stage skips
completed fits; interrupted individual fits restart from their seeded initial
state, not from an arbitrary intermediate best epoch. No parallel writers to
the same root. Each stage has10-epoch progress prints and per-epoch CSV logs.
Old2s EDGCN input guards and checkpoint parameter keys are retained.

No clinical/online claims, no guaranteed accuracy, no test-based hyperparameter
search. First validate the teacher implementations and this new pilot protocol.
