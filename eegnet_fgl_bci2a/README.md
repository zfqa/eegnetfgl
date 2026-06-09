# EEGNet-FGL for Early Motor Imagery EEG Classification on BCI Competition IV 2a

This project implements a PyTorch pipeline for early motor imagery EEG classification on MOABB `BNCI2014_001` / BCI Competition IV 2a.

The first-pass workflow focuses on subject 1:

1. Train a full-window 0-4s EEGNet teacher with cross entropy.
2. Freeze the teacher and run it online with the student during distillation.
3. Train early-window EEGNet students with CE at 0.5s, 1s, or 2s.
4. Train early-window EEGNet-FGL students with CE + frozen 4s teacher soft-label KL.
5. Test only the early-window student. The teacher is not used at test time.

## Environment

```bash
conda create -n eeg_fgl python=3.10 -y
conda activate eeg_fgl
pip install -r requirements.txt
```

If CUDA is unavailable, use `--device cpu`. The scripts automatically fall back to CPU when `--device cuda` is requested but CUDA is not available.

## Dataset

The data loader uses MOABB:

- Dataset: `BNCI2014_001`
- Paradigm: `MotorImagery`
- Classes: `left_hand`, `right_hand`, `feet`, `tongue`
- Filtering/window: `fmin=4`, `fmax=38`, `tmin=0`, `tmax=4`

By default, MOABB/MNE downloads are kept inside this project under `data/mne_data`.
If you already use a separate MNE cache, set `MNE_DATASETS_BNCI_PATH` before running the scripts.

The subject-dependent cross-session split is:

- session 1 / train session: train + validation
- validation: stratified 20% split from the train session
- session 2 / evaluation session: test

Standardization is computed from the training split only:

```python
mean = X_train.mean(axis=(0, 2), keepdims=True)
std = X_train.std(axis=(0, 2), keepdims=True) + 1e-6
```

### MOABB-Standard Paper Protocol

For paper-facing experiments, use the explicit MOABB-style protocol flag:

```bash
python train_student_ce.py --subject 1 --window 2 --protocol moabb_standard --device cuda
python train_eegnet_official.py --subject 1 --window 4 --protocol moabb_standard --device cuda
```

This fixes the loading/evaluation protocol to:

- Dataset: `BNCI2014_001`
- Paradigm: `MotorImagery`
- Evaluation: cross-session, training session for train/validation and evaluation session for test
- Classes: 4 motor imagery classes
- Channels: all 22 EEG channels
- Sampling rate: 250 Hz
- Time interval: 0-4 s after cue
- Band-pass: 8-32 Hz
- Standardization: fit on the training split only, then apply to validation/test

The `protocol` value is written to result CSV files so legacy project results and MOABB-standard results do not overwrite each other.

## Train Teacher

```bash
python train_teacher.py --subject 1 --device cuda
```

Smoke test:

```bash
python train_teacher.py --subject 1 --epochs 5 --device cuda
```

Output:

- `results/teacher/subject_1_teacher_best.pth`
- `results/teacher/teacher_results.csv`

## Optional: Save Teacher Logits

```bash
python save_teacher_logits.py --subject 1 --teacher_path results/teacher/subject_1_teacher_best.pth
```

Output:

- `data/teacher_logits/subject_1_train_logits.npy`
- `data/teacher_logits/subject_1_val_logits.npy`

No test logits are saved. The default FGL path no longer needs these files; use them only with `--distill_mode logits`.

## Train CE Student

```bash
python train_student_ce.py --subject 1 --window 1.0 --device cuda
```

Smoke test:

```bash
python train_student_ce.py --subject 1 --window 1.0 --epochs 5 --device cuda
```

Output:

- `results/student_ce/subject_1_window_1_ce_best.pth`
- `results/student_ce/student_ce_results.csv`

## Train FGL Student

```bash
python train_student_fgl.py --subject 1 --window 1.0 --alpha 0.5 --temperature 4.0 --teacher_path results/teacher/subject_1_teacher_best.pth --mentor_name EEGNet-Teacher-4s --method_name EEGNet-FGL --device cuda
```

Smoke test:

```bash
python train_student_fgl.py --subject 1 --window 1.0 --epochs 5 --alpha 0.5 --temperature 4.0 --teacher_path results/teacher/subject_1_teacher_best.pth --mentor_name EEGNet-Teacher-4s --method_name EEGNet-FGL --device cuda
```

Output:

- `results/student_fgl/subject_1_window_1_alpha_0.5_temp_4_fgl_best.pth`
- `results/student_fgl/student_fgl_results.csv`

## Train Riemann-GCN

This path builds a trial-level graph from Riemannian covariance geometry:

```text
early EEG -> covariance matrix -> Riemannian alignment -> tangent-space features -> Riemannian kNN graph -> GCN/MLP
```

Run 1s:

```bash
python train_riemann_gcn.py --subject 1 --window 1.0 --device cuda
```

Run 2s:

```bash
python train_riemann_gcn.py --subject 1 --window 2.0 --device cuda
```

Run tangent-space MLP instead of GCN:

```bash
python train_riemann_gcn.py --subject 1 --window 2.0 --model_type mlp --graph_lambda 0 --device cuda
```

Run classical Riemannian baselines:

```bash
python train_riemann_baseline.py --subject 1 --window 2.0
```

Train EEGNet + Riemann tangent fusion:

```bash
python train_eegnet_riemann_fusion.py --subject 1 --window 1.0 --fusion_mode logit_residual --init_eegnet_checkpoint results/student_ce/subject_1_window_1_ce_best.pth --freeze_eegnet --device cuda
```

Evaluate late probability fusion between a trained EEGNet CE model and Riemann-LogReg:

```bash
python evaluate_eegnet_riemann_ensemble.py --subject 1 --window 2.0 --cov_shrinkage 0.001 --device cuda
```

Useful knobs:

- `--graph_k`: number of Riemannian nearest neighbors, default `8`
- `--graph_lambda`: graph smoothness regularization, default `0.01`
- `--cov_shrinkage`: covariance regularization, default `0.001`
- `--model_type`: `gcn` or `mlp`, default `gcn`

Output:

- `results/riemann_gcn/subject_1_window_1_gcn_k_8_lambda_0.01_riemann_best.pth`
- `results/riemann_gcn/riemann_gcn_results.csv`
- `results/riemann_baseline/riemann_baseline_results.csv`
- `results/eegnet_riemann_fusion/eegnet_riemann_fusion_results.csv`
- `results/eegnet_riemann_fusion/eegnet_riemann_ensemble_results.csv`

## Run Subject 1 Pipeline

`run_all_subjects.py` defaults to subject 1 for the first working version:

```bash
python run_all_subjects.py --device cuda
```

By default, the FGL part distills every requested early window directly from the frozen 4s teacher:

1. 4s teacher is trained with four-class cross entropy.
2. Each early-window FGL student learns online from the frozen 4s teacher checkpoint.

To run all 9 subjects later:

```bash
python run_all_subjects.py --subjects 1 2 3 4 5 6 7 8 9 --device cuda
```

## Analyze Results

```bash
python analyze_results.py
```

Output:

- `results/all_subject_results.csv`
- `results/summary_results.csv`
- `results/figures/window_vs_accuracy.png`

Main result columns:

```text
method, subject, window, accuracy, macro_f1, kappa
```

Methods:

- `EEGNet-Teacher-4s`
- `EEGNet-CE`
- `EEGNet-FGL`
- `Riemann-GCN`

## FGL Loss

```python
student_log_prob = log_softmax(student_logits / tau, dim=1)
mentor_prob = softmax(mentor_logits / tau, dim=1)
kl = kl_div(student_log_prob, mentor_prob, reduction="batchmean") * tau**2
loss = alpha * CE(student_logits, labels) + (1 - alpha) * kl
```

In default online mode, `mentor_logits` are produced inside each training batch by the frozen 4s teacher checkpoint. The logits files are only used by `--distill_mode logits`.

The scripts expose `--alpha` and `--temperature`, so later ablations can run values such as:

- `alpha`: 0.1, 0.3, 0.5, 0.7, 0.9
- `temperature`: 1, 2, 4, 8

## Important Notes

- Do not use test-session full-window teacher outputs to train the student.
- Teacher weights are frozen during FGL distillation.
- Standardization uses training-set statistics only.
- FGL students use the frozen 4s teacher directly; no assistant model is trained.
- Student test uses only early EEG windows and never uses the teacher.
- Teacher, CE student, and FGL student must use the same subject, seed, and validation ratio to keep splits aligned.
- The test session is never used for model selection.

## Paper Experiment Matrix

The paper-facing matrix uses the MOABB-standard BNCI2014_001 protocol:
MotorImagery, 8-32 Hz, 0-4 s, 250 Hz, session-based train/evaluation split,
and train-only standardization.

Run the full multi-seed matrix:

```bash
python run_paper_experiment_matrix.py --subjects 1 2 3 4 5 6 7 8 9 --seeds 42 2024 3407 7 123 --groups all --root_dir results/paper_moabb_multiseed --epochs 300 --teacher_epochs 300 --eegnetv4_epochs 300 --batch_size 32 --lr 0.001 --device cuda
```

For staged execution:

```bash
python run_paper_experiment_matrix.py --groups teachers ce --subjects 1 2 3 4 5 6 7 8 9 --seeds 42 2024 3407 7 123
python run_paper_experiment_matrix.py --groups dual_ablation riemann_guided --subjects 1 2 3 4 5 6 7 8 9 --seeds 42 2024 3407 7 123
python run_paper_experiment_matrix.py --groups eegnetv4 baselines --subjects 1 2 3 4 5 6 7 8 9 --seeds 42 2024 3407 7 123
```

The matrix covers:

- Compact EEGNet 2s/4s CE.
- EEGNetV4 2s/4s.
- EEGNet teacher KD, Riemann teacher KD, and dual KD.
- Riemann spatial-only, Riemann KD + spatial, Riemann KD + geometry, and Riemann KD + geometry + spatial.
- Riemann MDM, Riemann tangent-space LogReg, and CSP + LDA.

Aggregate results, run paired significance tests, and generate figures:

```bash
python analyze_paper_results.py --results_roots results/paper_moabb_multiseed --output_dir results/paper_analysis --baseline_label "Compact EEGNet 2s" --mainline_label "Riemann-guided dual KD 2s"
```

Main outputs:

- `results/paper_analysis/paper_long_results.csv`
- `results/paper_analysis/paper_summary_by_method.csv`
- `results/paper_analysis/paper_paired_significance_tests.csv`
- `results/paper_analysis/paper_subject_seed_accuracy_pivot.csv`
- `results/paper_analysis/method_mean_accuracy_bar.png`
- `results/paper_analysis/subject_improvement_mainline_vs_baseline.png`
- `results/paper_analysis/riemann_val_vs_student_test_scatter.png`
