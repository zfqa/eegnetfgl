# Causal preprocessing and matched teacher-window experiment

This experiment addresses two separate questions:

1. Does offline forward-backward filtering allow samples after 2 s to affect the
   nominal 0-2 s input?
2. With all student settings matched, does a 4 s neural teacher provide more
   useful supervision than a 2 s neural teacher?

The strict path reads the cached BNCI MAT recordings, applies a fourth-order
one-pass Butterworth 8-32 Hz filter to each continuous run, and only then extracts
cue-aligned epochs. It preserves the same labels, trial order, channel set and
train/evaluation session split as the MOABB path.

## 1. Future-information audit

From the project root on the Linux server:

```bash
python audit_filter_future_leakage.py \
  --datasets bci2a bci2b \
  --subject 1 \
  --student_window 2.0 \
  --fmin 8 --fmax 32 --filter_order 4 \
  --data_dir data \
  --output_csv results/filter_future_leakage_audit_s1.csv
```

Expected behavior: changing the signal after 2 s changes the forward-backward
filtered 0-2 s prefix, whereas `causal_one_pass` reports exactly zero change.

## 2. End-to-end smoke test

```bash
CUDA_VISIBLE_DEVICES=3 python run_teacher_window_matched_kd.py \
  --dataset bci2a \
  --subjects 1 --seeds 42 \
  --preprocessing_mode causal_mat --causal_filter_order 4 --data_dir data \
  --epochs_teacher 3 --epochs_student 3 \
  --patience_teacher 3 --patience_student 3 \
  --device cuda \
  --output_dir results/teacher_window_matched_kd_smoke
```

This trains five short smoke-test jobs: 2 s and 4 s teachers, followed by matched
CE, 2s-to-2s KD, and 4s-to-2s KD students.

Use `--dataset bci2b` and a separate output directory for the corresponding
BCI IV-2b smoke test.

## 3. Formal 9-subject, 5-seed experiments

BCI IV-2a:

```bash
mkdir -p logs

CUDA_VISIBLE_DEVICES=3 nohup python -u run_teacher_window_matched_kd.py \
  --dataset bci2a \
  --subjects 1 2 3 4 5 6 7 8 9 \
  --seeds 42 2024 3407 7 123 \
  --preprocessing_mode causal_mat --causal_filter_order 4 --data_dir data \
  --epochs_teacher 300 --epochs_student 300 \
  --patience_teacher 100 --patience_student 100 \
  --batch_size 32 --lr 0.001 --weight_decay 0.0001 \
  --alpha 0.4 --temperature 1.0 \
  --device cuda \
  --output_dir results/teacher_window_matched_kd_causal \
  > logs/teacher_window_matched_kd_causal.out 2>&1 &

echo "PID: $!"
```

BCI IV-2b uses the same causal preprocessing and matched settings, but trains
3-channel, 2-class models and writes to a separate result root:

```bash
CUDA_VISIBLE_DEVICES=3 nohup python -u run_teacher_window_matched_kd.py \
  --dataset bci2b \
  --subjects 1 2 3 4 5 6 7 8 9 \
  --seeds 42 2024 3407 7 123 \
  --preprocessing_mode causal_mat --causal_filter_order 4 --data_dir data \
  --epochs_teacher 300 --epochs_student 300 \
  --patience_teacher 100 --patience_student 100 \
  --batch_size 32 --lr 0.001 --weight_decay 0.0001 \
  --alpha 0.4 --temperature 1.0 \
  --device cuda \
  --output_dir results/teacher_window_matched_kd_causal_bci2b \
  > logs/teacher_window_matched_kd_causal_bci2b.out 2>&1 &

echo "PID: $!"
```

The runner is resumable. Verified teacher checkpoints and existing student rows
with the exact configuration are skipped. Do not add `--rerun_existing` when
resuming. A `run_config.json` guard rejects attempts to mix different datasets
or training hyperparameters in the same output directory.

## 4. Monitor progress

```bash
tail -f logs/teacher_window_matched_kd_causal.out
```

```bash
pgrep -af 'run_teacher_window_matched_kd|train_teacher.py|train_student_dual_kd.py'
nvidia-smi
```

## 5. Read the final results

```bash
column -s, -t < results/teacher_window_matched_kd_causal/matched_summary.csv
column -s, -t < results/teacher_window_matched_kd_causal/matched_paired_tests.csv
```

Important comparisons are:

- `Matched EEGNet KD 2s-to-2s minus Matched CE 2s`;
- `Matched EEGNet KD 4s-to-2s minus Matched CE 2s`;
- `Matched EEGNet KD 4s-to-2s minus Matched EEGNet KD 2s-to-2s`.

The last comparison is the direct test of whether training-only future context is
valuable. Seeds are averaged within each subject before the paired tests, so the
inferential sample size is nine subjects rather than 45 subject-seed runs.

The new causal results form a separate protocol. They must not be numerically
combined with the earlier MOABB forward-backward-filtered baseline.
