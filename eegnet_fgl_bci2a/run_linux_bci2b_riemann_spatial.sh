#!/usr/bin/env bash
set -euo pipefail

# Linux runner for the current BCI IV-2b short-window main setting:
# 2 s Compact EEGNet + 4 s Riemann KD + spatial structure loss.
#
# Usage examples:
#   bash run_linux_bci2b_riemann_spatial.sh
#   SUBJECTS="1" SEEDS="42" EPOCHS=20 bash run_linux_bci2b_riemann_spatial.sh
#   DEVICE=cpu SUBJECTS="1" SEEDS="42" bash run_linux_bci2b_riemann_spatial.sh

export MNE_DATA="${MNE_DATA:-data/mne_data}"
export MNE_DATASETS_BNCI_PATH="${MNE_DATASETS_BNCI_PATH:-data/mne_data}"
export MNE_DONTWRITE_HOME="${MNE_DONTWRITE_HOME:-true}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-300}"
PATIENCE="${PATIENCE:-100}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
SUBJECTS="${SUBJECTS:-1 2 3 4 5 6 7 8 9}"
SEEDS="${SEEDS:-42 2024 3407 7 123}"
SAVE_DIR="${SAVE_DIR:-results/bci2b_compact_riemann_spatial_2s4s_all9_linux}"
LOG_DIR="${LOG_DIR:-results/linux_logs/bci2b_riemann_spatial}"

mkdir -p "$SAVE_DIR" "$LOG_DIR"

echo "MNE_DATA=$MNE_DATA"
echo "MNE_DATASETS_BNCI_PATH=$MNE_DATASETS_BNCI_PATH"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "DEVICE=$DEVICE"
echo "SUBJECTS=$SUBJECTS"
echo "SEEDS=$SEEDS"
echo "SAVE_DIR=$SAVE_DIR"

for seed in $SEEDS; do
  for subject in $SUBJECTS; do
    name="bci2b_s${subject}_seed${seed}_riemann_spatial"
    log_file="${LOG_DIR}/${name}.log"
    err_file="${LOG_DIR}/${name}.err"
    echo ">>> ${name}"
    "$PYTHON_BIN" train_student_riemann_guided.py \
      --dataset bci2b \
      --subject "$subject" \
      --window 2.0 \
      --riemann_window 4.0 \
      --model_variant compact \
      --fmin 8 \
      --fmax 32 \
      --tmin 0 \
      --tmax 4 \
      --eegnet_kd_weight 0.0 \
      --kl_weight 0.1 \
      --spatial_structure_weight 0.2 \
      --spatial_structure_source pre_depthwise \
      --spatial_structure_mode corr \
      --spatial_structure_trust_mode sample \
      --riemann_cov_estimator oas \
      --riemann_alignment none \
      --riemann_metric riemann \
      --riemann_method ts_logreg \
      --riemann_selection inner_cv \
      --riemann_selection_folds 5 \
      --teacher_calibration temperature \
      --seed "$seed" \
      --epochs "$EPOCHS" \
      --patience "$PATIENCE" \
      --batch_size "$BATCH_SIZE" \
      --lr "$LR" \
      --weight_decay "$WEIGHT_DECAY" \
      --device "$DEVICE" \
      --save_dir "$SAVE_DIR" \
      >"$log_file" 2>"$err_file"
    echo "OK ${name}"
  done
done

echo "All requested BCI IV-2b Riemann KD + spatial runs completed."
echo "Results CSV: ${SAVE_DIR}/riemann_guided_results.csv"

