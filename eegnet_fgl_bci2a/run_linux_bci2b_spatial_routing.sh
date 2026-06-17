#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SUBJECTS="${SUBJECTS:-1 2 3 4 5 6 7 8 9}"
SEEDS="${SEEDS:-42 2024 3407 7 123}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-300}"
EPOCHS_DETECTOR="${EPOCHS_DETECTOR:-300}"
PATIENCE="${PATIENCE:-100}"
PATIENCE_DETECTOR="${PATIENCE_DETECTOR:-80}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/bci2b_spatial_routing_all9_linux}"
LOG_DIR="${LOG_DIR:-results/linux_logs/bci2b_spatial_routing}"
CANDIDATE_SOURCE_ROOT="${CANDIDATE_SOURCE_ROOT:-}"
export OUTPUT_ROOT

mkdir -p "${LOG_DIR}"

echo "BCI2b spatial routing"
echo "subjects: ${SUBJECTS}"
echo "seeds: ${SEEDS}"
echo "device: ${DEVICE}"
echo "output: ${OUTPUT_ROOT}"

for seed in ${SEEDS}; do
  log_file="${LOG_DIR}/seed_${seed}.log"
  err_file="${LOG_DIR}/seed_${seed}.err"
  echo
  echo "===== seed ${seed} ====="
  command=(
    "${PYTHON_BIN}" run_bci2b_spatial_routing_seed.py
    --seed "${seed}"
    --subjects ${SUBJECTS}
    --device "${DEVICE}"
    --epochs "${EPOCHS}"
    --epochs_detector "${EPOCHS_DETECTOR}"
    --patience "${PATIENCE}"
    --patience_detector "${PATIENCE_DETECTOR}"
    --batch_size "${BATCH_SIZE}"
    --lr "${LR}"
    --weight_decay "${WEIGHT_DECAY}"
    --output_root "${OUTPUT_ROOT}"
  )
  if [[ -n "${CANDIDATE_SOURCE_ROOT}" ]]; then
    command+=(--candidate_source_root "${CANDIDATE_SOURCE_ROOT}")
  fi
  "${command[@]}" > >(tee "${log_file}") 2> >(tee "${err_file}" >&2)
done

"${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import os
import pandas as pd

root = Path(os.environ.get("OUTPUT_ROOT", "results/bci2b_spatial_routing_all9_linux"))
frames = []
for path in sorted(root.glob("seed_*/bci2b_spatial_routing/summary.csv")):
    frames.append(pd.read_csv(path))
if not frames:
    raise SystemExit("No summary.csv files found.")
summary = pd.concat(frames, ignore_index=True)
cols = [
    "fixed_ce_mean_acc",
    "fixed_riemann_spatial_mean_acc",
    "adaptive_ce_mean_acc",
    "adaptive_riemann_spatial_mean_acc",
    "global_mean_acc",
    "global_mean_f1",
    "global_mean_kappa",
]
print("\n===== all-seed summary =====")
print(summary[["seed", *cols]].to_string(index=False))
print("\n===== mean +/- std =====")
for col in cols:
    print(f"{col}: {summary[col].mean():.4f} +/- {summary[col].std(ddof=1):.4f}")
summary.to_csv(root / "all_seed_summary.csv", index=False)
PY
