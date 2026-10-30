#!/usr/bin/env bash
set -euo pipefail

# Reviewer 2.2: matched, five-seed extended comparison.  TEMDnet-style and
# TEM-NLnet-style are transparent adaptations to the common 11x869 benchmark;
# joint_gated_moe trains both PC-BiLSTM experts and its gate end-to-end.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="${CODE_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-${CODE_ROOT}/data/full}"
OUT_ROOT="${OUT_ROOT:-${CODE_ROOT}/artifacts/extended_baselines}"
PYTHON="${PYTHON:-python}"
DEVICE_IDS="${DEVICE_IDS:-0}"

if [[ ! -f "${DATA_ROOT}/train/train_data.npy" ]]; then
  echo "Full benchmark data are required at DATA_ROOT=${DATA_ROOT}" >&2
  exit 2
fi

for seed in 44 45 46 47 48; do
  for model in temdnet_style temnlnet_style joint_gated_moe; do
    if [[ "${model}" == "joint_gated_moe" ]]; then
      microbatch=8
      accumulation=16
    else
      microbatch=64
      accumulation=2
    fi
    "${PYTHON}" "${CODE_ROOT}/training/train_external_baselines.py" \
      --model "${model}" \
      --train-data "${DATA_ROOT}/train/train_data.npy" \
      --train-labels "${DATA_ROOT}/train/train_labels.npy" \
      --val-data "${DATA_ROOT}/val/val_data.npy" \
      --val-labels "${DATA_ROOT}/val/val_labels.npy" \
      --test-data "${DATA_ROOT}/test/test_data.npy" \
      --test-labels "${DATA_ROOT}/test/test_labels.npy" \
      --data-root "${DATA_ROOT}" \
      --output-dir "${OUT_ROOT}/${model}/seed_${seed}" \
      --batch-size "${microbatch}" --grad-accum-steps "${accumulation}" --num-workers 4 \
      --max-epochs 120 --min-epochs 40 --patience 25 \
      --selection-metric rrmse --device cuda --device-ids "${DEVICE_IDS}" --seed "${seed}"
  done
done
