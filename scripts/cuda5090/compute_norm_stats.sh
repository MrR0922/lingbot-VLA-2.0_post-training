#!/usr/bin/env bash
set -Eeuo pipefail

SOURCE_DIR="${LINGBOT_VLA_SOURCE:-/RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/source/lingbot-vla-v2}"
DATA_LIST="${DATA_LIST:-/models/robotwin-persistent/data/robotwin_clean_v30.txt}"
OUTPUT="${OUTPUT:-/models/robotwin-persistent/data/robotwin_clean.json}"
PYTHON="${PYTHON:-/opt/robotwin-env/bin/python}"

[[ -f "${DATA_LIST}" ]] || { echo "Missing clean training list: ${DATA_LIST}" >&2; exit 1; }
[[ "$(wc -l <"${DATA_LIST}")" -eq 50 ]] || { echo "Training list must contain 50 lines" >&2; exit 1; }
mkdir -p "$(dirname "${OUTPUT}")"
cd "${SOURCE_DIR}"

CUDA_VISIBLE_DEVICES=0 "${PYTHON}" scripts/compute_norm_stats.py \
  ./configs/vla/norm_compute/post_data.yaml \
  --data.data_name multi \
  --data.robot_name robotwin \
  --data.train_path "${DATA_LIST}" \
  --data.norm_path "${OUTPUT}"

[[ -s "${OUTPUT}" ]] || { echo "Norm stats were not created: ${OUTPUT}" >&2; exit 1; }
echo "Clean-only normalization stats: ${OUTPUT}"
