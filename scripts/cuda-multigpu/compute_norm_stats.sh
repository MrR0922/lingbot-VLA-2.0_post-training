#!/usr/bin/env bash
set -Eeuo pipefail

SOURCE_DIR="${LINGBOT_VLA_SOURCE:-/RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/source/lingbot-vla-v2}"
DATA_LIST="${DATA_LIST:-/models/robotwin-persistent/data/robotwin_clean_v30.txt}"
OUTPUT="${OUTPUT:-/models/robotwin-persistent/data/robotwin_clean.json}"
PYTHON="${PYTHON:-/opt/robotwin-env/bin/python}"
PROFILE="${PROFILE:-7x4090}"
SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_ROOT}/profile.sh"
NORM_NPROC="${NORM_NPROC:-${GPU_COUNT}}"
NORM_WORKERS_PER_RANK="${NORM_WORKERS_PER_RANK:-2}"

[[ -f "${DATA_LIST}" ]] || { echo "Missing clean training list: ${DATA_LIST}" >&2; exit 1; }
[[ "$(wc -l <"${DATA_LIST}")" -eq 50 ]] || { echo "Training list must contain 50 lines" >&2; exit 1; }
mkdir -p "$(dirname "${OUTPUT}")"
cd "${SOURCE_DIR}"

COMMON_ARGS=(
  scripts/compute_norm_stats.py
  ./configs/vla/norm_compute/post_data.yaml
  --data.data_name multi
  --data.robot_name robotwin
  --data.train_path "${DATA_LIST}"
  --data.norm_path "${OUTPUT}"
  --data.num_workers "${NORM_WORKERS_PER_RANK}"
)

if [[ "${NORM_NPROC}" -gt 1 ]]; then
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${PYTHON}" -m torch.distributed.run \
    --standalone --nproc-per-node="${NORM_NPROC}" "${COMMON_ARGS[@]}"
else
  CUDA_VISIBLE_DEVICES=0 LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 \
    "${PYTHON}" "${COMMON_ARGS[@]}"
fi

[[ -s "${OUTPUT}" ]] || { echo "Norm stats were not created: ${OUTPUT}" >&2; exit 1; }
echo "Clean-only normalization stats: ${OUTPUT}"
