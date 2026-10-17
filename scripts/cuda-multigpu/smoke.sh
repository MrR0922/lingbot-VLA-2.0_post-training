#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/profile.sh"

/opt/robotwin-env/bin/python \
  /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/scripts/cuda-multigpu/container_check.py
/opt/robotwin-env/bin/python -m torch.distributed.run --standalone --nproc-per-node="${GPU_COUNT}" \
  /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/scripts/cuda-multigpu/nccl_check.py

MAX_STEPS="${MAX_STEPS:-20}" SAVE_STEPS="${SAVE_STEPS:-20}" \
  "${SCRIPT_DIR}/train.sh"
