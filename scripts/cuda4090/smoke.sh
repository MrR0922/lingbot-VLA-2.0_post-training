#!/usr/bin/env bash
set -Eeuo pipefail

/opt/robotwin-env/bin/python \
  /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/scripts/cuda4090/container_check.py
/opt/robotwin-env/bin/python -m torch.distributed.run --standalone --nproc-per-node=8 \
  /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/scripts/cuda4090/nccl_check.py

MAX_STEPS="${MAX_STEPS:-20}" SAVE_STEPS="${SAVE_STEPS:-20}" \
  /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/scripts/cuda4090/train_8x4090.sh
