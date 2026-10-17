#!/usr/bin/env bash
set -Eeuo pipefail

PORT="${PORT:-13400}"
DEVICE_ID="${DEVICE_ID:-1}"
TEST_NUM="${TEST_NUM:-10}"
PLANNER="${ROBOTWIN_EE_PLANNER:-curobo}"
PYTHON="${PYTHON:-/opt/robotwin-env/bin/python}"

export PYOPENGL_PLATFORM=egl
if [[ "${PLANNER}" == "mplib" ]]; then
  export ROBOTWIN_DISABLE_CUROBO=1
  export ROBOTWIN_EE_PLANNER=mplib
else
  unset ROBOTWIN_DISABLE_CUROBO
  export ROBOTWIN_EE_PLANNER=curobo
fi

cd /RoboTwin
"${PYTHON}" scripts/eval_policy_xpolicylab.py \
  --task_name adjust_bottle \
  --task_config demo_clean \
  --policy_name LingBot-VLA-v2 \
  --protocol lingbot_vla_v2 \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --device_id "${DEVICE_ID}" \
  --seed 0 \
  --test_num "${TEST_NUM}" \
  --expert_check true \
  --accept_expert_info_on_failure true \
  --eval_batch false \
  --additional_info eval_video_log=false
