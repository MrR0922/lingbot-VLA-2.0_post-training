#!/usr/bin/env bash
set -Eeuo pipefail

PERSISTENT_ROOT="${PERSISTENT_ROOT:-/models/robotwin-persistent}"
PROFILE_SCRIPT="/RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/scripts/cuda-multigpu/profile.sh"
source "${PROFILE_SCRIPT}"
mkdir -p \
  /workspace/runtime/.cache/huggingface \
  /workspace/runtime/outputs \
  /workspace/runtime/eval_result

if [[ "${REQUIRE_PREPARED_DATA:-0}" == "1" ]]; then
  required_paths=(
    "${PERSISTENT_ROOT}/data/lerobot-clean"
    "${PERSISTENT_ROOT}/data/robotwin_clean_v30.txt"
    "${PERSISTENT_ROOT}/models/robbyant_lingbot-vla-v2-6b/model.safetensors.index.json"
    "${PERSISTENT_ROOT}/models/Qwen3-VL-4B-Instruct-config-tokenizer"
  )
  for required_path in "${required_paths[@]}"; do
    if [[ ! -e "${required_path}" ]]; then
      echo "Missing persistent RoboTwin content: ${required_path}" >&2
      echo "Mount the prepared tree at ${PERSISTENT_ROOT}." >&2
      exit 1
    fi
  done
fi

exec "$@"
