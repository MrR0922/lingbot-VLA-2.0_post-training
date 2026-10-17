#!/usr/bin/env bash
set -Eeuo pipefail

SOURCE_DIR="${LINGBOT_VLA_SOURCE:-/RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/source/lingbot-vla-v2}"
MODEL_ROOT="${MODEL_ROOT:-/models/robotwin-persistent/models}"
PYTHON="${PYTHON:-/opt/robotwin-env/bin/python}"
GPU_ID="${1:-0}"
PORT="${2:-13400}"
LOG_FILE="${3:-/workspace/runtime/outputs/logs/lingbot_server.log}"
USE_COMPILE="${4:-False}"
MODEL_PATH="${5:-${MODEL_ROOT}/robbyant_lingbot-vla-v2-6b}"
TRAINING_CONFIG="${LINGBOTVLA_TRAINING_CONFIG:-${MODEL_PATH}/lingbotvla_cli.yaml}"

[[ -d "${MODEL_PATH}" ]] || { echo "Missing model directory: ${MODEL_PATH}" >&2; exit 1; }
mkdir -p "$(dirname "${LOG_FILE}")"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export QWEN3VL_PATH="${MODEL_ROOT}/Qwen3-VL-4B-Instruct-config-tokenizer"
unset HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES HSA_OVERRIDE_GFX_VERSION
unset AITER_TRITON_ONLY FLASH_ATTENTION_TRITON_AMD_ENABLE
if [[ -f "${TRAINING_CONFIG}" ]]; then
  export LINGBOTVLA_TRAINING_CONFIG="${TRAINING_CONFIG}"
fi

cd "${SOURCE_DIR}"
exec "${PYTHON}" -m deploy.lingbot_vla_v2_policy \
  --model_path "${MODEL_PATH}" \
  --use_length 25 \
  --use_bf16 True \
  --use_fp32 False \
  --use_compile "${USE_COMPILE}" \
  --port "${PORT}" >>"${LOG_FILE}" 2>&1
