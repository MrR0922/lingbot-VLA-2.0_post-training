#!/usr/bin/env bash
# Continue a no-Docker server setup through a checkpoint-validated smoke run.
set -Eeuo pipefail

PROFILE="${PROFILE:-7x4090}"
SCRIPT_ROOT="/RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/scripts/cuda-multigpu"
source "${SCRIPT_ROOT}/profile.sh"
CUDA_SETUP_PID="${CUDA_SETUP_PID:-}"
MODEL_DOWNLOAD_PID="${MODEL_DOWNLOAD_PID:-}"
DATA_CONVERT_PID="${DATA_CONVERT_PID:-}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/workspace/runtime}"
LOG_ROOT="${LOG_ROOT:-${RUNTIME_ROOT}/outputs/logs}"
STATUS_FILE="${STATUS_FILE:-${RUNTIME_ROOT}/pipeline.status}"
PYTHON="${PYTHON:-/opt/robotwin-env/bin/python}"
FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.8.3.post2}"
FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS:-80}"
FLASH_ATTN_MAX_JOBS="${FLASH_ATTN_MAX_JOBS:-4}"
MODEL_ROOT="/models/robotwin-persistent/models"
DATA_ROOT="/models/robotwin-persistent/data"
SMOKE_OUTPUT="${SMOKE_OUTPUT:-${RUNTIME_ROOT}/outputs/smoke_${PROFILE}}"

mkdir -p "${LOG_ROOT}" "${SMOKE_OUTPUT}"

status() {
  printf '%s\t%s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "${STATUS_FILE}"
}

wait_for_pid() {
  local name="$1" pid="$2"
  [[ -n "${pid}" ]] || return 0
  status "waiting for ${name} pid=${pid}"
  while kill -0 "${pid}" 2>/dev/null; do sleep 30; done
  status "${name} pid=${pid} finished"
}

run_logged() {
  local name="$1"
  shift
  status "START ${name}"
  "$@" >"${LOG_ROOT}/${name}.log" 2>&1
  status "PASS ${name}"
}

wait_for_pid cuda-toolkit "${CUDA_SETUP_PID}"
[[ -x /usr/local/cuda-12.8/bin/nvcc ]]
run_logged flash-attn-install env \
  CUDA_HOME=/usr/local/cuda-12.8 \
  PATH=/usr/local/cuda-12.8/bin:/opt/robotwin-env/bin:/usr/bin:/bin \
  FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS}" \
  MAX_JOBS="${FLASH_ATTN_MAX_JOBS}" \
  "${PYTHON}" -m pip install --no-build-isolation "flash-attn==${FLASH_ATTN_VERSION}"

wait_for_pid model-download "${MODEL_DOWNLOAD_PID}"
for required in \
  "${MODEL_ROOT}/robbyant_lingbot-vla-v2-6b/model.safetensors.index.json" \
  "${MODEL_ROOT}/robbyant_lingbot-vla-v2-6b/depth/model.pt" \
  "${MODEL_ROOT}/robbyant_lingbot-vla-v2-6b/dino_video/teacher_step_10000.pth" \
  "${MODEL_ROOT}/moge-2-vitb-normal/model.pt"; do
  [[ -s "${required}" ]] || { status "FAIL missing ${required}"; exit 1; }
done
status "PASS model-prerequisites"

wait_for_pid data-conversion "${DATA_CONVERT_PID}"
[[ "$(grep -cve '^[[:space:]]*$' "${DATA_ROOT}/robotwin_clean_v30.txt")" -eq 50 ]]
[[ "$(find "${DATA_ROOT}/lerobot-clean" -mindepth 3 -maxdepth 3 -type f -path '*/meta/info.json' | wc -l)" -eq 50 ]]
status "PASS clean50-prerequisites"

run_logged norm-stats env PROFILE="${PROFILE}" "${SCRIPT_ROOT}/compute_norm_stats.sh"
run_logged container-check env PROFILE="${PROFILE}" "${PYTHON}" "${SCRIPT_ROOT}/container_check.py"
run_logged nccl-check env PROFILE="${PROFILE}" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
  "${PYTHON}" -m torch.distributed.run --standalone --nproc-per-node=7 \
  "${SCRIPT_ROOT}/nccl_check.py"

run_logged smoke-20 env \
  PROFILE="${PROFILE}" MAX_STEPS=20 SAVE_STEPS=20 \
  OUTPUT_DIR="${SMOKE_OUTPUT}" ENABLE_RESUME=false \
  "${SCRIPT_ROOT}/train.sh"
run_logged verify-step20 env PROFILE="${PROFILE}" OUTPUT_DIR="${SMOKE_OUTPUT}" \
  "${SCRIPT_ROOT}/checkpoint.sh" verify "${SMOKE_OUTPUT}/checkpoints/global_step_20"
run_logged export-step20 env PROFILE="${PROFILE}" OUTPUT_DIR="${SMOKE_OUTPUT}" \
  "${SCRIPT_ROOT}/checkpoint.sh" export-hf \
  "${SMOKE_OUTPUT}/checkpoints/global_step_20" \
  "${SMOKE_OUTPUT}/checkpoints/global_step_20/hf_ckpt"
run_logged audit-step20 env PROFILE="${PROFILE}" OUTPUT_DIR="${SMOKE_OUTPUT}" \
  "${SCRIPT_ROOT}/checkpoint.sh" audit-update \
  "${MODEL_ROOT}/robbyant_lingbot-vla-v2-6b" \
  "${SMOKE_OUTPUT}/checkpoints/global_step_20/hf_ckpt" \
  "${SMOKE_OUTPUT}/checkpoint_update_step20.json"

run_logged resume-to-25 env \
  PROFILE="${PROFILE}" MAX_STEPS=25 SAVE_STEPS=25 \
  OUTPUT_DIR="${SMOKE_OUTPUT}" ENABLE_RESUME=true \
  "${SCRIPT_ROOT}/train.sh"
run_logged verify-step25 env PROFILE="${PROFILE}" OUTPUT_DIR="${SMOKE_OUTPUT}" \
  "${SCRIPT_ROOT}/checkpoint.sh" verify "${SMOKE_OUTPUT}/checkpoints/global_step_25"

status "PIPELINE_COMPLETE smoke=${SMOKE_OUTPUT}"
