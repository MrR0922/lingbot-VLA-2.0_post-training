#!/usr/bin/env bash
set -Eeuo pipefail

SOURCE="${SOURCE:-/opt/data/robotwin-persistent}"
PERSIST_ROOT="${PERSIST_ROOT:-/opt/data/private/lingbot-vla-v2}"
TARGET="${TARGET:-${PERSIST_ROOT}/robotwin-persistent}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/workspace/runtime}"
OLD_PIPELINE_PID="${OLD_PIPELINE_PID:-}"
PIPELINE_SCRIPT="/RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/scripts/cuda-multigpu/run_server_pipeline.sh"

mkdir -p "${TARGET}" "${RUNTIME_ROOT}/outputs/logs"
printf '%s\t%s\n' "$(date --iso-8601=seconds)" "START persistent asset migration ${SOURCE} -> ${TARGET}" >> "${RUNTIME_ROOT}/pipeline.status"

# First pass is online: the active normalization process may continue reading.
rsync -a --info=progress2 "${SOURCE}/" "${TARGET}/"

kill_tree() {
  local parent="$1" child
  while read -r child; do
    [[ -n "${child}" ]] && kill_tree "${child}"
  done < <(pgrep -P "${parent}" 2>/dev/null || true)
  kill -TERM "${parent}" 2>/dev/null || true
}

if [[ -n "${OLD_PIPELINE_PID}" ]] && kill -0 "${OLD_PIPELINE_PID}" 2>/dev/null; then
  printf '%s\t%s\n' "$(date --iso-8601=seconds)" "PAUSE old single-rank pipeline pid=${OLD_PIPELINE_PID}" >> "${RUNTIME_ROOT}/pipeline.status"
  kill_tree "${OLD_PIPELINE_PID}"
  sleep 5
fi

# Capture files that changed during the online pass, then switch atomically.
rsync -a --delete "${SOURCE}/" "${TARGET}/"
backup="${SOURCE}.ephemeral-backup-$(date +%Y%m%d-%H%M%S)"
mv "${SOURCE}" "${backup}"
ln -s "${TARGET}" "${SOURCE}"

printf '%s\t%s\n' "$(date --iso-8601=seconds)" "PASS persistent asset migration; backup=${backup}" >> "${RUNTIME_ROOT}/pipeline.status"
printf '%s\t%s\n' "$(date --iso-8601=seconds)" "RESTART 7-rank normalization and training pipeline" >> "${RUNTIME_ROOT}/pipeline.status"

exec env PROFILE=7x4090 NORM_NPROC=7 NORM_WORKERS_PER_RANK=2 \
  FLASH_ATTN_VERSION=2.8.3.post2 FLASH_ATTN_CUDA_ARCHS=80 FLASH_ATTN_MAX_JOBS=4 \
  "${PIPELINE_SCRIPT}"
