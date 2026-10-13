#!/usr/bin/env bash
set -Eeuo pipefail

ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-/RoboTwin}"
DATA_ROOT="${DATA_ROOT:-/models/robotwin-persistent/data}"
LEROBOT_ROOT="${LEROBOT_ROOT:-${DATA_ROOT}/lerobot-clean}"
MANIFEST="${MANIFEST:-${DATA_ROOT}/robotwin_clean_v30.txt}"
LEROBOT_PYTHON="${LEROBOT_PYTHON:-/opt/lerobot-env/bin/python}"
CONVERSION_JOBS="${CONVERSION_JOBS:-4}"
LOG_ROOT="${LOG_ROOT:-/workspace/runtime/outputs/data-conversion}"

cd "${ROBOTWIN_ROOT}"
mkdir -p "${LEROBOT_ROOT}" "${LOG_ROOT}"
export HF_LEROBOT_HOME="${LEROBOT_ROOT}"
export LEROBOT_VIDEO_BACKEND="${LEROBOT_VIDEO_BACKEND:-pyav}"
export LEROBOT_VIDEO_CODEC="${LEROBOT_VIDEO_CODEC:-h264}"

# This tested helper creates data/demo_clean/<task> for exactly the 50 clean
# tasks. HF_ARCHIVE_CACHE may point at a pre-populated archive cache.
if [[ ! -d data/demo_clean || "$(find data/demo_clean -mindepth 1 -maxdepth 1 -type d | wc -l)" -ne 50 ]]; then
  HF_ARCHIVE_CACHE="${HF_ARCHIVE_CACHE:-${DATA_ROOT}/raw-clean}" \
  HF_KEEP_ARCHIVES=1 \
  HF_MAX_WORKERS="${HF_MAX_WORKERS:-8}" \
  HF_EXTRACT_WORKERS="${HF_EXTRACT_WORKERS:-8}" \
  PATH="/opt/lerobot-env/bin:${PATH}" \
    bash scripts/download_xpolicylab_data.sh
fi

mapfile -t tasks < <(find data/demo_clean -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
[[ "${#tasks[@]}" -eq 50 ]] || { echo "Expected 50 demo_clean tasks, found ${#tasks[@]}" >&2; exit 1; }

convert_task() {
  local task="$1"
  local repo_id="${task}_joint_v30"
  local log_file="${LOG_ROOT}/${task}.log"
  local output_dir="${HF_LEROBOT_HOME}/${repo_id}"
  if [[ -f "${output_dir}/meta/info.json" ]]; then
    echo "[existing] ${task}"
    return
  fi
  rm -rf "${output_dir}"
  if "${LEROBOT_PYTHON}" XPolicyLab/scripts/transform_lerobot_v30_format.py \
      "demo_clean.${task}.aloha_agilex" --repo_id "${repo_id}" --max_episode 50 \
      >"${log_file}" 2>&1 && \
      ! grep -q '^Failed files:' "${log_file}" && \
      [[ -f "${output_dir}/meta/info.json" ]]; then
    echo "[converted] ${task}"
  else
    tail -100 "${log_file}" >&2 || true
    return 1
  fi
}
export -f convert_task
export LEROBOT_PYTHON HF_LEROBOT_HOME LOG_ROOT
printf '%s\n' "${tasks[@]}" | xargs -r -P "${CONVERSION_JOBS}" -n 1 bash -c 'convert_task "$1"' _

: >"${MANIFEST}"
for task in "${tasks[@]}"; do
  printf 'robotwin %s/%s_joint_v30\n' "${LEROBOT_ROOT}" "${task}" >>"${MANIFEST}"
done
[[ "$(wc -l <"${MANIFEST}")" -eq 50 ]]
echo "LeRobot clean-50 data ready: ${MANIFEST}"
