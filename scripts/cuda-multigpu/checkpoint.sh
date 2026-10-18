#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/profile.sh"
ACTION="${1:-list}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/runtime/outputs/robotwin_clean_${PROFILE}}"
CHECKPOINT_ROOT="${OUTPUT_DIR}/checkpoints"

latest_checkpoint() {
  find "${CHECKPOINT_ROOT}" -mindepth 1 -maxdepth 1 -type d -name 'global_step_*' -printf '%f\t%p\n' 2>/dev/null \
    | sort -t_ -k3,3n | tail -1 | cut -f2-
}

case "${ACTION}" in
  list)
    find "${CHECKPOINT_ROOT}" -mindepth 1 -maxdepth 1 -type d -name 'global_step_*' -printf '%f\n' 2>/dev/null \
      | sort -t_ -k3,3n
    ;;
  latest)
    latest_checkpoint
    ;;
  verify)
    checkpoint="${2:-$(latest_checkpoint)}"
    [[ -n "${checkpoint}" ]] || { echo "No checkpoint found under ${CHECKPOINT_ROOT}" >&2; exit 1; }
    for part in model optimizer; do
      [[ -f "${checkpoint}/${part}/.metadata" ]] || { echo "Incomplete DCP component: ${checkpoint}/${part}" >&2; exit 1; }
    done
    compgen -G "${checkpoint}/extra_state/extra_state_rank_*.pt" >/dev/null || {
      echo "Missing rank extra_state files: ${checkpoint}/extra_state" >&2
      exit 1
    }
    echo "DCP checkpoint is complete: ${checkpoint}"
    ;;
  export-hf)
    checkpoint="${2:-$(latest_checkpoint)}"
    [[ -n "${checkpoint}" ]] || { echo "No checkpoint found under ${CHECKPOINT_ROOT}" >&2; exit 1; }
    hf_output="${3:-${checkpoint}/hf_ckpt}"
    "${SCRIPT_DIR}/export_hf.sh" "${OUTPUT_DIR}" "${checkpoint}" "${hf_output}"
    ;;
  audit-update)
    before="${2:?usage: checkpoint.sh audit-update BEFORE_MODEL AFTER_HF [REPORT]}"
    after="${3:?usage: checkpoint.sh audit-update BEFORE_MODEL AFTER_HF [REPORT]}"
    report="${4:-${OUTPUT_DIR}/checkpoint_update_audit.json}"
    "${PYTHON:-/opt/robotwin-env/bin/python}" "${SCRIPT_DIR}/audit_hf_update.py" \
      --before "${before}" \
      --after "${after}" \
      --report "${report}"
    ;;
  *)
    echo "usage: checkpoint.sh {list|latest|verify|export-hf|audit-update} ..." >&2
    exit 2
    ;;
esac
