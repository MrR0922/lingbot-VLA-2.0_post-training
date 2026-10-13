#!/usr/bin/env bash
set -Eeuo pipefail

TRAINING_OUTPUT="${1:?usage: export_hf.sh TRAINING_OUTPUT DCP_CHECKPOINT [HF_OUTPUT]}"
CHECKPOINT="${2:?usage: export_hf.sh TRAINING_OUTPUT DCP_CHECKPOINT [HF_OUTPUT]}"
HF_OUTPUT="${3:-${TRAINING_OUTPUT}/hf_ckpt}"
PYTHON="${PYTHON:-/opt/robotwin-env/bin/python}"
CONVERTER="/RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/scripts/convert_full_sft_dcp.py"

[[ -d "${TRAINING_OUTPUT}/model_assets" ]] || { echo "Missing model_assets in ${TRAINING_OUTPUT}" >&2; exit 1; }
[[ -e "${CHECKPOINT}" ]] || { echo "Missing DCP checkpoint: ${CHECKPOINT}" >&2; exit 1; }
[[ ! -e "${HF_OUTPUT}" ]] || { echo "HF output already exists: ${HF_OUTPUT}" >&2; exit 1; }

"${PYTHON}" "${CONVERTER}" \
  --checkpoint "${CHECKPOINT}" \
  --training-output "${TRAINING_OUTPUT}" \
  --output "${HF_OUTPUT}"

echo "Deployable hf_ckpt: ${HF_OUTPUT}"
