#!/usr/bin/env bash
set -Eeuo pipefail

SOURCE_DIR="${LINGBOT_VLA_SOURCE:-/RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/source/lingbot-vla-v2}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/profile.sh"
PYTHON="${PYTHON:-/opt/robotwin-env/bin/python}"
MAX_STEPS="${MAX_STEPS:-100}"
SAVE_STEPS="${SAVE_STEPS:-${MAX_STEPS}}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/runtime/outputs/robotwin_clean_${PROFILE}}"
OPTIMIZER="${OPTIMIZER:-adamw}"
ENABLE_RESUME="${ENABLE_RESUME:-auto}"
SAVE_HF_WEIGHTS="${SAVE_HF_WEIGHTS:-false}"
LOAD_CHECKPOINT_PATH="${LOAD_CHECKPOINT_PATH:-}"
FREEZE_VISION_ENCODER="${FREEZE_VISION_ENCODER:-false}"
TRAIN_EXPERT_ONLY="${TRAIN_EXPERT_ONLY:-false}"

[[ "$(awk -F, '{print NF}' <<<"${GPU_IDS}")" -eq "${GPU_COUNT}" ]] || { echo "GPU_IDS must contain ${GPU_COUNT} IDs" >&2; exit 2; }
for path in \
  /models/robotwin-persistent/data/robotwin_clean_v30.txt \
  /models/robotwin-persistent/data/robotwin_clean.json \
  /models/robotwin-persistent/models/robbyant_lingbot-vla-v2-6b/model.safetensors.index.json \
  /models/robotwin-persistent/models/robbyant_lingbot-vla-v2-6b/depth/model.pt \
  /models/robotwin-persistent/models/robbyant_lingbot-vla-v2-6b/dino_video/teacher_step_10000.pth \
  /models/robotwin-persistent/models/robbyant_lingbot-vla-v2-6b/dino_video/config.yaml \
  /models/robotwin-persistent/models/moge-2-vitb-normal/model.pt; do
  [[ -e "${path}" ]] || { echo "Missing training prerequisite: ${path}" >&2; exit 1; }
done
[[ "$(grep -cve '^[[:space:]]*$' /models/robotwin-persistent/data/robotwin_clean_v30.txt)" -eq 50 ]] || {
  echo "Clean manifest must contain exactly 50 non-empty entries" >&2
  exit 1
}
[[ "$("${PYTHON}" -c 'import torch; print(torch.cuda.device_count())')" -ge "${GPU_COUNT}" ]] || {
  echo "PyTorch must see at least ${GPU_COUNT} CUDA GPUs" >&2
  exit 1
}

if [[ "${ENABLE_RESUME}" == "auto" ]]; then
  if compgen -G "${OUTPUT_DIR}/checkpoints/global_step_*" >/dev/null; then ENABLE_RESUME=true; else ENABLE_RESUME=false; fi
fi

mkdir -p "${OUTPUT_DIR}" /workspace/runtime/outputs/logs
cd "${SOURCE_DIR}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
unset HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES HSA_OVERRIDE_GFX_VERSION

echo "profile=${PROFILE} gpus=${GPU_IDS} output=${OUTPUT_DIR} resume=${ENABLE_RESUME} save_steps=${SAVE_STEPS}"
checkpoint_args=()
if [[ -n "${LOAD_CHECKPOINT_PATH}" ]]; then
  [[ -d "${LOAD_CHECKPOINT_PATH}" ]] || { echo "Missing checkpoint: ${LOAD_CHECKPOINT_PATH}" >&2; exit 1; }
  checkpoint_args+=(--train.load_checkpoint_path "${LOAD_CHECKPOINT_PATH}")
fi
"${PYTHON}" -m torch.distributed.run --standalone --nproc-per-node="${GPU_COUNT}" \
  -m tasks.vla.train_lingbotvla "${CONFIG}" \
  --train.output_dir "${OUTPUT_DIR}" \
  --train.optimizer "${OPTIMIZER}" \
  --train.max_steps "${MAX_STEPS}" \
  --train.save_steps "${SAVE_STEPS}" \
  --train.ckpt_manager dcp \
  --train.save_hf_weights "${SAVE_HF_WEIGHTS}" \
  --train.enable_resume "${ENABLE_RESUME}" \
  --train.freeze_vision_encoder "${FREEZE_VISION_ENCODER}" \
  --train.train_expert_only "${TRAIN_EXPERT_ONLY}" \
  "${checkpoint_args[@]}" \
  2>&1 | tee "/workspace/runtime/outputs/logs/train_${PROFILE}_${MAX_STEPS}steps.log"
