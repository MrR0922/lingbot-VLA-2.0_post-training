#!/usr/bin/env bash
set -Eeuo pipefail

SOURCE_DIR="${LINGBOT_VLA_SOURCE:-/RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/source/lingbot-vla-v2}"
CONFIG="${CONFIG:-/RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/training/robotwin_8x4090.yaml}"
PYTHON="${PYTHON:-/opt/robotwin-env/bin/python}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
MAX_STEPS="${MAX_STEPS:-100}"
SAVE_STEPS="${SAVE_STEPS:-${MAX_STEPS}}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/runtime/outputs/robotwin_clean_8x4090_${MAX_STEPS}steps}"
OPTIMIZER="${OPTIMIZER:-adamw}"
ENABLE_RESUME="${ENABLE_RESUME:-false}"

[[ "$(awk -F, '{print NF}' <<<"${GPU_IDS}")" -eq 8 ]] || { echo "GPU_IDS must contain 8 IDs" >&2; exit 2; }
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
[[ "$("${PYTHON}" -c 'import torch; print(torch.cuda.device_count())')" -eq 8 ]] || {
  echo "PyTorch must see exactly 8 CUDA GPUs" >&2
  exit 1
}

mkdir -p "${OUTPUT_DIR}" /workspace/runtime/outputs/logs
cd "${SOURCE_DIR}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
unset HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES HSA_OVERRIDE_GFX_VERSION

"${PYTHON}" -m torch.distributed.run --standalone --nproc-per-node=8 \
  -m tasks.vla.train_lingbotvla "${CONFIG}" \
  --train.output_dir "${OUTPUT_DIR}" \
  --train.optimizer "${OPTIMIZER}" \
  --train.max_steps "${MAX_STEPS}" \
  --train.save_steps "${SAVE_STEPS}" \
  --train.enable_resume "${ENABLE_RESUME}" \
  2>&1 | tee "/workspace/runtime/outputs/logs/train_${MAX_STEPS}steps.log"
