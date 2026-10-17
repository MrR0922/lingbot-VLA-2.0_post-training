#!/usr/bin/env bash
set -Eeuo pipefail

MODEL_ROOT="${MODEL_ROOT:-/models/robotwin-persistent/models}"
HF_CLI="${HF_CLI:-/opt/robotwin-env/bin/huggingface-cli}"
mkdir -p "${MODEL_ROOT}"

"${HF_CLI}" download robbyant/lingbot-vla-v2-6b \
  --revision 11c703bf6a5c1f45b3b69168482da11fdbba53d7 \
  --local-dir "${MODEL_ROOT}/robbyant_lingbot-vla-v2-6b"
"${HF_CLI}" download Qwen/Qwen3-VL-4B-Instruct \
  --revision ebb281ec70b05090aa6165b016eac8ec08e71b17 \
  --include '*.json' '*.txt' '*.jinja' merges.txt vocab.json \
  --local-dir "${MODEL_ROOT}/Qwen3-VL-4B-Instruct-config-tokenizer"
"${HF_CLI}" download Ruicheng/moge-2-vitb-normal \
  --revision ca5f0e07ff01d3e5a364c1d954ed12ee1814b368 \
  --local-dir "${MODEL_ROOT}/moge-2-vitb-normal"

required=(
  "${MODEL_ROOT}/robbyant_lingbot-vla-v2-6b/model.safetensors.index.json"
  "${MODEL_ROOT}/robbyant_lingbot-vla-v2-6b/depth/model.pt"
  "${MODEL_ROOT}/robbyant_lingbot-vla-v2-6b/dino_video/teacher_step_10000.pth"
  "${MODEL_ROOT}/robbyant_lingbot-vla-v2-6b/dino_video/config.yaml"
  "${MODEL_ROOT}/moge-2-vitb-normal/model.pt"
)
for path in "${required[@]}"; do
  [[ -f "${path}" ]] || { echo "Missing downloaded model file: ${path}" >&2; exit 1; }
done
echo "Model assets ready under ${MODEL_ROOT}"
