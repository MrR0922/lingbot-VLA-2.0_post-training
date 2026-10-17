#!/usr/bin/env bash
set -Eeuo pipefail

ASSET_ROOT="${ASSET_ROOT:-/models/robotwin-persistent/assets}"
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-/workspace/runtime/downloads/robotwin-assets}"
HF_CLI="${HF_CLI:-/opt/robotwin-env/bin/huggingface-cli}"
REVISION="${REVISION:-a967b852afa21a9cbf19a198f7e653109042e87c}"

mkdir -p "${ASSET_ROOT}" "${DOWNLOAD_ROOT}"
"${HF_CLI}" download TianxingChen/RoboTwin2.0 \
  --repo-type dataset \
  --revision "${REVISION}" \
  --include background_texture.zip embodiments.zip objects.zip \
  --local-dir "${DOWNLOAD_ROOT}"

for archive in background_texture.zip embodiments.zip objects.zip; do
  unzip -q -o "${DOWNLOAD_ROOT}/${archive}" -d "${ASSET_ROOT}"
done
[[ -f "${ASSET_ROOT}/objects/objaverse/list.json" ]] || {
  echo "Simulation assets are incomplete under ${ASSET_ROOT}" >&2
  exit 1
}
echo "RoboTwin simulation assets ready under ${ASSET_ROOT}"
