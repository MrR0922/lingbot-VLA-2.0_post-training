#!/usr/bin/env bash
set -Eeuo pipefail

IMAGE_NAME="${IMAGE_NAME:-robotwin-lingbot-vla-v2:cuda12.8-py3.12-torch2.9.1-rtx5090}"
PERSISTENT_ROOT="${PERSISTENT_ROOT:-/models/robotwin-persistent}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/workspace/robotwin-runtime}"
CONTAINER_NAME="${CONTAINER_NAME:-robotwin-lingbot-cuda5090}"

mkdir -p "${RUNTIME_ROOT}"
docker run --rm -it \
  --name "${CONTAINER_NAME}" \
  --gpus all \
  --ipc=host \
  --shm-size=64g \
  --network=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e REQUIRE_PREPARED_DATA="${REQUIRE_PREPARED_DATA:-0}" \
  -v "${PERSISTENT_ROOT}:/models/robotwin-persistent" \
  -v "${RUNTIME_ROOT}:/workspace/runtime" \
  "${IMAGE_NAME}" "$@"
