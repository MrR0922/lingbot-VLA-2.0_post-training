#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
source "${REPOSITORY_ROOT}/scripts/cuda-multigpu/profile.sh"
PERSISTENT_ROOT="${PERSISTENT_ROOT:-/models/robotwin-persistent}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/workspace/robotwin-runtime}"
CONTAINER_NAME="${CONTAINER_NAME:-robotwin-lingbot-${PROFILE}}"

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
  -e PROFILE="${PROFILE}" \
  -v "${PERSISTENT_ROOT}:/models/robotwin-persistent" \
  -v "${RUNTIME_ROOT}:/workspace/runtime" \
  "${IMAGE_NAME}" "$@"
