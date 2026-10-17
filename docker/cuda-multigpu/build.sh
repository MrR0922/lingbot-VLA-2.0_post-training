#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
source "${REPOSITORY_ROOT}/scripts/cuda-multigpu/profile.sh"
BASE_IMAGE="${BASE_IMAGE:-nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04}"

build_args=(
  --file "${SCRIPT_DIR}/Dockerfile"
  --tag "${IMAGE_NAME}"
  --build-arg "BASE_IMAGE=${BASE_IMAGE}"
  --build-arg "CUDA_ARCH_LIST=${CUDA_ARCH_LIST}"
  --progress plain
)
if [[ -n "${PIP_INDEX_URL:-}" ]]; then
  build_args+=(--build-arg "PIP_INDEX_URL=${PIP_INDEX_URL}")
fi

DOCKER_BUILDKIT=1 docker build "${build_args[@]}" "${REPOSITORY_ROOT}"
echo "Built ${IMAGE_NAME}"
