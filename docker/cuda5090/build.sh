#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-robotwin-lingbot-vla-v2:cuda12.8-py3.12-torch2.9.1-rtx5090}"
BASE_IMAGE="${BASE_IMAGE:-nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04}"

build_args=(
  --file "${SCRIPT_DIR}/Dockerfile"
  --tag "${IMAGE_NAME}"
  --build-arg "BASE_IMAGE=${BASE_IMAGE}"
  --progress plain
)
if [[ -n "${PIP_INDEX_URL:-}" ]]; then
  build_args+=(--build-arg "PIP_INDEX_URL=${PIP_INDEX_URL}")
fi

DOCKER_BUILDKIT=1 docker build "${build_args[@]}" "${REPOSITORY_ROOT}"
echo "Built ${IMAGE_NAME}"
