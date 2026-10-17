#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/profile.sh"

command -v nvidia-smi >/dev/null
command -v docker >/dev/null

mapfile -t gpu_rows < <(nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader,nounits)
if [[ "${#gpu_rows[@]}" -lt "${GPU_COUNT}" ]]; then
  echo "Expected at least ${GPU_COUNT} GPUs, found ${#gpu_rows[@]}" >&2
  exit 1
fi
for row in "${gpu_rows[@]:0:${GPU_COUNT}}"; do
  IFS=',' read -r name memory driver capability <<<"${row}"
  name="${name# }"; memory="${memory# }"; driver="${driver# }"; capability="${capability# }"
  [[ "${name}" == *"${GPU_MODEL}"* ]] || { echo "Unexpected GPU for ${PROFILE}: ${name}" >&2; exit 1; }
  [[ "${driver%%.*}" -ge 570 ]] || { echo "Driver ${driver} is below 570" >&2; exit 1; }
  [[ "${capability}" == "${GPU_CC}" ]] || { echo "Unexpected compute capability: ${capability}" >&2; exit 1; }
  (( ${memory%.*} >= GPU_MIN_GIB * 1024 )) || { echo "GPU memory ${memory} MiB is below ${GPU_MIN_GIB} GiB" >&2; exit 1; }
done

docker info >/dev/null
container_gpu_count=$(docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 \
  nvidia-smi --query-gpu=name --format=csv,noheader | grep -c "${GPU_MODEL}")
(( container_gpu_count >= GPU_COUNT )) || {
  echo "Container sees ${container_gpu_count} matching GPUs, expected ${GPU_COUNT}" >&2
  exit 1
}

for path in /models/robotwin-persistent /workspace/robotwin-runtime; do
  [[ -d "${path}" ]] || { echo "Missing host directory: ${path}" >&2; exit 1; }
done
available_kib=$(df --output=avail /models/robotwin-persistent | tail -1 | tr -d ' ')
(( available_kib >= 500 * 1024 * 1024 )) || {
  echo "Less than 500 GiB available under /models/robotwin-persistent" >&2
  exit 1
}

printf '%s\n' "Host check passed for ${PROFILE}, CUDA 12.8 container access, and persistent storage."
