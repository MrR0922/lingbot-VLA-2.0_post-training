#!/usr/bin/env bash
set -Eeuo pipefail

command -v nvidia-smi >/dev/null
command -v docker >/dev/null

mapfile -t gpu_rows < <(nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader,nounits)
if [[ "${#gpu_rows[@]}" -ne 8 ]]; then
  echo "Expected 8 GPUs, found ${#gpu_rows[@]}" >&2
  exit 1
fi
for row in "${gpu_rows[@]}"; do
  IFS=',' read -r name memory driver capability <<<"${row}"
  name="${name# }"; memory="${memory# }"; driver="${driver# }"; capability="${capability# }"
  [[ "${name}" == *"RTX 4090"* ]] || { echo "Unexpected GPU: ${name}" >&2; exit 1; }
  [[ "${driver%%.*}" -ge 570 ]] || { echo "Driver ${driver} is below 570" >&2; exit 1; }
  [[ "${capability}" == "8.9" ]] || { echo "Unexpected compute capability: ${capability}" >&2; exit 1; }
  (( ${memory%.*} >= 23000 )) || { echo "GPU memory ${memory} MiB is below the expected 24 GB class" >&2; exit 1; }
done

docker info >/dev/null
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 \
  nvidia-smi --query-gpu=name --format=csv,noheader | grep -c 'RTX 4090' | grep -qx '8'

for path in /models/robotwin-persistent /workspace/robotwin-runtime; do
  [[ -d "${path}" ]] || { echo "Missing host directory: ${path}" >&2; exit 1; }
done
available_kib=$(df --output=avail /models/robotwin-persistent | tail -1 | tr -d ' ')
(( available_kib >= 500 * 1024 * 1024 )) || {
  echo "Less than 500 GiB available under /models/robotwin-persistent" >&2
  exit 1
}

printf '%s\n' "Host check passed for 8x RTX 4090, CUDA 12.8 container access, and persistent storage."
