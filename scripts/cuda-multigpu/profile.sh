#!/usr/bin/env bash
# Source this file from CUDA launch/check scripts. PROFILE may be overridden.

PROFILE="${PROFILE:-4x4090}"
case "${PROFILE}" in
  4x4090)
    GPU_COUNT=4; GPU_MODEL="RTX 4090"; GPU_CC="8.9"; GPU_MIN_GIB=22
    CUDA_ARCH_LIST="8.9+PTX"; GPU_SKU=4090; CONFIG_NAME="robotwin_4x4090.yaml"
    ;;
  8x4090)
    GPU_COUNT=8; GPU_MODEL="RTX 4090"; GPU_CC="8.9"; GPU_MIN_GIB=22
    CUDA_ARCH_LIST="8.9+PTX"; GPU_SKU=4090; CONFIG_NAME="robotwin_8x4090.yaml"
    ;;
  4x5090)
    GPU_COUNT=4; GPU_MODEL="RTX 5090"; GPU_CC="12.0"; GPU_MIN_GIB=30
    CUDA_ARCH_LIST="12.0+PTX"; GPU_SKU=5090; CONFIG_NAME="robotwin_4x5090.yaml"
    ;;
  8x5090)
    GPU_COUNT=8; GPU_MODEL="RTX 5090"; GPU_CC="12.0"; GPU_MIN_GIB=30
    CUDA_ARCH_LIST="12.0+PTX"; GPU_SKU=5090; CONFIG_NAME="robotwin_8x5090.yaml"
    ;;
  *)
    echo "Unknown PROFILE=${PROFILE}; choose 4x4090, 8x4090, 4x5090, or 8x5090" >&2
    return 2 2>/dev/null || exit 2
    ;;
esac

GPU_IDS="${GPU_IDS:-$(seq -s, 0 $((GPU_COUNT - 1)))}"
CONFIG="${CONFIG:-/RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/training/${CONFIG_NAME}}"
IMAGE_NAME="${IMAGE_NAME:-robotwin-lingbot-vla-v2:cuda12.8-rtx${GPU_SKU}}"
export PROFILE GPU_COUNT GPU_MODEL GPU_CC GPU_MIN_GIB CUDA_ARCH_LIST GPU_SKU CONFIG_NAME GPU_IDS CONFIG IMAGE_NAME
