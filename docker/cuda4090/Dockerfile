# syntax=docker/dockerfile:1.7
ARG BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="LingBot-VLA-v2 RoboTwin CUDA 12.8 external-data runtime" \
      org.opencontainers.image.description="Training and evaluation runtime for 8x RTX 4090"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    TORCH_CUDA_ARCH_LIST="8.9+PTX" \
    MAX_JOBS=16 \
    HF_HOME=/workspace/runtime/.cache/huggingface \
    HF_LEROBOT_HOME=/RoboTwin/data/lerobot-clean \
    LEROBOT_VIDEO_BACKEND=pyav \
    LEROBOT_VIDEO_CODEC=h264 \
    PYOPENGL_PLATFORM=egl \
    TOKENIZERS_PARALLELISM=false \
    ROBOTWIN_ROOT=/RoboTwin \
    LINGBOT_VLA_SOURCE=/RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/source/lingbot-vla-v2 \
    LINGBOT_VLA_PYTHON=/opt/robotwin-env/bin/python \
    QWEN3VL_PATH=/models/robotwin-persistent/models/Qwen3-VL-4B-Instruct-config-tokenizer

ARG PIP_INDEX_URL=
ARG ROBOTWIN_COMMIT=266f3aadf505a4f7fe9af0faa41a20f5f47cd123
ARG XPOLICYLAB_COMMIT=c37109c500be67d0dea6b36bf7337bbd26e763cd
ARG LINGBOT_VLA_COMMIT=951475ae1b1d87553e7dc47c97b53a3d695c0d13
ARG REPRO_COMMIT=27b2f6fcb6c715e2cbb6db4431381f87c94b86ed

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
      ca-certificates git git-lfs curl wget ffmpeg unzip \
      libgl1 libglib2.0-0 libvulkan1 vulkan-tools mesa-vulkan-drivers \
      build-essential ninja-build cmake pkg-config openssh-client \
      python3.12 python3.12-venv python3-pip && \
    git lfs install --system

# The Radeon reproduction pins the integration dependencies and contains the
# LeRobot/LoRA/path compatibility patches. Hardware-specific ROCm packages are
# deliberately not installed here.
RUN git clone https://github.com/ZiguanWang/Robotwin-radeon-cloud.git /opt/robotwin-repro && \
    git -C /opt/robotwin-repro checkout "${REPRO_COMMIT}" && \
    test "$(git -C /opt/robotwin-repro rev-parse HEAD)" = "${REPRO_COMMIT}"

RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python3.12 -m venv /opt/robotwin-env && \
    if [[ -n "${PIP_INDEX_URL}" ]]; then /opt/robotwin-env/bin/python -m pip config set global.index-url "${PIP_INDEX_URL}"; fi && \
    /opt/robotwin-env/bin/python -m pip install --upgrade 'pip<26' 'setuptools<81' wheel && \
    /opt/robotwin-env/bin/python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
      torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 && \
    /opt/robotwin-env/bin/python -m pip install --prefer-binary \
      -r /opt/robotwin-repro/docker/requirements.txt && \
    /opt/robotwin-env/bin/python -m pip install --no-deps \
      -r /opt/robotwin-repro/docker/requirements-no-deps.txt && \
    /opt/robotwin-env/bin/python -m pip install --prefer-binary open3d==0.19.0

# Keep conversion dependencies isolated from the model runtime.
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python3.12 -m venv /opt/lerobot-env && \
    /opt/lerobot-env/bin/python -m pip install --upgrade 'pip<26' 'setuptools<81' wheel && \
    /opt/lerobot-env/bin/python -m pip install --prefer-binary \
      'lerobot[dataset]==0.6.0' 'h5py==3.14.0'

# Build NVIDIA FlashAttention against the CUDA 12.8/PyTorch environment. The
# architecture list embeds native Ada code and PTX fallback.
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    /opt/robotwin-env/bin/python -m pip install --no-build-isolation flash-attn==2.8.4

RUN git clone --recurse-submodules https://github.com/RoboTwin-Platform/RoboTwin.git /RoboTwin && \
    git -C /RoboTwin checkout "${ROBOTWIN_COMMIT}" && \
    git -C /RoboTwin submodule update --init --recursive && \
    test "$(git -C /RoboTwin rev-parse HEAD)" = "${ROBOTWIN_COMMIT}" && \
    test "$(git -C /RoboTwin/XPolicyLab rev-parse HEAD)" = "${XPOLICYLAB_COMMIT}" && \
    git -C /RoboTwin apply /opt/robotwin-repro/docker/patches/robotwin-rocm-reproduction.patch && \
    git -C /RoboTwin/XPolicyLab apply /opt/robotwin-repro/docker/patches/xpolicylab-lerobot-v30.patch && \
    mkdir -p /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/source && \
    git clone https://github.com/robbyant/lingbot-vla-v2.git \
      /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/source/lingbot-vla-v2 && \
    git -C /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/source/lingbot-vla-v2 \
      checkout "${LINGBOT_VLA_COMMIT}" && \
    git -C /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/source/lingbot-vla-v2 \
      apply /opt/robotwin-repro/docker/patches/lingbot-vla-v2-rocm.patch && \
    cp -a /opt/robotwin-repro/docker/assets/experiments/lingbot_vla_v2_6b_robotwin/. \
      /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/ && \
    /opt/robotwin-env/bin/python -m pip install -e /RoboTwin/XPolicyLab && \
    /opt/robotwin-env/bin/python -m pip install --no-deps -e \
      /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/source/lingbot-vla-v2

COPY configs/vla/robotwin/robotwin_8x4090.yaml \
  /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/training/robotwin_8x4090.yaml
COPY scripts/cuda4090/ /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/scripts/cuda4090/
COPY docker/cuda4090/entrypoint.sh /usr/local/bin/robotwin-cuda-entrypoint

RUN chmod 0755 /usr/local/bin/robotwin-cuda-entrypoint \
      /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/scripts/cuda4090/*.sh && \
    rm -rf /RoboTwin/assets /RoboTwin/data /RoboTwin/eval_result \
      /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/models && \
    mkdir -p /models/robotwin-persistent /workspace/runtime/eval_result && \
    ln -s /models/robotwin-persistent/assets /RoboTwin/assets && \
    ln -s /models/robotwin-persistent/data /RoboTwin/data && \
    ln -s /workspace/runtime/eval_result /RoboTwin/eval_result && \
    ln -s /models/robotwin-persistent/models \
      /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/models

RUN /opt/robotwin-env/bin/python - <<'PY'
import torch
import flash_attn
assert torch.version.cuda is not None, torch.__version__
assert torch.version.hip is None
assert tuple(map(int, torch.version.cuda.split('.')[:2])) >= (12, 8)
print("torch", torch.__version__, "CUDA", torch.version.cuda, "flash-attn", flash_attn.__version__)
PY

WORKDIR /RoboTwin
ENTRYPOINT ["/usr/local/bin/robotwin-cuda-entrypoint"]
CMD ["bash"]
