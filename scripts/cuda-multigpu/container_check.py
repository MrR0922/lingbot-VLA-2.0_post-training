#!/usr/bin/env python3
import os
from importlib.metadata import version

import torch

gpu_count = int(os.environ.get("GPU_COUNT", "4"))
gpu_cc = tuple(map(int, os.environ.get("GPU_CC", "8.9").split(".")))
gpu_min_gib = int(os.environ.get("GPU_MIN_GIB", "22"))
assert torch.cuda.is_available()
assert torch.cuda.device_count() >= gpu_count, torch.cuda.device_count()
assert torch.version.hip is None, torch.version.hip
assert torch.version.cuda is not None
assert tuple(map(int, torch.version.cuda.split(".")[:2])) >= (12, 8)

for index in range(gpu_count):
    props = torch.cuda.get_device_properties(index)
    assert (props.major, props.minor) == gpu_cc, (index, props.major, props.minor)
    assert props.total_memory >= gpu_min_gib * 1024**3, (index, props.total_memory)

import flash_attn  # noqa: E402

print("torch", torch.__version__, "cuda", torch.version.cuda)
print("flash-attn", version("flash-attn"))
print("NCCL", torch.cuda.nccl.version())
print("GPUs", [torch.cuda.get_device_name(i) for i in range(gpu_count)])
print("ROCm variables", {k: os.environ[k] for k in os.environ if k.startswith(("ROCM", "HIP", "HSA"))})
assert not any(k.startswith(("ROCM", "HIP", "HSA")) for k in os.environ)
print("container check passed")
