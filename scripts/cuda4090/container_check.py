#!/usr/bin/env python3
import os
from importlib.metadata import version

import torch

assert torch.cuda.is_available()
assert torch.cuda.device_count() == 8, torch.cuda.device_count()
assert torch.version.hip is None, torch.version.hip
assert torch.version.cuda is not None
assert tuple(map(int, torch.version.cuda.split(".")[:2])) >= (12, 8)

for index in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(index)
    assert props.major == 8 and props.minor == 9, (index, props.major, props.minor)
    assert props.total_memory >= 22 * 1024**3, (index, props.total_memory)

import flash_attn  # noqa: E402

print("torch", torch.__version__, "cuda", torch.version.cuda)
print("flash-attn", version("flash-attn"))
print("NCCL", torch.cuda.nccl.version())
print("GPUs", [torch.cuda.get_device_name(i) for i in range(8)])
print("ROCm variables", {k: os.environ[k] for k in os.environ if k.startswith(("ROCM", "HIP", "HSA"))})
assert not any(k.startswith(("ROCM", "HIP", "HSA")) for k in os.environ)
print("container check passed")
