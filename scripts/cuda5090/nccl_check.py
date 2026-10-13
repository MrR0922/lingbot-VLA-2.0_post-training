#!/usr/bin/env python3
import os

import torch
import torch.distributed as dist

dist.init_process_group("nccl")
rank = dist.get_rank()
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
value = torch.tensor([float(rank + 1)], device="cuda")
dist.all_reduce(value)
expected = dist.get_world_size() * (dist.get_world_size() + 1) / 2
assert value.item() == expected, (rank, value.item(), expected)
torch.cuda.synchronize()
if rank == 0:
    print(f"NCCL all-reduce passed on {dist.get_world_size()} GPUs")
dist.destroy_process_group()
