# LingBot-VLA 2.0 + RoboTwin clean-50 on 4/7/8×RTX 4090/5090

This setup is an NVIDIA CUDA 12.8 port of the pinned Radeon Cloud reproduction.
It uses an external data layout so models, datasets, logs, and checkpoints do
not become Docker image layers.

## 1. Host prerequisites

- Linux x86_64 with 4, 7, or 8 identical RTX 4090 (24 GB), or 4/8 RTX 5090 (32 GB) GPUs.
- NVIDIA driver 570 or newer.
- Docker with NVIDIA Container Toolkit.
- At least 500 GiB free under `/models/robotwin-persistent`.

Select one profile for every command. The default is `4x4090`.

| `PROFILE` | GPUs | FSDP shard | Grad accumulation | Global batch | Activation offload |
| --- | ---: | ---: | ---: | ---: | --- |
| `4x4090` | 4 | 4 | 2 | 8 | enabled |
| `7x4090` | 7 | 7 | 1 | 7 | enabled |
| `8x4090` | 8 | 8 | 1 | 8 | disabled |
| `4x5090` | 4 | 4 | 2 | 8 | disabled |
| `8x5090` | 8 | 8 | 1 | 8 | disabled |

Run, for example:

```bash
sudo mkdir -p /models/robotwin-persistent /workspace/robotwin-runtime
PROFILE=4x4090 bash scripts/cuda-multigpu/host_check.sh
```

## 2. Verify and stage the clean-50 archives

On the machine containing the downloaded RoboTwin ZIPs:

```bash
python3 scripts/cuda-multigpu/prepare_clean50.py \
  --archive-root /path/to/robotwin2_clean/dataset \
  --training-list assets/training_data/robotwin.txt \
  --output-manifest /tmp/clean50_archives.tsv \
  --stage-dir /models/robotwin-persistent/data/raw-clean \
  --zip-test
```

The script reads only the 50 clean entries selected by LingBot's official
RoboTwin list. It rejects missing, empty, ambiguous, or corrupt archives. It
hard-links files when possible and falls back to copying across filesystems.
These raw archives are retained as the auditable source set. The tested
XPolicyLab conversion helper fetches the corresponding official
`demo_clean.zip` HDF5 archives into the same persistent data tree before
creating LeRobot v3 datasets.

## 3. Build and start the container

```bash
PROFILE=4x4090 bash docker/cuda-multigpu/build.sh
PROFILE=4x4090 bash docker/cuda-multigpu/run.sh bash
```

The 4/7/8-card profiles of the same GPU model share one image. Build
once for RTX 4090 and once for RTX 5090 only if both GPU types are used. The
build selects FlashAttention code generation for Ada `sm_89` or Blackwell
`sm_120`; do not reuse the 4090 image on a 5090 host or vice versa.

The image pins:

- CUDA 12.8.1, Ubuntu 24.04, Python 3.12;
- PyTorch 2.9.1 cu128 and FlashAttention 2.8.4;
- RoboTwin `266f3aadf505a4f7fe9af0faa41a20f5f47cd123`;
- XPolicyLab `c37109c500be67d0dea6b36bf7337bbd26e763cd`;
- LingBot-VLA-v2 `951475ae1b1d87553e7dc47c97b53a3d695c0d13`;
- Radeon reproduction patches at `27b2f6fcb6c715e2cbb6db4431381f87c94b86ed`.

ROCm/AITER packages and AMD device variables are absent. CuRobo remains the
default NVIDIA planner; set `ROBOTWIN_EE_PLANNER=mplib` only for comparison.

## 4. Container checks and assets

Inside the container:

```bash
/opt/robotwin-env/bin/python \
  /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/scripts/cuda-multigpu/container_check.py

/opt/robotwin-env/bin/python -m torch.distributed.run \
  --standalone --nproc-per-node="${GPU_COUNT}" \
  /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/scripts/cuda-multigpu/nccl_check.py

bash /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/scripts/cuda-multigpu/download_models.sh
bash /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/scripts/cuda-multigpu/download_sim_assets.sh
```

## 5. Convert data and compute clean-only statistics

```bash
bash /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/scripts/cuda-multigpu/convert_clean50.sh
bash /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/scripts/cuda-multigpu/compute_norm_stats.sh
```

Expected outputs:

```text
/models/robotwin-persistent/data/lerobot-clean/<task>_joint_v30/
/models/robotwin-persistent/data/robotwin_clean_v30.txt
/models/robotwin-persistent/data/robotwin_clean.json
```

The manifest and normalization statistics contain clean data only.

## 6. Profile switching and staged training

Start with 20 steps:

```bash
PROFILE=4x4090 MAX_STEPS=20 SAVE_STEPS=20 \
  bash /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/scripts/cuda-multigpu/smoke.sh
```

Then run 100 and 500 steps while checking per-GPU peak memory and step time:

```bash
PROFILE=4x4090 MAX_STEPS=100 SAVE_STEPS=100 \
  bash /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/scripts/cuda-multigpu/train.sh

PROFILE=4x4090 MAX_STEPS=500 SAVE_STEPS=100 \
  bash /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/scripts/cuda-multigpu/train.sh
```

The baseline uses FSDP2 full shard, BF16, gradient checkpointing, AdamW,
micro-batch 1, and global batch 8. Do not switch to Muon or Distributed Muon
until the AdamW smoke run saves and restores a DCP checkpoint successfully.
The `4x4090` and experimental `7x4090` profiles enable activation offload for
the 24 GB cards. The 7-card profile uses global batch 7 because no integer
gradient accumulation can produce batch 8 from seven micro-batches. Its
20-step smoke test is a hard gate: keep at least 1 GiB free per GPU. To switch hardware, change only `PROFILE`;
the script selects GPU count, IDs, config, output directory, and image tag.

To resume an existing output directory:

```bash
PROFILE=4x4090 ENABLE_RESUME=true MAX_STEPS=1000 SAVE_STEPS=100 \
  bash /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/scripts/cuda-multigpu/train.sh
```

`ENABLE_RESUME=auto` is the default. A new profile writes to its own directory,
`/workspace/runtime/outputs/robotwin_clean_<profile>`, so checkpoints from
different world sizes are never mixed accidentally.

To resume from one exact DCP directory instead of the latest checkpoint, set:

```bash
PROFILE=4x4090 \
LOAD_CHECKPOINT_PATH=/workspace/runtime/outputs/robotwin_clean_4x4090/checkpoints/global_step_500 \
MAX_STEPS=1000 bash scripts/cuda-multigpu/train.sh
```

## 7. Checkpoint policy

Training checkpoints use PyTorch Distributed Checkpoint (DCP). Each save under
`checkpoints/global_step_<N>` contains sharded `model`, `optimizer`, and
per-rank `extra_state`, including the scheduler, dataloader, and RNG state.
`SAVE_STEPS` controls periodic saves; reaching `MAX_STEPS` always saves the
last step. DCP is the authoritative resumable format.

```bash
# List and verify the latest checkpoint.
PROFILE=4x4090 bash scripts/cuda-multigpu/checkpoint.sh list
PROFILE=4x4090 bash scripts/cuda-multigpu/checkpoint.sh verify

# Export the latest valid DCP checkpoint to <checkpoint>/hf_ckpt.
PROFILE=4x4090 bash scripts/cuda-multigpu/checkpoint.sh export-hf
```

HF conversion is disabled during normal training to avoid pausing every save
and duplicating large weights. Set `SAVE_HF_WEIGHTS=true` only when an HF copy
is required at every checkpoint. Prefer the explicit `export-hf` command for
selected milestones. Keep DCP for resume and use `hf_ckpt` for inference.

## 8. Closed-loop evaluation

After a successful DCP save/restore test, convert the selected full-SFT DCP
checkpoint into a deployable Hugging Face directory:

```bash
PROFILE=4x4090 bash \
  /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/scripts/cuda-multigpu/checkpoint.sh export-hf
```

Start the CUDA server on GPU 0 (do not use the copied Radeon launcher):

```bash
bash /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/scripts/cuda-multigpu/launch_server.sh \
  0 13400 /workspace/runtime/outputs/logs/hf_server.log False \
  /workspace/runtime/outputs/robotwin_clean_4x4090/checkpoints/global_step_<N>/hf_ckpt
```

In a second container shell, evaluate 10 clean `adjust_bottle` episodes on GPU
1. CuRobo is the NVIDIA default; use `ROBOTWIN_EE_PLANNER=mplib` to compare the
fallback planner.

```bash
TEST_NUM=10 DEVICE_ID=1 \
  bash /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/scripts/cuda-multigpu/eval_adjust_bottle.sh
```

The server and simulator intentionally use different GPUs. Evaluation results
are persisted under `/workspace/runtime/eval_result`.
