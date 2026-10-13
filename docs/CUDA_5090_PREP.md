# LingBot-VLA 2.0 + RoboTwin clean-50 on 8×RTX 5090

This setup is an NVIDIA CUDA 12.8 port of the pinned Radeon Cloud reproduction.
It uses an external data layout so models, datasets, logs, and checkpoints do
not become Docker image layers.

## 1. Host prerequisites

- Linux x86_64 with eight RTX 5090 GPUs.
- NVIDIA driver 570 or newer.
- Docker with NVIDIA Container Toolkit.
- At least 500 GiB free under `/models/robotwin-persistent`.

Run:

```bash
sudo mkdir -p /models/robotwin-persistent /workspace/robotwin-runtime
bash scripts/cuda5090/host_check.sh
```

## 2. Verify and stage the clean-50 archives

On the machine containing the downloaded RoboTwin ZIPs:

```bash
python3 scripts/cuda5090/prepare_clean50.py \
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
bash docker/cuda5090/build.sh
bash docker/cuda5090/run.sh bash
```

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
  /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/scripts/cuda5090/container_check.py

/opt/robotwin-env/bin/python -m torch.distributed.run \
  --standalone --nproc-per-node=8 \
  /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/scripts/cuda5090/nccl_check.py

bash /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/scripts/cuda5090/download_models.sh
bash /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/scripts/cuda5090/download_sim_assets.sh
```

## 5. Convert data and compute clean-only statistics

```bash
bash /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/scripts/cuda5090/convert_clean50.sh
bash /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/scripts/cuda5090/compute_norm_stats.sh
```

Expected outputs:

```text
/models/robotwin-persistent/data/lerobot-clean/<task>_joint_v30/
/models/robotwin-persistent/data/robotwin_clean_v30.txt
/models/robotwin-persistent/data/robotwin_clean.json
```

The manifest and normalization statistics contain clean data only.

## 6. Smoke and staged training

Start with 20 steps:

```bash
MAX_STEPS=20 SAVE_STEPS=20 \
  bash /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/scripts/cuda5090/smoke.sh
```

Then run 100 and 500 steps while checking per-GPU peak memory and step time:

```bash
MAX_STEPS=100 SAVE_STEPS=100 \
  bash /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/scripts/cuda5090/train_8x5090.sh

MAX_STEPS=500 SAVE_STEPS=500 \
  OUTPUT_DIR=/workspace/runtime/outputs/robotwin_clean_500steps \
  bash /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/scripts/cuda5090/train_8x5090.sh
```

The baseline uses FSDP2 full shard, BF16, gradient checkpointing, AdamW,
micro-batch 1, and global batch 8. Do not switch to Muon or Distributed Muon
until the AdamW smoke run saves and restores a DCP checkpoint successfully.

To resume an existing output directory:

```bash
ENABLE_RESUME=true OUTPUT_DIR=/workspace/runtime/outputs/robotwin_clean_500steps \
  MAX_STEPS=1000 SAVE_STEPS=500 \
  bash /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/scripts/cuda5090/train_8x5090.sh
```

## 7. Export and closed-loop evaluation

After a successful DCP save/restore test, convert the selected full-SFT DCP
checkpoint into a deployable Hugging Face directory:

```bash
bash /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/scripts/cuda5090/export_hf.sh \
  /workspace/runtime/outputs/robotwin_clean_500steps \
  /workspace/runtime/outputs/robotwin_clean_500steps/<dcp-checkpoint> \
  /workspace/runtime/outputs/robotwin_clean_500steps/hf_ckpt
```

Start the CUDA server on GPU 0 (do not use the copied Radeon launcher):

```bash
bash /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/scripts/cuda5090/launch_server.sh \
  0 13400 /workspace/runtime/outputs/logs/hf_server.log False \
  /workspace/runtime/outputs/robotwin_clean_500steps/hf_ckpt
```

In a second container shell, evaluate 10 clean `adjust_bottle` episodes on GPU
1. CuRobo is the NVIDIA default; use `ROBOTWIN_EE_PLANNER=mplib` to compare the
fallback planner.

```bash
TEST_NUM=10 DEVICE_ID=1 \
  bash /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/scripts/cuda5090/eval_adjust_bottle.sh
```

The server and simulator intentionally use different GPUs. Evaluation results
are persisted under `/workspace/runtime/eval_result`.
