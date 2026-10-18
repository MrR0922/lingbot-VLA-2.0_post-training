#!/usr/bin/env python3
"""Verify that an exported training checkpoint changed real model tensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open


def tensor_map(model_dir: Path) -> dict[str, Path]:
    index = model_dir / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text())["weight_map"]
        return {name: model_dir / filename for name, filename in weight_map.items()}

    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no safetensors weights under {model_dir}")
    result: dict[str, Path] = {}
    for filename in files:
        with safe_open(filename, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name in result:
                    raise RuntimeError(f"duplicate tensor {name!r} in {model_dir}")
                result[name] = filename
    return result


def read_tensor(filename: Path, name: str) -> torch.Tensor:
    with safe_open(filename, framework="pt", device="cpu") as handle:
        return handle.get_tensor(name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--require-pattern", default="qwen_expert")
    parser.add_argument("--sample-limit", type=int, default=50)
    args = parser.parse_args()

    before = tensor_map(args.before)
    after = tensor_map(args.after)
    common = sorted(before.keys() & after.keys())
    if not common:
        raise RuntimeError("the two checkpoints have no common tensor names")

    changed = 0
    unchanged = 0
    changed_required = 0
    compared_values = 0
    max_abs_diff = 0.0
    sum_abs_diff = 0.0
    samples: list[dict[str, object]] = []

    for name in common:
        left = read_tensor(before[name], name)
        right = read_tensor(after[name], name)
        if left.shape != right.shape:
            raise RuntimeError(f"shape changed for {name}: {left.shape} != {right.shape}")
        if not (left.is_floating_point() and right.is_floating_point()):
            continue
        difference = (left.float() - right.float()).abs()
        tensor_max = float(difference.max()) if difference.numel() else 0.0
        tensor_mean = float(difference.mean()) if difference.numel() else 0.0
        compared_values += difference.numel()
        max_abs_diff = max(max_abs_diff, tensor_max)
        sum_abs_diff += tensor_mean * difference.numel()
        if tensor_max > 0:
            changed += 1
            if args.require_pattern in name:
                changed_required += 1
            if len(samples) < args.sample_limit:
                samples.append(
                    {"name": name, "max_abs_diff": tensor_max, "mean_abs_diff": tensor_mean}
                )
        else:
            unchanged += 1

    report = {
        "before": str(args.before),
        "after": str(args.after),
        "common_tensor_names": len(common),
        "changed_floating_tensors": changed,
        "unchanged_floating_tensors": unchanged,
        "required_pattern": args.require_pattern,
        "changed_required_pattern_tensors": changed_required,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff_all_values": sum_abs_diff / compared_values if compared_values else 0.0,
        "changed_samples": samples,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))

    if changed == 0:
        raise RuntimeError("no floating-point model tensor changed")
    if args.require_pattern and changed_required == 0:
        raise RuntimeError(
            f"no changed tensor matched required pattern {args.require_pattern!r}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
