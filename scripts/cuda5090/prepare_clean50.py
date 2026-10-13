#!/usr/bin/env python3
"""Resolve, verify, and optionally stage the official LingBot RoboTwin clean-50 archives."""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path


def clean_entries(training_list: Path) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for raw_line in training_list.read_text().splitlines():
        line = raw_line.strip()
        if not line or "_clean_50-50" not in line:
            continue
        _, dataset_path = line.split(maxsplit=1)
        name = Path(dataset_path).name.removesuffix("-50")
        entries.append((line, name))
    if len(entries) != 50:
        raise RuntimeError(f"expected 50 clean entries, found {len(entries)}")
    return entries


def resolve_archive(archive_root: Path, dataset_name: str) -> tuple[str, Path]:
    task_dirs = sorted(path for path in archive_root.iterdir() if path.is_dir())
    matches: list[tuple[str, Path]] = []
    for task_dir in task_dirs:
        prefix = f"{task_dir.name}-"
        if dataset_name.startswith(prefix):
            archive = task_dir / f"{dataset_name[len(prefix):]}.zip"
            if archive.is_file():
                matches.append((task_dir.name, archive))
    if len(matches) != 1:
        raise RuntimeError(f"could not uniquely resolve {dataset_name}: {matches}")
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--training-list", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--stage-dir", type=Path)
    parser.add_argument("--copy", action="store_true", help="copy instead of hard-linking staged files")
    parser.add_argument("--zip-test", action="store_true", help="read and CRC-check every archive member")
    args = parser.parse_args()

    rows: list[tuple[str, str, Path, int]] = []
    failures: list[str] = []
    for _, dataset_name in clean_entries(args.training_list):
        try:
            task, archive = resolve_archive(args.archive_root, dataset_name)
            size = archive.stat().st_size
            if size == 0:
                raise RuntimeError("archive is empty")
            with zipfile.ZipFile(archive) as zf:
                if zf.testzip() if args.zip_test else not zf.namelist():
                    raise RuntimeError("ZIP integrity check failed")
            rows.append((task, dataset_name, archive.resolve(), size))
        except Exception as exc:  # report all broken entries together
            failures.append(f"{dataset_name}: {exc}")

    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest_lines = ["task\tdataset\tarchive\tbytes"]
    for task, dataset_name, archive, size in rows:
        manifest_lines.append(f"{task}\t{dataset_name}\t{archive}\t{size}")
    args.output_manifest.write_text("\n".join(manifest_lines) + "\n")

    if args.stage_dir:
        for task, _, archive, _ in rows:
            destination = args.stage_dir / task / archive.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if destination.stat().st_size != archive.stat().st_size:
                    raise RuntimeError(f"staged file has wrong size: {destination}")
                continue
            if args.copy:
                shutil.copy2(archive, destination)
            else:
                try:
                    destination.hardlink_to(archive)
                except OSError:
                    shutil.copy2(archive, destination)

    total_gib = sum(row[3] for row in rows) / 1024**3
    print(f"verified {len(rows)} clean archives ({total_gib:.2f} GiB)")
    print(f"manifest: {args.output_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
