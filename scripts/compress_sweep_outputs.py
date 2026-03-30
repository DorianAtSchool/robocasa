#!/usr/bin/env python3
"""Compress an existing sweep output directory in place.

This converts non-map PNG renders to JPEG when the JPEG is smaller, rewrites
runtime JSON references to the chosen files, and removes the redundant
``source_trajectory`` copy from ``adapted_trajectory.json``.

Usage:
    python scripts/compress_sweep_outputs.py \
        --sweep-dir tmp/sweep_all_tasks_all_trajectories_L11_L42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


def is_map_png(path: str | Path) -> bool:
    path = Path(path)
    if path.suffix.lower() != ".png":
        return False
    name = path.name.lower()
    stem = path.stem.lower()
    return name == "initial_map.png" or "_map_" in name or stem.endswith("_map")


def format_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    unit = units[0]
    for unit in units:
        if value < 1024 or unit == units[-1]:
            break
        value /= 1024
    if unit == "B":
        return f"{int(value)} {unit}"
    return f"{value:.1f} {unit}"


def sum_file_sizes(paths: list[Path]) -> int:
    return sum(path.stat().st_size for path in paths if path.exists())


def discover_camera_pngs(sweep_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in sweep_dir.rglob("*.png")
        if path.is_file() and not is_map_png(path)
    )


def rewrite_runtime_image_path(path_str: str) -> str:
    path = Path(path_str)
    if path.suffix.lower() != ".png" or is_map_png(path):
        return path_str
    jpg_path = path.with_suffix(".jpg")
    if path.exists():
        return path_str
    if jpg_path.exists():
        return str(jpg_path)
    return path_str


def rewrite_path_fields(payload: dict) -> bool:
    changed = False
    if not isinstance(payload, dict):
        return changed

    image_path = payload.get("image_path")
    if isinstance(image_path, str):
        new_path = rewrite_runtime_image_path(image_path)
        if new_path != image_path:
            payload["image_path"] = new_path
            changed = True

    image_paths = payload.get("image_paths")
    if isinstance(image_paths, list):
        new_paths = [
            rewrite_runtime_image_path(path_str) if isinstance(path_str, str) else path_str
            for path_str in image_paths
        ]
        if new_paths != image_paths:
            payload["image_paths"] = new_paths
            changed = True

    return changed


def rewrite_tool_calls(tool_calls: list[dict]) -> bool:
    changed = False
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        args = tool_call.get("args")
        if isinstance(args, dict):
            changed |= rewrite_path_fields(args)
    return changed


def rewrite_step_list(steps: list[dict]) -> bool:
    changed = False
    for step in steps:
        if not isinstance(step, dict):
            continue
        changed |= rewrite_path_fields(step)
        args = step.get("args")
        if isinstance(args, dict):
            changed |= rewrite_path_fields(args)
        details = step.get("details")
        if isinstance(details, dict):
            changed |= rewrite_path_fields(details)
    return changed


def rewrite_json_files(sweep_dir: Path) -> tuple[int, int]:
    files_changed = 0
    bytes_saved = 0

    for path in sorted(sweep_dir.rglob("adapted_trajectory.json")):
        original_size = path.stat().st_size
        with open(path) as f:
            payload = json.load(f)
        changed = False
        if isinstance(payload, dict):
            if payload.pop("source_trajectory", None) is not None:
                changed = True
            tool_calls = payload.get("tool_calls")
            if isinstance(tool_calls, list):
                changed |= rewrite_tool_calls(tool_calls)
        if changed:
            with open(path, "w") as f:
                json.dump(payload, f, indent=2)
            files_changed += 1
            bytes_saved += original_size - path.stat().st_size

    for file_name in ("plan.json",):
        for path in sorted(sweep_dir.rglob(file_name)):
            original_size = path.stat().st_size
            with open(path) as f:
                payload = json.load(f)
            changed = False
            if isinstance(payload, list):
                changed = rewrite_tool_calls(payload)
            if changed:
                with open(path, "w") as f:
                    json.dump(payload, f, indent=2)
                files_changed += 1
                bytes_saved += original_size - path.stat().st_size

    for file_name in ("metadata.json", "trajectory_execution_metadata.json"):
        for path in sorted(sweep_dir.rglob(file_name)):
            original_size = path.stat().st_size
            with open(path) as f:
                payload = json.load(f)
            changed = False
            if isinstance(payload, dict):
                steps = payload.get("steps")
                if isinstance(steps, list):
                    changed = rewrite_step_list(steps)
            if changed:
                with open(path, "w") as f:
                    json.dump(payload, f, indent=2)
                files_changed += 1
                bytes_saved += original_size - path.stat().st_size

    return files_changed, bytes_saved


def convert_camera_pngs(camera_pngs: list[Path], quality: int) -> tuple[int, int, int]:
    converted = 0
    bytes_before = 0
    bytes_after = 0

    for png_path in camera_pngs:
        if not png_path.exists():
            continue

        jpg_path = png_path.with_suffix(".jpg")
        original_size = png_path.stat().st_size
        bytes_before += original_size

        try:
            with Image.open(png_path) as image:
                image.convert("RGB").save(
                    jpg_path,
                    format="JPEG",
                    quality=quality,
                    optimize=True,
                )
        except Exception:
            if jpg_path.exists():
                jpg_path.unlink()
            bytes_after += original_size
            continue

        compressed_size = jpg_path.stat().st_size
        if compressed_size < original_size:
            png_path.unlink()
            converted += 1
            bytes_after += compressed_size
        else:
            jpg_path.unlink()
            bytes_after += original_size

    return converted, bytes_before, bytes_after


def main():
    parser = argparse.ArgumentParser(description="Compress existing sweep outputs in place")
    parser.add_argument("--sweep-dir", type=str, required=True, help="Path to sweep output directory")
    parser.add_argument("--quality", type=int, default=85, help="JPEG quality for camera renders")
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    if not sweep_dir.is_dir():
        raise FileNotFoundError(f"No such directory: {sweep_dir}")

    camera_pngs = discover_camera_pngs(sweep_dir)
    map_pngs = sorted(path for path in sweep_dir.rglob("*.png") if path.is_file() and is_map_png(path))
    total_before = sum_file_sizes([path for path in sweep_dir.rglob("*") if path.is_file()])
    camera_before = sum_file_sizes(camera_pngs)

    print(f"Sweep dir: {sweep_dir}")
    print(f"Camera PNGs: {len(camera_pngs)} ({format_bytes(camera_before)})")
    print(f"Map PNGs kept: {len(map_pngs)} ({format_bytes(sum_file_sizes(map_pngs))})")

    converted, camera_bytes_before, camera_bytes_after = convert_camera_pngs(
        camera_pngs,
        quality=args.quality,
    )
    files_changed, json_bytes_saved = rewrite_json_files(sweep_dir)

    total_after = sum_file_sizes([path for path in sweep_dir.rglob("*") if path.is_file()])

    print()
    print(f"Converted camera PNGs: {converted}/{len(camera_pngs)}")
    print(
        "Camera image bytes:"
        f" {format_bytes(camera_bytes_before)} -> {format_bytes(camera_bytes_after)}"
        f" ({format_bytes(camera_bytes_before - camera_bytes_after)} saved)"
    )
    print(f"JSON files updated: {files_changed} ({format_bytes(json_bytes_saved)} saved)")
    print(
        f"Total sweep size: {format_bytes(total_before)} -> {format_bytes(total_after)}"
        f" ({format_bytes(total_before - total_after)} saved)"
    )


if __name__ == "__main__":
    main()
