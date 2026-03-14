#!/usr/bin/env python3
"""Diagnostic: verify TrajectoryRunner cameras work for both robots + room_view."""
import os
os.environ["MUJOCO_GL"] = "osmesa"

import numpy as np
import imageio
from robocasa.utils.trajectory_runner import TrajectoryRunner

runner = TrajectoryRunner(
    task_name="Kitchen",
    robots=2,
    layout=11,
    style=34,
    seed=42,
)

print("=== All cameras in sim ===")
for i in range(runner.env.sim.model.ncam):
    name = runner.env.sim.model.camera_id2name(i)
    print(f"  {i}: {name}")

print(f"\n=== TrajectoryRunner camera list ===")
for cam in runner.camera_names:
    print(f"  {cam}")

# Render all cameras
print("\n=== Render test ===")
os.makedirs("/tmp/cam_test2", exist_ok=True)
images = runner.render()
for name, frame in images.items():
    std = frame.std()
    path = f"/tmp/cam_test2/{name}.png"
    imageio.imwrite(path, frame)
    status = "OK" if std > 5 else "GRAY/BAD"
    print(f"  {name}: {frame.shape}, std={std:.1f} [{status}]")

runner.close()
