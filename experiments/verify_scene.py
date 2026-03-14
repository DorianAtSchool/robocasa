#!/usr/bin/env python
"""Quick script to verify object & fixture placement in PrepareCoffee."""
import os
import numpy as np
os.environ['MUJOCO_GL'] = 'osmesa'

from robocasa.utils.trajectory_runner import TrajectoryRunner

runner = TrajectoryRunner(task_name='PrepareCoffee', robots=2, layout=11, style=34, seed=42)
scene = runner.get_scene_description()

print("=== Objects ===")
for oid, info in scene['objects'].items():
    print(f"  {oid}: {info}")

print("\n=== Fixtures ===")
for fid, info in scene['fixtures'].items():
    ftype = info['fixture_type']
    pos = info['position']
    print(f"  {fid}: type={ftype} pos=[{pos[0]:.2f}, {pos[1]:.2f}]")

# Find coffee machine
coffee_machine = next(fid for fid, info in scene['fixtures'].items() if 'coffee_machine' in info['fixture_type'])
cm_pos = np.array(scene['fixtures'][coffee_machine]['position'][:2])

# Find nearest counter
counters = [(fid, info) for fid, info in scene['fixtures'].items() if 'counter' in info['fixture_type']]
counters.sort(key=lambda x: np.linalg.norm(np.array(x[1]['position'][:2]) - cm_pos))
print(f"\n=== Counter distances from {coffee_machine} ===")
for fid, info in counters:
    d = np.linalg.norm(np.array(info['position'][:2]) - cm_pos)
    print(f"  {fid}: {d:.3f}m")

print(f"\n=== Robot positions ===")
for i in range(2):
    p = runner._get_robot_position(i)
    print(f"  Robot {i}: [{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}]")

runner.close()
print("\nDONE")
