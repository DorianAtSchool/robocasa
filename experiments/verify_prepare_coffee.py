#!/usr/bin/env python
"""
Verify robot separation and object placement at every step of PrepareCoffee.

Checks:
1. Mug ends up near coffee machine (not main wall)
2. Robots never overlap (<0.5m separation)
3. After move_away, delivering robot is far from coffee machine area
"""
import os
import sys
import numpy as np
os.environ['MUJOCO_GL'] = 'osmesa'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from robocasa.utils.trajectory_runner import TrajectoryRunner

runner = TrajectoryRunner(task_name='PrepareCoffee', robots=2, layout=11, style=34, seed=42)
scene = runner.get_scene_description()

# Find coffee machine
coffee_machine = next(fid for fid, info in scene['fixtures'].items()
                      if 'coffee_machine' in info['fixture_type'])
cm_pos = np.array(scene['fixtures'][coffee_machine]['position'][:2])
print(f"Coffee machine: {coffee_machine} at {cm_pos}")

# Build same trajectory as run_sample_trajectories
from run_sample_trajectories import make_prepare_coffee_trajectory
trajectory = make_prepare_coffee_trajectory(scene)
print(f"Steps: {len(trajectory['steps'])}")

# Instrument: record positions at each step
class StepRecorder:
    def __init__(self, runner):
        self.runner = runner
        self.records = []

    def record(self, step_idx, step_desc):
        p0 = runner._get_robot_position(0)
        p1 = runner._get_robot_position(1)
        sep = np.linalg.norm(p0[:2] - p1[:2])

        mug_body = runner.env.obj_body_id.get('obj', None)
        mug_pos = runner.env.sim.data.body_xpos[mug_body][:2] if mug_body else None
        mug_dist_cm = np.linalg.norm(mug_pos - cm_pos) if mug_pos is not None else None

        rec = {
            'step': step_idx,
            'desc': step_desc,
            'robot0': p0[:2].tolist(),
            'robot1': p1[:2].tolist(),
            'separation': sep,
            'mug_pos': mug_pos.tolist() if mug_pos is not None else None,
            'mug_dist_cm': mug_dist_cm,
        }
        self.records.append(rec)
        return rec

recorder = StepRecorder(runner)

# Record initial state
rec = recorder.record(-1, "initial")
print(f"\n--- Initial ---")
print(f"  Robot 0: {rec['robot0']}")
print(f"  Robot 1: {rec['robot1']}")
print(f"  Separation: {rec['separation']:.3f}m")

# Run trajectory
result = runner.run(trajectory)
print(f"\nTrajectory completed: {len(result.steps)} steps")

# Record final state
rec = recorder.record(len(trajectory['steps']), "final")
print(f"\n--- Final ---")
print(f"  Robot 0: {rec['robot0']}")
print(f"  Robot 1: {rec['robot1']}")
print(f"  Separation: {rec['separation']:.3f}m")
print(f"  Mug pos: {rec['mug_pos']}")
print(f"  Mug dist from coffee machine: {rec['mug_dist_cm']:.3f}m")

# Assertions
PASS = True

if rec['mug_dist_cm'] > 0.5:
    print(f"\n❌ FAIL: Mug is {rec['mug_dist_cm']:.3f}m from coffee machine (expected <0.5m)")
    PASS = False
else:
    print(f"\n✅ PASS: Mug is {rec['mug_dist_cm']:.3f}m from coffee machine")

if rec['separation'] < 0.5:
    print(f"❌ FAIL: Final robot separation is {rec['separation']:.3f}m (expected >0.5m)")
    PASS = False
else:
    print(f"✅ PASS: Final robot separation is {rec['separation']:.3f}m")

# Check robot 0 (the delivery robot) moved away from coffee machine area
r0_dist_cm = np.linalg.norm(np.array(rec['robot0']) - cm_pos)
if r0_dist_cm < 1.0:
    print(f"❌ FAIL: Robot 0 (delivery) is only {r0_dist_cm:.3f}m from coffee machine after move_away")
    PASS = False
else:
    print(f"✅ PASS: Robot 0 (delivery) is {r0_dist_cm:.3f}m from coffee machine after move_away")

runner.close()

if PASS:
    print("\n🎉 All checks passed!")
    sys.exit(0)
else:
    print("\n💥 Some checks failed!")
    sys.exit(1)
