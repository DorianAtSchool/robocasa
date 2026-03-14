#!/usr/bin/env python3
"""Test: can we get a non-gray room_view using the same approach as two_robot_video_sample?"""
import os
os.environ["MUJOCO_GL"] = "osmesa"

import numpy as np
import imageio
from robosuite.controllers import load_composite_controller_config
from robosuite.environments.base import make
import robocasa  # noqa
import robocasa.utils.camera_utils as CamUtils
from robocasa.wrappers.enclosing_wall_render_wrapper import EnclosingWallRenderWrapper

env = make(
    env_name="Kitchen",
    robots=["PandaOmron", "PandaOmron"],
    controller_configs=load_composite_controller_config(robot="PandaOmron"),
    layout_ids=[11], style_ids=[34], seed=42,
    has_renderer=False, has_offscreen_renderer=True,
    use_camera_obs=False, ignore_done=True,
)

os.makedirs("/tmp/room_test", exist_ok=True)

# --- Test 1: No wrapper, just reset + forward (our current approach) ---
env.reset()
env.sim.forward()
render_ctx = env.sim._render_context_offscreen
cfg = CamUtils.LAYOUT_CAMS.get(env.layout_id, CamUtils.DEFAULT_LAYOUT_CAM)
render_ctx.cam.lookat[:] = cfg["lookat"]
render_ctx.cam.distance = cfg["distance"]
render_ctx.cam.azimuth = cfg["azimuth"]
render_ctx.cam.elevation = cfg["elevation"]
render_ctx.render(width=512, height=512, camera_id=-1)
frame = render_ctx.read_pixels(512, 512)[::-1]
imageio.imwrite("/tmp/room_test/1_reset_forward.png", frame)
print(f"Test 1 (reset+forward): mean={frame.mean():.1f}, std={frame.std():.1f}")

# --- Test 2: reset + zero step (what we added) ---
env.reset()
low, _ = env.action_spec
env.step(np.zeros_like(low))
render_ctx = env.sim._render_context_offscreen
render_ctx.cam.lookat[:] = cfg["lookat"]
render_ctx.cam.distance = cfg["distance"]
render_ctx.cam.azimuth = cfg["azimuth"]
render_ctx.cam.elevation = cfg["elevation"]
render_ctx.render(width=512, height=512, camera_id=-1)
frame = render_ctx.read_pixels(512, 512)[::-1]
imageio.imwrite("/tmp/room_test/2_reset_step.png", frame)
print(f"Test 2 (reset+step): mean={frame.mean():.1f}, std={frame.std():.1f}")

# --- Test 3: With EnclosingWallRenderWrapper (like two_robot_video_sample) ---
env2 = make(
    env_name="Kitchen",
    robots=["PandaOmron", "PandaOmron"],
    controller_configs=load_composite_controller_config(robot="PandaOmron"),
    layout_ids=[11], style_ids=[34], seed=42,
    has_renderer=False, has_offscreen_renderer=True,
    use_camera_obs=False, ignore_done=True,
)
env2 = EnclosingWallRenderWrapper(env2, alpha=0.1, enabled=True)
env2.reset()
low2, _ = env2.action_spec
env2.step(np.zeros_like(low2))
render_ctx2 = env2.sim._render_context_offscreen
render_ctx2.cam.lookat[:] = cfg["lookat"]
render_ctx2.cam.distance = cfg["distance"]
render_ctx2.cam.azimuth = cfg["azimuth"]
render_ctx2.cam.elevation = cfg["elevation"]
render_ctx2.render(width=512, height=512, camera_id=-1)
frame2 = render_ctx2.read_pixels(512, 512)[::-1]
imageio.imwrite("/tmp/room_test/3_wrapper_step.png", frame2)
print(f"Test 3 (wrapper+step): mean={frame2.mean():.1f}, std={frame2.std():.1f}")

# --- Test 4: Multiple steps (like the video sample which renders after every step) ---
for _ in range(10):
    env2.step(np.zeros_like(low2))
render_ctx2.cam.lookat[:] = cfg["lookat"]
render_ctx2.cam.distance = cfg["distance"]
render_ctx2.cam.azimuth = cfg["azimuth"]
render_ctx2.cam.elevation = cfg["elevation"]
render_ctx2.render(width=512, height=512, camera_id=-1)
frame3 = render_ctx2.read_pixels(512, 512)[::-1]
imageio.imwrite("/tmp/room_test/4_wrapper_10steps.png", frame3)
print(f"Test 4 (wrapper+10steps): mean={frame3.mean():.1f}, std={frame3.std():.1f}")

env.close()
env2.close()
