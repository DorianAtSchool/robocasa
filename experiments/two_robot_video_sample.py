#!/usr/bin/env python3
"""
Headless sample: run a dual-PandaOmron Kitchen env and save 6 rollout videos.

Usage:
  python experiments/two_robot_video_sample.py
  python experiments/two_robot_video_sample.py --output-dir /tmp/videos
"""

import argparse
import os

import imageio
import numpy as np
from robosuite.controllers import load_composite_controller_config
from robosuite.environments.base import make

import robocasa  # noqa: F401  # registers RoboCasa envs
import robocasa.utils.camera_utils as CamUtils
from robocasa.wrappers.enclosing_wall_render_wrapper import EnclosingWallRenderWrapper


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="videos")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--layout", type=int, default=11)
    parser.add_argument("--style", type=int, default=34)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for env + random actions",
    )
    parser.add_argument(
        "--show-walls",
        action="store_true",
        help="Keep enclosing walls opaque (default: make enclosing walls translucent)",
    )
    parser.add_argument(
        "--gl",
        type=str,
        default="osmesa",
        choices=["osmesa", "egl", "glfw"],
        help="MuJoCo OpenGL backend for offscreen rendering",
    )
    return parser.parse_args()


def compute_top_cam_config(env, room_cam_config):
    """
    Derive an overhead camera that keeps the full kitchen footprint in frame.
    """
    bbox_xy_points = []
    for fixture in env.fixtures.values():
        if not hasattr(fixture, "get_bbox_points"):
            continue
        try:
            bbox_points = np.asarray(fixture.get_bbox_points(), dtype=float)
        except Exception:
            continue
        if bbox_points.ndim != 2 or bbox_points.shape[1] < 2:
            continue
        bbox_xy_points.append(bbox_points[:, :2])

    if not bbox_xy_points:
        return dict(
            lookat=list(room_cam_config["lookat"]),
            distance=max(room_cam_config["distance"] * 1.8, 8.0),
            azimuth=room_cam_config["azimuth"],
            elevation=-89.0,
        )

    xy_points = np.concatenate(bbox_xy_points, axis=0)
    min_xy = np.min(xy_points, axis=0)
    max_xy = np.max(xy_points, axis=0)
    center_xy = 0.5 * (min_xy + max_xy)
    max_radius = np.max(np.linalg.norm(xy_points - center_xy, axis=1))

    # MuJoCo free camera uses global fovy for perspective projection.
    fovy_deg = float(getattr(env.sim.model.vis.global_, "fovy", 45.0))
    half_fovy_rad = np.deg2rad(np.clip(fovy_deg, 1.0, 89.0) / 2.0)
    required_distance = (max_radius / np.tan(half_fovy_rad)) * 1.2

    lookat = np.asarray(room_cam_config["lookat"], dtype=float).copy()
    lookat[:2] = center_xy
    return dict(
        lookat=lookat.tolist(),
        distance=float(max(8.0, required_distance)),
        azimuth=float(room_cam_config["azimuth"]),
        elevation=-89.0,
    )


def main():
    args = parse_args()

    os.environ.setdefault("MUJOCO_GL", args.gl)
    os.makedirs(args.output_dir, exist_ok=True)

    fixed_camera_map = {
        "robot0_view": "robot0_robotview",
        "robot1_view": "robot1_robotview",
        "robot0_wrist": "robot0_eye_in_hand",
        "robot1_wrist": "robot1_eye_in_hand",
    }

    env = make(
        env_name="Kitchen",
        robots=["PandaOmron", "PandaOmron"],
        controller_configs=load_composite_controller_config(robot="PandaOmron"),
        layout_ids=[args.layout],
        style_ids=[args.style],
        seed=args.seed,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=False,
        ignore_done=True,
    )
    env = EnclosingWallRenderWrapper(env, alpha=0.1, enabled=not args.show_walls)

    try:
        env.reset()
        rng = np.random.default_rng(args.seed)
        low, high = env.action_spec
        available_cameras = {
            env.sim.model.camera_id2name(i) for i in range(env.sim.model.ncam)
        }
        missing_cameras = [
            cam_name
            for cam_name in fixed_camera_map.values()
            if cam_name not in available_cameras
        ]
        if missing_cameras:
            raise RuntimeError(
                f"Missing expected cameras: {missing_cameras}. "
                f"Available cameras: {sorted(available_cameras)}"
            )

        # RoboCasa paper-style room view uses a layout-wide free camera.
        room_cam_config = CamUtils.LAYOUT_CAMS.get(
            env.layout_id, CamUtils.DEFAULT_LAYOUT_CAM
        )
        top_cam_config = compute_top_cam_config(env, room_cam_config)
        render_ctx = env.sim._render_context_offscreen
        if render_ctx is None:
            raise RuntimeError("Offscreen render context is not initialized.")

        room_labels = ("room_view", "top_view")
        all_video_labels = (*room_labels, *fixed_camera_map.keys())
        writers = {
            label: imageio.get_writer(
                os.path.join(args.output_dir, f"{label}.mp4"), fps=args.fps
            )
            for label in all_video_labels
        }

        try:
            for _ in range(args.steps):
                action = rng.uniform(low, high)

                env.step(action)

                # Render layout-wide free camera (paper-style room view).
                render_ctx.cam.lookat[:] = room_cam_config["lookat"]
                render_ctx.cam.distance = room_cam_config["distance"]
                render_ctx.cam.azimuth = room_cam_config["azimuth"]
                render_ctx.cam.elevation = room_cam_config["elevation"]
                render_ctx.render(
                    width=args.width,
                    height=args.height,
                    camera_id=-1,
                )
                room_frame = render_ctx.read_pixels(args.width, args.height)
                writers["room_view"].append_data(room_frame[::-1])

                # Render top-down free camera for whole-kitchen overview.
                render_ctx.cam.lookat[:] = top_cam_config["lookat"]
                render_ctx.cam.distance = top_cam_config["distance"]
                render_ctx.cam.azimuth = top_cam_config["azimuth"]
                render_ctx.cam.elevation = top_cam_config["elevation"]
                render_ctx.render(
                    width=args.width,
                    height=args.height,
                    camera_id=-1,
                )
                top_frame = render_ctx.read_pixels(args.width, args.height)
                writers["top_view"].append_data(top_frame[::-1])

                for label, cam_name in fixed_camera_map.items():
                    frame = env.sim.render(
                        height=args.height,
                        width=args.width,
                        camera_name=cam_name,
                    )[::-1]
                    writers[label].append_data(frame)
        finally:
            for writer in writers.values():
                writer.close()
    finally:
        env.close()

    print(f"Saved 6 videos to {args.output_dir}")
    for label in all_video_labels:
        print(f"  - {os.path.join(args.output_dir, f'{label}.mp4')}")


if __name__ == "__main__":
    main()
