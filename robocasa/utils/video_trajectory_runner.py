"""
Video trajectory runner — replays recorded demo arm motion and renders videos.

    For each ``interact`` or ``move_object`` step the arm+gripper joint values
    are copied directly from a matching recorded demonstration.  ``navigate``
    steps smoothly interpolate the robot base to the target fixture.
    ``move_away`` shifts the robot slightly away from its nearest fixture.

Usage::

    runner = VideoTrajectoryRunner(
        task_name="Kitchen", robots=2, layout=11, style=34, seed=42,
        output_dir="trajectory_videos/PrepareCoffee_layout11_seed42",
    )
    scene = runner.get_scene_description()
    trajectory = build_trajectory(scene)
    runner.run(trajectory)
    runner.close()
"""

from __future__ import annotations

import json
from pathlib import Path

import imageio
import numpy as np

import robocasa.utils.camera_utils as CamUtils
import robocasa.utils.env_utils as EnvUtils
import robocasa.utils.lerobot_utils as LU
from robocasa.utils.dataset_registry_utils import get_ds_path
from robocasa.utils.robot_color_utils import recolor_robot
from robocasa.utils.trajectory_runner import TrajectoryRunner


# ---------------------------------------------------------------------------
# Atomic task mapping
# ---------------------------------------------------------------------------

DEMO_TASK_MAP = {
    # (action, keyword) -> atomic task name in dataset registry
    ("move_object", "cabinet"): "PickPlaceCabinetToCounter",
    ("move_object", "sink"): "PickPlaceCounterToSink",
    ("move_object", "coffee"): "CoffeeSetupMug",
    ("move_object", "microwave"): "PickPlaceCounterToMicrowave",
    ("move_object", "counter"): "PickPlaceCabinetToCounter",
    ("interact", "open", "cabinet"): "OpenCabinet",
    ("interact", "close", "cabinet"): "CloseCabinet",
    ("interact", "open", "fridge"): "OpenFridge",
    ("interact", "close", "fridge"): "CloseFridge",
    ("interact", "turn_on", "coffee"): "StartCoffeeMachine",
    ("interact", "turn_on", "microwave"): "TurnOnMicrowave",
}

# Fallback demo for move_object when no specific match is found
_MOVE_OBJECT_FALLBACK = "PickPlaceCabinetToCounter"
_LOCAL_DATASET_FALLBACKS = {
    "OpenCabinet": ("PrepareCoffee",),
    "CloseCabinet": ("PrepareCoffee",),
    "CoffeeSetupMug": ("PrepareCoffee",),
    "StartCoffeeMachine": ("PrepareCoffee",),
    "PickPlaceCounterToMicrowave": ("MicrowaveThawing",),
    "TurnOnMicrowave": ("MicrowaveThawing",),
    "OpenFridge": ("HotDogSetup",),
    "CloseFridge": ("HotDogSetup",),
}

# Number of static frames to render for non-motion steps.
_STATIC_FRAMES = 10
# Number of settle frames after a navigate teleport.
_NAVIGATE_FRAMES = 20

# Approximate split points for composite demonstrations so individual trajectory
# steps do not replay the entire task.
_DEMO_CLIP_SPECS = {
    ("OpenCabinet", "interact", "open"): (0.00, 1.00, 60),
    ("CloseCabinet", "interact", "close"): (0.00, 1.00, 60),
    ("CoffeeSetupMug", "move_object", None): (0.00, 1.00, 120),
    ("StartCoffeeMachine", "interact", "turn_on"): (0.00, 1.00, 60),
    ("TurnOnMicrowave", "interact", "turn_on"): (0.00, 1.00, 60),
    ("PrepareCoffee", "interact", "open"): (0.00, 0.18, 90),
    ("PrepareCoffee", "move_object", None): (0.16, 0.82, 180),
    ("PrepareCoffee", "interact", "turn_on"): (0.80, 1.00, 90),
    ("MicrowaveThawing", "move_object", None): (0.08, 0.80, 120),
    ("MicrowaveThawing", "interact", "turn_on"): (0.78, 1.00, 60),
    # HotDogSetup: fridge open is at ~frames 1431-1600 of 2373 (≈0.60-0.67)
    ("HotDogSetup", "interact", "open"): (0.58, 0.68, 90),
    ("HotDogSetup", "interact", "close"): (0.58, 0.68, 60),
    # HotDogSetup: move_object covers pick+place arm motion
    ("HotDogSetup", "move_object", None): (0.00, 1.00, 120),
}
_DEFAULT_CLIP_FRAMES = {
    "move_object": 72,
    "interact": 36,
}
# How far (metres) to shift a robot when "moving away" from its current fixture.
_MOVE_AWAY_OFFSET = 0.8


def _lookup_demo_task(action: str, args: dict) -> str:
    """Find the atomic task name that matches a trajectory step.

    Raises ``ValueError`` if no matching demo exists (no silent fallback).
    """
    fixture_id = (args.get("to") or args.get("fixture") or "").lower()
    interact_action = (args.get("action") or "").lower()

    # Try most-specific key first: (action, interact_action, fixture_keyword)
    for keyword in _fixture_keywords(fixture_id):
        key3 = (action, interact_action, keyword)
        if key3 in DEMO_TASK_MAP:
            return DEMO_TASK_MAP[key3]

    # Try (action, interact_action)
    key2_act = (action, interact_action)
    if key2_act in DEMO_TASK_MAP:
        return DEMO_TASK_MAP[key2_act]

    # Try (action, fixture_keyword)
    for keyword in _fixture_keywords(fixture_id):
        key2 = (action, keyword)
        if key2 in DEMO_TASK_MAP:
            return DEMO_TASK_MAP[key2]

    # Fallback for move_object: use a generic pick-place demo
    if action == "move_object":
        return _MOVE_OBJECT_FALLBACK

    raise ValueError(
        f"No matching demo for action={action!r} args={args!r}. "
        f"Available mappings: {list(DEMO_TASK_MAP.keys())}"
    )


def _resolve_demo_task(step: dict) -> str:
    """Resolve the atomic task for a step, honoring explicit overrides."""
    override = step.get("atomic_task") or step.get("demo_task")
    if override:
        return override
    return _lookup_demo_task(step.get("action", ""), step.get("args", {}))


def _fixture_keywords(fixture_id: str) -> list[str]:
    """Extract search keywords from a fixture ID like 'cab_SwingLeftDoor_42'."""
    fid = fixture_id.lower()
    keywords = []
    if "cabinet" in fid or "cab_" in fid or fid.startswith("cab"):
        keywords.append("cabinet")
    if "coffee" in fid:
        keywords.append("coffee")
    if "micro" in fid:
        keywords.append("microwave")
    if "sink" in fid:
        keywords.append("sink")
    if "stove" in fid:
        keywords.append("stove")
    if "counter" in fid:
        keywords.append("counter")
    if "fridge" in fid or "freezer" in fid:
        keywords.append("fridge")
    return keywords


def _find_local_demo_dataset_path(task_name: str) -> Path | None:
    """Resolve a locally available dataset path for a task."""
    for split in ("pretrain", "target"):
        try:
            path = get_ds_path(task_name, source="human", split=split)
        except (ValueError, Exception):
            continue
        if path is not None:
            path = Path(path)
            if path.exists():
                return path
    return None


def _get_demo_dataset_path(task_name: str) -> tuple[Path, str]:
    """Resolve the dataset path for a task, with local fallbacks when needed."""
    local_path = _find_local_demo_dataset_path(task_name)
    if local_path is not None:
        return local_path, task_name

    for fallback_task in _LOCAL_DATASET_FALLBACKS.get(task_name, ()):
        local_path = _find_local_demo_dataset_path(fallback_task)
        if local_path is not None:
            return local_path, fallback_task

    # Try to download
    try:
        from robocasa.scripts.download_datasets import download_datasets
        download_datasets(
            tasks=[task_name], split=["pretrain", "target"], source=["human"]
        )
    except Exception:
        pass

    local_path = _find_local_demo_dataset_path(task_name)
    if local_path is not None:
        return local_path, task_name

    for fallback_task in _LOCAL_DATASET_FALLBACKS.get(task_name, ()):
        local_path = _find_local_demo_dataset_path(fallback_task)
        if local_path is not None:
            return local_path, fallback_task

    raise FileNotFoundError(
        f"Could not find or download demo dataset for task {task_name!r}"
    )


def compute_top_cam_config(env, room_cam_config):
    """Derive an overhead camera config (copied from two_robot_video_sample.py)."""
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


# ---------------------------------------------------------------------------
# VideoTrajectoryRunner
# ---------------------------------------------------------------------------

class VideoTrajectoryRunner(TrajectoryRunner):
    """TrajectoryRunner that produces videos via demo playback.

    Inherits env setup, scene description, and robot positioning from
    :class:`TrajectoryRunner`.  Overrides :meth:`run` to render video
    frames and play back recorded demos for interact/move_object steps.
    """

    def __init__(
        self,
        task_name: str = "Kitchen",
        robots: int = 2,
        layout: int | None = None,
        style: int | None = None,
        seed: int | None = None,
        render_width: int = 1280,
        render_height: int = 720,
        fps: int = 20,
        output_dir: str = "trajectory_videos",
        gl_backend: str = "osmesa",
    ):
        super().__init__(
            task_name=task_name,
            robots=robots,
            layout=layout,
            style=style,
            seed=seed,
            render_width=render_width,
            render_height=render_height,
            gl_backend=gl_backend,
        )

        self.fps = fps
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Recolour robot1 for visual distinction
        recolor_robot(self.env.sim, robot_idx=1)

        # Camera configs for free cameras
        self._room_cam_config = CamUtils.LAYOUT_CAMS.get(
            self.env.layout_id, CamUtils.DEFAULT_LAYOUT_CAM
        )
        self._top_cam_config = compute_top_cam_config(self.env, self._room_cam_config)

        # Fixed camera mapping (label -> MuJoCo camera name)
        self._fixed_camera_map = {
            "robot0_view": "robot0_robotview",
            "robot1_view": "robot1_robotview",
            "robot0_wrist": "robot0_eye_in_hand",
            "robot1_wrist": "robot1_eye_in_hand",
        }

        # All video labels
        self._free_cam_labels = ("room_view", "top_view")
        self._all_labels = (*self._free_cam_labels, *self._fixed_camera_map.keys())

        # Video writers (created in run())
        self._writers: dict[str, imageio.core.Format.Writer] | None = None
        # Frame counter per step for metadata
        self._frame_count = 0
        self._step_metadata: list[dict] = []

        # Demo state cache: task_name -> states array
        self._demo_cache: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Video rendering
    # ------------------------------------------------------------------

    def _render_free_camera_for_env(self, env, cam_config: dict) -> np.ndarray:
        """Render a free camera for the requested environment."""
        render_ctx = env.sim._render_context_offscreen
        if render_ctx is None:
            return np.zeros((self.render_height, self.render_width, 3), dtype=np.uint8)
        render_ctx.cam.lookat[:] = cam_config["lookat"]
        render_ctx.cam.distance = cam_config["distance"]
        render_ctx.cam.azimuth = cam_config["azimuth"]
        render_ctx.cam.elevation = cam_config["elevation"]
        render_ctx.render(
            width=self.render_width,
            height=self.render_height,
            camera_id=-1,
        )
        return render_ctx.read_pixels(self.render_width, self.render_height)[::-1]

    def _render_free_camera(self, cam_config: dict) -> np.ndarray:
        """Render a free camera for the main shared environment."""
        return self._render_free_camera_for_env(self.env, cam_config)

    def _record_frames(self, frames: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Append already-rendered frames to writers and advance frame count."""
        if self._writers is not None:
            for label in self._all_labels:
                if label in frames and label in self._writers:
                    self._writers[label].append_data(frames[label])

        self._frame_count += 1
        return frames

    def _render_main_env_frames(self) -> dict[str, np.ndarray]:
        """Render all cameras from the main shared environment without recording."""
        frames = {}

        frames["room_view"] = self._render_free_camera(self._room_cam_config)
        frames["top_view"] = self._render_free_camera(self._top_cam_config)

        for label, cam_name in self._fixed_camera_map.items():
            try:
                frame = self.env.sim.render(
                    height=self.render_height,
                    width=self.render_width,
                    camera_name=cam_name,
                )[::-1]
                frames[label] = frame
            except Exception:
                frames[label] = np.zeros(
                    (self.render_height, self.render_width, 3), dtype=np.uint8
                )
        return frames

    def _render_and_record(self) -> dict[str, np.ndarray]:
        """Render all cameras, append to video writers, return frames dict."""
        return self._record_frames(self._render_main_env_frames())

    def _render_static_frames(self, n_frames: int = _STATIC_FRAMES) -> dict[str, np.ndarray]:
        """Render n static frames (no sim change). Returns last frame dict."""
        frames = {}
        for _ in range(n_frames):
            frames = self._render_and_record()
        return frames

    # ------------------------------------------------------------------
    # Demo playback
    # ------------------------------------------------------------------

    # PandaOmron joint layout in both demo and 2-robot envs:
    #   qpos[0:4]  — mobile base (forward, side, yaw, torso_height)
    #   qpos[4:11] — arm (joint1–joint7)
    #   qpos[11:13] — gripper (finger_joint1, finger_joint2)
    # In demo flattened state: state[0]=time, state[1:14]=robot qpos
    # In 2-robot env: robot0 at qpos[0:13], robot1 at qpos[13:26]
    _ROBOT_NQPOS = 13  # total qpos per PandaOmron
    _DEMO_QPOS_OFFSET = 1  # skip time in flattened state
    # We only replay arm + gripper (indices 4–12 relative to robot start),
    # not the mobile base — the base is positioned by _move_robot_near_fixture.
    _ARM_GRIPPER_SLICE = slice(4, 13)  # 7 arm + 2 gripper = 9 DOF

    def _get_demo_states(
        self, task_name: str, ep_idx: int = 0
    ) -> tuple[np.ndarray, str]:
        """Load demo states, with caching, and return the resolved task name."""
        ds_path, resolved_task = _get_demo_dataset_path(task_name)
        cache_key = f"{resolved_task}_{ep_idx}"
        if cache_key not in self._demo_cache:
            states = LU.get_episode_states(ds_path, ep_idx)
            self._demo_cache[cache_key] = states
        return self._demo_cache[cache_key], resolved_task

    def _get_demo_clip(
        self,
        task_name: str,
        action: str,
        args: dict,
        ep_idx: int = 0,
        downsample: bool = True,
    ) -> np.ndarray:
        """Select the sub-clip that best matches the current step."""
        states, resolved_task = self._get_demo_states(task_name, ep_idx)
        if len(states) == 0:
            return states

        interact_action = (args.get("action") or "").lower() if action == "interact" else None
        start_frac, end_frac, max_frames = _DEMO_CLIP_SPECS.get(
            (resolved_task, action, interact_action),
            (0.0, 1.0, _DEFAULT_CLIP_FRAMES.get(action, len(states))),
        )
        start_idx = int(start_frac * max(len(states) - 1, 1))
        end_idx = int(np.ceil(end_frac * len(states)))
        end_idx = max(start_idx + 2, min(len(states), end_idx))
        clip = states[start_idx:end_idx]
        if not downsample or len(clip) <= max_frames:
            return clip

        idx = np.linspace(0, len(clip) - 1, num=max_frames, dtype=int)
        return clip[idx]

    def _get_robot_arm_gripper_qpos(self, robot_idx: int) -> np.ndarray:
        """Return the current 7 arm + 2 gripper qpos for a robot."""
        start = robot_idx * self._ROBOT_NQPOS + 4
        end = robot_idx * self._ROBOT_NQPOS + self._ROBOT_NQPOS
        return self.env.sim.data.qpos[start:end].copy()

    def _set_robot_arm_gripper_qpos(self, robot_idx: int, qpos: np.ndarray):
        """Overwrite the current robot arm + gripper qpos."""
        start = robot_idx * self._ROBOT_NQPOS + 4
        end = robot_idx * self._ROBOT_NQPOS + self._ROBOT_NQPOS
        self.env.sim.data.qpos[start:end] = qpos
        self.env.sim.forward()

    def _get_robot_yaw(self, robot_idx: int) -> float | None:
        """Return the robot's current world-frame base yaw."""
        yaw_jnt = f"mobilebase{robot_idx}_joint_mobile_yaw"
        try:
            addr = self.env.sim.model.get_joint_qpos_addr(yaw_jnt)
        except Exception:
            return None

        local_yaw = float(self.env.sim.data.qpos[addr])
        anchor_ori = getattr(
            self.env,
            "init_robot_base_ori_anchors",
            [None] * (robot_idx + 1),
        )[robot_idx]
        if anchor_ori is None:
            return local_yaw
        return float(local_yaw + anchor_ori[2])

    def _set_robot_base_pose(
        self,
        robot_idx: int,
        pos: np.ndarray | list[float],
        yaw: float | None,
    ):
        """Set the robot's base translation and yaw."""
        EnvUtils.set_robot_to_position(self.env, np.asarray(pos, dtype=float), robot_idx=robot_idx)
        self._set_robot_yaw(robot_idx, yaw)
        self.env.sim.forward()

    def _compute_robot_fixture_pose(
        self,
        robot_idx: int,
        fixture_id: str | None,
    ) -> tuple[np.ndarray, float | None]:
        """Compute a robot base pose near a fixture without mutating the sim."""
        if fixture_id is None or fixture_id not in self._fixtures:
            return self._get_robot_position(robot_idx), self._get_robot_yaw(robot_idx)

        fxtr = self._fixtures[fixture_id]
        try:
            pos, ori = EnvUtils.compute_robot_base_placement_pose(
                self.env,
                ref_fixture=fxtr,
                robot_idx=robot_idx,
            )
            yaw = ori[2] if ori is not None else None
            return np.asarray(pos, dtype=float), yaw
        except Exception:
            fallback = np.array(fxtr.pos, dtype=float)
            angle = getattr(fxtr, "rot", 0.0) or 0.0
            fallback[0] += 0.6 * np.cos(angle + np.pi)
            fallback[1] += 0.6 * np.sin(angle + np.pi)
            return fallback, None

    def _get_object_pose(self, object_id: str) -> tuple[np.ndarray, np.ndarray]:
        """Return the object's world position and quaternion."""
        obj = self.env.objects[object_id]
        qpos = self.env.sim.data.get_joint_qpos(obj.joints[0]).copy()
        return qpos[:3], qpos[3:7]

    def _set_object_pose(
        self,
        object_id: str,
        pos: np.ndarray | list[float],
        quat: np.ndarray | list[float],
    ):
        """Place the object at the requested pose."""
        obj = self.env.objects[object_id]
        self.env.sim.data.set_joint_qpos(
            obj.joints[0],
            np.concatenate([np.asarray(pos, dtype=float), np.asarray(quat, dtype=float)]),
        )
        self.env.sim.forward()

    def _get_move_object_source_fixture(self, object_id: str | None) -> str | None:
        """Resolve the fixture the object currently belongs to."""
        if not object_id:
            return None
        if object_id in self._object_locations:
            return self._object_locations[object_id]
        if object_id not in self.env.objects:
            return None

        obj_pos, _ = self._get_object_pose(object_id)
        return self._find_object_fixture(obj_pos)

    def _play_demo_clip(
        self,
        robot_idx: int,
        task_name: str,
        action: str,
        args: dict,
        ep_idx: int = 0,
    ):
        """Replay a demo clip by directly injecting arm+gripper qpos each frame.

        The robot base should already be positioned near the relevant fixture
        before this is called.  Only the arm and gripper joints are overwritten
        — the base stays put.
        """
        clip = self._get_demo_clip(task_name, action, args, ep_idx, downsample=True)
        if len(clip) == 0:
            self._render_static_frames(_STATIC_FRAMES)
            return

        demo_arm_start = self._DEMO_QPOS_OFFSET + 4
        demo_arm_end = self._DEMO_QPOS_OFFSET + self._ROBOT_NQPOS

        for state in clip:
            arm_gripper_qpos = state[demo_arm_start:demo_arm_end]
            self._set_robot_arm_gripper_qpos(robot_idx, arm_gripper_qpos)
            self._render_and_record()

    def _play_raw_demo_clip(
        self,
        robot_idx: int,
        task_name: str,
        ep_idx: int,
        start_frame: int,
        end_frame: int,
        max_frames: int = 120,
    ):
        """Replay a raw frame range from a demo, injecting both arm qpos and
        fixture/object state from the demo.

        The demo was recorded in a 1-robot env.  Its flattened state is::

            [time(1), qpos(demo_nq), qvel(demo_nv)]

        where ``qpos = robot(13) + fixtures(demo_nq-13)``.  In the 2-robot env
        the layout is ``qpos = robot0(13) + robot1(13) + fixtures(env_nq-26)``.

        Because the env is created with the same layout/style as the demo
        episode, the fixture joint ordering is identical — we copy the demo's
        fixture qpos directly into the 2-robot env's fixture slots, and inject
        the demo's arm+gripper qpos into the acting robot.
        """
        states, _ = self._get_demo_states(task_name, ep_idx)
        end_frame = min(end_frame, len(states))
        clip = states[start_frame:end_frame]
        if len(clip) == 0:
            self._render_static_frames(_STATIC_FRAMES)
            return

        # Downsample if too long
        if len(clip) > max_frames:
            idx = np.linspace(0, len(clip) - 1, num=max_frames, dtype=int)
            clip = clip[idx]

        demo_arm_start = self._DEMO_QPOS_OFFSET + 4
        demo_arm_end = self._DEMO_QPOS_OFFSET + self._ROBOT_NQPOS

        # Fixture qpos mapping:
        # Demo state: qpos starts at offset 1, robot at [1:14], fixtures at [14:1+demo_nq]
        # Demo nq = (state_len - 1) // 2 + correction for free joints
        # Simpler: fixture qpos in demo starts at offset 1 + 13 = 14
        demo_fixture_start = self._DEMO_QPOS_OFFSET + self._ROBOT_NQPOS  # = 14

        # In 2-robot env: fixture qpos starts after both robots
        env_nq = self.env.sim.model.nq
        env_fixture_start = 2 * self._ROBOT_NQPOS  # = 26
        env_fixture_count = env_nq - env_fixture_start

        # Demo nq: state = [time(1), qpos(demo_nq), qvel(demo_nv)]
        # demo_nq + demo_nv = state_len - 1
        # For same-layout envs, fixture count should match
        demo_state_len = clip.shape[1]
        # Figure out demo_nq from state length (nq > nv due to free joints)
        # nq - nv = number_of_free_joints, and nq + nv = state_len - 1
        # We know the number of fixture joints (env_fixture_count) is the same
        demo_nq = self._ROBOT_NQPOS + env_fixture_count
        demo_fixture_count = min(env_fixture_count, demo_nq - self._ROBOT_NQPOS)

        for state in clip:
            # Inject arm + gripper qpos into the acting robot
            arm_gripper_qpos = state[demo_arm_start:demo_arm_end]
            self._set_robot_arm_gripper_qpos(robot_idx, arm_gripper_qpos)

            # Copy fixture/object qpos from demo into env
            demo_fix = state[demo_fixture_start:demo_fixture_start + demo_fixture_count]
            if len(demo_fix) == env_fixture_count:
                self.env.sim.data.qpos[env_fixture_start:env_fixture_start + env_fixture_count] = demo_fix
                self.env.sim.forward()

            self._render_and_record()

    def _play_move_object_demo(
        self,
        robot_idx: int,
        task_name: str,
        args: dict,
        ep_idx: int = 0,
    ):
        """Replay a pick-place demo clip, then teleport the object to its destination.

        The arm motion comes directly from the demo dataset.  The object is
        moved to the target fixture **after** the clip finishes (not mid-flight).
        """
        self._play_demo_clip(robot_idx, task_name, "move_object", args, ep_idx)

        # After the demo clip, place the object at the destination
        object_id = args.get("object")
        to_fixture = args.get("to")
        if object_id and to_fixture:
            try:
                self.move_object(object_id, to_fixture)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Navigate / move away
    # ------------------------------------------------------------------

    def _navigate_and_record(
        self, robot_idx: int, fixture_id: str, n_frames: int = _NAVIGATE_FRAMES
    ):
        """Smoothly interpolate the robot base to the fixture over *n_frames*.

        The base position is linearly interpolated so the robot glides into
        place rather than teleporting.
        """
        start_pos = self._get_robot_position(robot_idx)
        start_yaw = self._get_robot_yaw(robot_idx)
        target_pos, target_yaw = self._compute_robot_fixture_pose(robot_idx, fixture_id)

        for i in range(n_frames):
            alpha = (i + 1) / n_frames
            pos = (1.0 - alpha) * start_pos + alpha * target_pos
            if start_yaw is not None and target_yaw is not None:
                yaw = (1.0 - alpha) * start_yaw + alpha * target_yaw
            else:
                yaw = target_yaw if target_yaw is not None else start_yaw
            self._set_robot_base_pose(robot_idx, pos, yaw)
            self._render_and_record()

        # Snap to final position and resolve collisions with other robots
        self._move_robot_near_fixture(robot_idx, fixture_id)
        self._render_static_frames(5)

    def _move_away_and_record(
        self, robot_idx: int, n_frames: int = _NAVIGATE_FRAMES
    ):
        """Shift the robot slightly away from its current position.

        Moves the robot *_MOVE_AWAY_OFFSET* metres in the direction opposite
        to the nearest fixture, so the other robot can approach.
        """
        cur_pos = self._get_robot_position(robot_idx)

        # Find the nearest fixture to figure out which direction to move away
        best_fid, best_dist = None, float("inf")
        for fid, fxtr in self._fixtures.items():
            d = float(np.linalg.norm(np.array(fxtr.pos[:2]) - cur_pos[:2]))
            if d < best_dist:
                best_dist = d
                best_fid = fid

        if best_fid is not None:
            fxtr_pos = np.array(self._fixtures[best_fid].pos[:2])
            away_dir = cur_pos[:2] - fxtr_pos
            norm = np.linalg.norm(away_dir)
            if norm > 1e-3:
                away_dir /= norm
            else:
                away_dir = np.array([1.0, 0.0])
        else:
            away_dir = np.array([1.0, 0.0])

        target_pos = cur_pos.copy()
        target_pos[:2] += away_dir * _MOVE_AWAY_OFFSET

        start_pos = cur_pos.copy()
        for i in range(n_frames):
            alpha = (i + 1) / n_frames
            pos = (1.0 - alpha) * start_pos + alpha * target_pos
            EnvUtils.set_robot_to_position(self.env, pos, robot_idx=robot_idx)
            self.env.sim.forward()
            self._render_and_record()

    # ------------------------------------------------------------------
    # Trajectory execution
    # ------------------------------------------------------------------

    def run(self, trajectory: dict | list) -> dict:
        """Execute a trajectory with demo playback and video rendering.

        Returns a metadata dict with step timing information.
        """
        if isinstance(trajectory, dict):
            steps = trajectory.get("steps", [])
            task = trajectory.get("task", "unknown")
        else:
            steps = trajectory
            task = "unknown"

        scene = self.get_scene_description()

        # Initialize render context with a zero-action step
        low, _ = self.env.action_spec
        self.env.step(np.zeros_like(low))

        # Create video writers
        self._writers = {
            label: imageio.get_writer(
                str(self.output_dir / f"{label}.mp4"), fps=self.fps
            )
            for label in self._all_labels
        }
        self._frame_count = 0
        self._step_metadata = []

        # Create frames directory for before/after PNGs
        frames_dir = self.output_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Render initial state
            initial_frames = self._render_static_frames(5)
            self._save_key_frames(frames_dir, "initial", initial_frames)

            for i, step in enumerate(steps):
                agent_id = step.get("agent_id", "agent_0")
                action = step.get("action", "")
                args = step.get("args", {})
                robot_idx = int(agent_id.replace("agent_", ""))
                playback_mode = "static"

                start_frame = self._frame_count

                # Capture before
                before_frames = self._render_and_record()
                self._save_key_frames(
                    frames_dir, f"step_{i:03d}_before", before_frames
                )

                # Execute based on action type
                if action == "communicate" or action == "wait":
                    self._render_static_frames(_STATIC_FRAMES)

                elif action == "navigate":
                    fixture_id = args.get("fixture") or args.get("fixture_id")
                    if fixture_id:
                        self._navigate_and_record(robot_idx, fixture_id)
                        playback_mode = "navigate"
                    else:
                        self._render_static_frames(_STATIC_FRAMES)

                elif action == "move_away":
                    self._move_away_and_record(robot_idx)
                    playback_mode = "navigate"

                elif action == "move_object":
                    demo_task = _resolve_demo_task(step)
                    # Position robot near the source fixture before playing the demo
                    source_fixture = args.get("from") or self._get_move_object_source_fixture(
                        args.get("object")
                    )
                    if source_fixture:
                        self._move_robot_near_fixture(robot_idx, source_fixture)
                    self._play_move_object_demo(robot_idx, demo_task, args)
                    playback_mode = "demo_playback"

                elif action == "teleport_object":
                    # Pure teleportation: move robot near source, teleport
                    # object to destination, render static frames. No arm demo.
                    source_fixture = args.get("from") or self._get_move_object_source_fixture(
                        args.get("object")
                    )
                    if source_fixture:
                        self._move_robot_near_fixture(robot_idx, source_fixture)
                    object_id = args.get("object")
                    to_fixture = args.get("to")
                    if object_id and to_fixture:
                        try:
                            self.move_object(object_id, to_fixture)
                        except Exception:
                            pass
                    self._render_static_frames(_STATIC_FRAMES)
                    playback_mode = "teleport"

                elif action == "interact":
                    demo_task = _resolve_demo_task(step)
                    # Position robot near the fixture before playing the demo
                    fixture_id = args.get("fixture") or args.get("fixture_id")
                    if fixture_id:
                        self._move_robot_near_fixture(robot_idx, fixture_id)
                    self._play_demo_clip(robot_idx, demo_task, action, args)
                    # Apply the fixture state change (open/close/turn_on)
                    act = args.get("action")
                    if fixture_id and act:
                        try:
                            self.interact(fixture_id, act)
                        except Exception:
                            pass
                    playback_mode = "demo_playback"

                elif action == "teleport_interact":
                    # Pure teleportation: change fixture state without arm demo.
                    fixture_id = args.get("fixture") or args.get("fixture_id")
                    if fixture_id:
                        self._move_robot_near_fixture(robot_idx, fixture_id)
                    act = args.get("action")
                    if fixture_id and act:
                        try:
                            self.interact(fixture_id, act)
                        except Exception:
                            pass
                    self._render_static_frames(_STATIC_FRAMES)
                    playback_mode = "teleport"

                elif action == "demo_clip":
                    # Raw demo frame range replay (used by demo-split mode)
                    task = args.get("task", "")
                    ep = args.get("episode", 0)
                    sf = args.get("start_frame", 0)
                    ef = args.get("end_frame", 0)
                    self._play_raw_demo_clip(robot_idx, task, ep, sf, ef)
                    playback_mode = "demo_playback"

                else:
                    raise ValueError(
                        f"Step {i}: unknown action {action!r}. "
                        f"Expected: navigate, move_object, teleport_object, "
                        f"interact, teleport_interact, communicate, wait, "
                        f"move_away, demo_clip"
                    )

                # Capture after
                after_frames = self._render_and_record()
                self._save_key_frames(
                    frames_dir, f"step_{i:03d}_after", after_frames
                )

                end_frame = self._frame_count
                self._step_metadata.append({
                    "step_index": i,
                    "agent_id": agent_id,
                    "action": action,
                    "args": args,
                    "playback_mode": playback_mode,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                })

        finally:
            # Close all writers
            if self._writers:
                for w in self._writers.values():
                    w.close()
                self._writers = None

        # Save trajectory.json
        with open(self.output_dir / "trajectory.json", "w") as f:
            json.dump(
                {"task": task, "steps": steps} if isinstance(trajectory, dict) else steps,
                f,
                indent=2,
            )

        # Save scene.json
        with open(self.output_dir / "scene.json", "w") as f:
            json.dump(scene, f, indent=2)

        # Save metadata.json
        metadata = {
            "task": task,
            "total_frames": self._frame_count,
            "fps": self.fps,
            "resolution": [self.render_width, self.render_height],
            "steps": self._step_metadata,
        }
        with open(self.output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        return metadata

    def _save_key_frames(
        self, frames_dir: Path, prefix: str, frames: dict[str, np.ndarray]
    ):
        """Save a single set of key frames as PNGs."""
        for label, img in frames.items():
            parts = label.split("_", 1)
            if len(parts) == 2 and parts[0].startswith("robot"):
                path = frames_dir / parts[0] / parts[1] / f"{prefix}.png"
            else:
                path = frames_dir / label / f"{prefix}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            imageio.imwrite(str(path), img)

    def close(self):
        """Close the shared env."""
        super().close()
