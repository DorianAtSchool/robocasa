"""
Trajectory runner for multi-agent task visualization.

Builds a Kitchen environment, extracts the scene description (fixtures, objects),
and executes trajectories defined as sequences of three primitives:

  - move_object(object, from, to)
  - interact(fixture, action, ...)
  - communicate(to, message)

Each step produces before/after visual observations from multiple cameras.
Robots are teleported near the fixture they interact with so camera views match.

Usage:

    runner = TrajectoryRunner(task_name="Kitchen", robots=2, layout=11, style=34)
    scene = runner.get_scene_description()
    # ... pass scene to LLM, get trajectory back ...
    result = runner.run(trajectory)
    # result.steps[i].before / .after  -> dict of camera_name -> np.ndarray
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import robosuite.utils.transform_utils as T
from robosuite.controllers import load_composite_controller_config
from robosuite.environments.base import make

import robocasa  # noqa: F401  — registers envs
import robocasa.utils.camera_utils as CamUtils
import robocasa.utils.env_utils as EnvUtils
import robocasa.utils.object_utils as OU
from robocasa.models.fixtures import FixtureType
from robocasa.wrappers.enclosing_wall_render_wrapper import EnclosingWallRenderWrapper
from robocasa.models.fixtures.fixture import Fixture
from robocasa.models.fixtures.fixture_utils import fixture_is_type
from robocasa.utils.occupancy_grid import OccupancyGrid
from robocasa.utils.placement import ContinuousPlacement

# Minimum 2-D distance between robots; used as a safety check.
# Two robots side-by-side need ~0.40 m (0.18 radius × 2 + margin).
MIN_ROBOT_SEPARATION = 0.40
_OBJECT_PLACEMENT_MARGIN = 0.01
_OBJECT_PLACEMENT_STEP = 0.08
_OBJECT_PLACEMENT_MAX_AXIS_SAMPLES = 7


# ---------------------------------------------------------------------------
# Fixture type classification
# ---------------------------------------------------------------------------

# Fixture types where objects can be placed on/in.
_PLACEABLE_FIXTURE_TYPES: set[int] = {
    FixtureType.COUNTER,
    FixtureType.COUNTER_NON_CORNER,
    FixtureType.COUNTER_NON_DINING,
    FixtureType.DINING_COUNTER,
    FixtureType.ISLAND,
    FixtureType.CABINET,
    FixtureType.CABINET_WITH_DOOR,
    FixtureType.CABINET_SINGLE_DOOR,
    FixtureType.CABINET_DOUBLE_DOOR,
    FixtureType.DRAWER,
    FixtureType.TOP_DRAWER,
    FixtureType.SINK,
    FixtureType.STOVE,
    FixtureType.MICROWAVE,
    FixtureType.OVEN,
    FixtureType.FRIDGE,
    FixtureType.COFFEE_MACHINE,
    FixtureType.DISH_RACK,
}

# Per-robot cameras to keep, in priority order.
# Not all robots have agentview cameras (robot1 often only has robotview + eye_in_hand).
_AGENT_CAMERA_SUFFIXES = [
    "agentview_center",
    "agentview_left",
    "agentview_right",
    "eye_in_hand",
]

# Room-view framing parameters. These are intentionally separate from top-view
# distance tuning so we can keep the oblique room camera tighter around the
# active task workspace while preserving enough margin to avoid accidental
# cropping.
ROOM_VIEW_FIXTURE_RADIUS = 1.35
ROOM_VIEW_XY_MARGIN = 1.36
ROOM_VIEW_Z_LOOKAT_FRACTION = 0.45
ROOM_VIEW_BASE_DISTANCE_SCALE = 1.00
ROOM_VIEW_MIN_DISTANCE = 4.0
TOP_VIEW_XY_MARGIN = 1.12
TOP_VIEW_MIN_DISTANCE = 6.0


def _classify_fixture(fixture: Fixture) -> str | None:
    """Return the most specific FixtureType name for a fixture, or None."""
    # Check specific types before general ones to get the most useful label.
    priority_order = [
        FixtureType.COFFEE_MACHINE,
        FixtureType.MICROWAVE,
        FixtureType.STOVE,
        FixtureType.OVEN,
        FixtureType.SINK,
        FixtureType.FRIDGE,
        FixtureType.DISHWASHER,
        FixtureType.TOASTER,
        FixtureType.TOASTER_OVEN,
        FixtureType.BLENDER,
        FixtureType.STAND_MIXER,
        FixtureType.ELECTRIC_KETTLE,
        FixtureType.DISH_RACK,
        FixtureType.TOP_DRAWER,
        FixtureType.DRAWER,
        FixtureType.CABINET_SINGLE_DOOR,
        FixtureType.CABINET_DOUBLE_DOOR,
        FixtureType.CABINET_WITH_DOOR,
        FixtureType.CABINET,
        FixtureType.ISLAND,
        FixtureType.DINING_COUNTER,
        FixtureType.COUNTER_NON_DINING,
        FixtureType.COUNTER_NON_CORNER,
        FixtureType.COUNTER,
        FixtureType.STOOL,
        FixtureType.WINDOW,
    ]
    for ftype in priority_order:
        try:
            if fixture_is_type(fixture, ftype):
                return ftype.name.lower()
        except (ValueError, Exception):
            continue
    return None


def _get_interactions(fixture: Fixture) -> list[str]:
    """Return list of interaction verbs this fixture supports."""
    interactions = []
    if hasattr(fixture, "door_joint_names") and fixture.door_joint_names:
        interactions.extend(["open", "close"])
    # Knob / button joints — exclude door, drawer, and stack joints
    if hasattr(fixture, "_joint_infos"):
        skip_patterns = ("door", "drawer", "stack")
        non_door = [
            j for j in fixture._joint_infos
            if not any(p in j.lower() for p in skip_patterns)
        ]
        if non_door:
            interactions.append("turn_on")
    return interactions


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FixtureInfo:
    """Scene description of a single fixture for the LLM."""
    fixture_id: str
    fixture_type: str
    position: list[float]
    interactions: list[str]
    can_place_objects: bool
    nearby_fixtures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class ObjectInfo:
    """Scene description of a single object."""
    object_id: str
    object_type: str
    location: str  # fixture_id where the object currently sits

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class StepResult:
    """Visual observations for one trajectory step."""
    step_index: int
    agent_id: str
    action: str
    args: dict
    before: dict[str, np.ndarray]  # camera_name -> image
    after: dict[str, np.ndarray]

    def save_images(self, output_dir: str | Path, format: str = "png"):
        """Save before/after images to disk."""
        import imageio

        output_dir = Path(output_dir)
        for phase in ("before", "after"):
            images = getattr(self, phase)
            for cam_name, img in images.items():
                path = output_dir / f"step_{self.step_index:03d}_{phase}_{cam_name}.{format}"
                imageio.imwrite(str(path), img)


@dataclass
class AgentStepView:
    """One step from a single agent's perspective."""
    step_index: int
    action: str | None  # the action taken, or None if this agent didn't act
    args: dict | None
    message_received: str | None  # incoming message from other agent
    before: dict[str, np.ndarray]
    after: dict[str, np.ndarray]


@dataclass
class AgentTrajectory:
    """Per-agent view of a trajectory, for VLM training."""
    agent_id: str
    camera_names: list[str]
    initial_obs: dict[str, np.ndarray]
    steps: list[AgentStepView]
    scene: dict

    def save(self, output_dir: str | Path, format: str = "png"):
        """Save per-agent training data."""
        import imageio

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for cam_name, img in self.initial_obs.items():
            imageio.imwrite(str(output_dir / f"initial_{cam_name}.{format}"), img)
        for step in self.steps:
            for phase in ("before", "after"):
                images = getattr(step, phase)
                for cam_name, img in images.items():
                    path = output_dir / f"step_{step.step_index:03d}_{phase}_{cam_name}.{format}"
                    imageio.imwrite(str(path), img)

        meta = {
            "agent_id": self.agent_id,
            "cameras": self.camera_names,
            "steps": [
                {
                    "step_index": s.step_index,
                    "action": s.action,
                    "args": s.args,
                    "message_received": s.message_received,
                }
                for s in self.steps
            ],
        }
        with open(output_dir / "agent_trajectory.json", "w") as f:
            json.dump(meta, f, indent=2)


@dataclass
class TrajectoryResult:
    """Full result from running a trajectory."""
    initial_obs: dict[str, np.ndarray]
    steps: list[StepResult]
    scene: dict

    def save(self, output_dir: str | Path, format: str = "png"):
        """Save all images and the scene description."""
        import imageio

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for cam_name, img in self.initial_obs.items():
            imageio.imwrite(str(output_dir / f"initial_{cam_name}.{format}"), img)
        for step in self.steps:
            step.save_images(output_dir, format=format)
        with open(output_dir / "scene.json", "w") as f:
            json.dump(self.scene, f, indent=2)

    def split_by_agent(self) -> dict[str, AgentTrajectory]:
        """
        Split the combined trajectory into per-agent views for VLM training.

        Each agent sees every step but with different info:
          - Steps where it acts: action/args populated
          - Steps where the other agent acts: action=None, just before/after images
          - Incoming communications: message_received is populated
          - Camera images filtered to this agent's cameras + shared room_view
        """
        agent_ids = sorted({s.agent_id for s in self.steps})
        all_cameras = list(self.initial_obs.keys())

        def agent_cameras(agent_id: str) -> list[str]:
            idx = agent_id.replace("agent_", "")
            prefix = f"robot{idx}_"
            agent_cams = [c for c in all_cameras if c.startswith(prefix)]
            shared = [c for c in all_cameras if c == "room_view"]
            return agent_cams + shared if agent_cams else all_cameras

        def filter_cameras(images: dict[str, np.ndarray], cameras: list[str]) -> dict[str, np.ndarray]:
            return {k: v for k, v in images.items() if k in cameras}

        result = {}
        for agent_id in agent_ids:
            cams = agent_cameras(agent_id)
            agent_steps = []

            for step in self.steps:
                is_actor = step.agent_id == agent_id
                message_received = None

                if not is_actor and step.action == "communicate":
                    to = (step.args or {}).get("to")
                    if to == agent_id:
                        message_received = (step.args or {}).get("message")

                agent_steps.append(AgentStepView(
                    step_index=step.step_index,
                    action=step.action if is_actor else None,
                    args=step.args if is_actor else None,
                    message_received=message_received,
                    before=filter_cameras(step.before, cams),
                    after=filter_cameras(step.after, cams),
                ))

            result[agent_id] = AgentTrajectory(
                agent_id=agent_id,
                camera_names=cams,
                initial_obs=filter_cameras(self.initial_obs, cams),
                steps=agent_steps,
                scene=self.scene,
            )

        return result


# ---------------------------------------------------------------------------
# TrajectoryRunner
# ---------------------------------------------------------------------------

class TrajectoryRunner:
    """
    Builds a Kitchen env, extracts scene info, and executes trajectories
    by teleporting objects and toggling fixture states.
    Robots are moved near the fixture they interact with each step.
    """

    def __init__(
        self,
        task_name: str = "Kitchen",
        robots: int = 2,
        layout: int | None = None,
        style: int | None = None,
        seed: int | None = None,
        camera_names: list[str] | None = None,
        render_width: int = 512,
        render_height: int = 512,
        gl_backend: str = "osmesa",
        placement: str = "continuous",
        cell_size: float = 0.10,
        align_to_wall: bool = True,
        standoff: float = 0.40,
        sample_spacing: float = 0.08,
        robot_radius: float = 0.18,
    ):
        os.environ.setdefault("MUJOCO_GL", gl_backend)

        self._num_robots = robots
        robot_list = ["PandaOmron"] * robots

        env_kwargs = dict(
            env_name=task_name,
            robots=robot_list,
            controller_configs=load_composite_controller_config(robot="PandaOmron"),
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=False,
            ignore_done=True,
        )
        if layout is not None:
            env_kwargs["layout_ids"] = [layout]
        if style is not None:
            env_kwargs["style_ids"] = [style]
        if seed is not None:
            env_kwargs["seed"] = seed

        self.env = make(**env_kwargs)
        # Wrap env to make enclosing walls translucent (required for room_view
        # free camera to see through walls, same as two_robot_video_sample.py)
        self.env = EnclosingWallRenderWrapper(self.env, alpha=0.1, enabled=True)
        self.env.reset()

        self.render_width = render_width
        self.render_height = render_height

        # Build camera list: per-robot cameras + room_view
        if camera_names is None:
            self.camera_names = self._build_camera_list()
        else:
            self.camera_names = camera_names

        # Build fixture index and placement strategy
        self._fixtures: dict[str, Fixture] = dict(self.env.fixtures)
        self._placement_mode = placement
        grid_kwargs = dict(cell_size=cell_size, align_to_wall=align_to_wall,
                           standoff=standoff, sample_spacing=sample_spacing)
        continuous_kwargs = dict(standoff=standoff, sample_spacing=sample_spacing, robot_radius=robot_radius)
        # Always init both strategies: grid for cell-based queries and
        # continuous for standability validation (room bounds, collision, enclosed).
        self._occupancy_grid = OccupancyGrid(self._fixtures, **grid_kwargs)
        self._continuous = ContinuousPlacement(self._fixtures, **continuous_kwargs)

        base_room_cam_config = CamUtils.LAYOUT_CAMS.get(
            self.env.layout_id, CamUtils.DEFAULT_LAYOUT_CAM
        )
        self._room_cam_config = self._compute_room_cam_config(dict(base_room_cam_config))
        self._top_cam_config = self._compute_top_cam_config(self._room_cam_config)

        self._scene: dict | None = None
        self._object_locations: dict[str, str] = {}

    def _build_camera_list(self) -> list[str]:
        """Build the trimmed camera list: per-robot cameras + shared room_view."""
        all_cams = set(
            self.env.sim.model.camera_id2name(i)
            for i in range(self.env.sim.model.ncam)
        )
        cameras = []
        for robot_idx in range(self._num_robots):
            for suffix in _AGENT_CAMERA_SUFFIXES:
                cam = f"robot{robot_idx}_{suffix}"
                if cam in all_cams:
                    cameras.append(cam)
        # room_view is rendered via free camera (shared across agents)
        cameras.append("room_view")
        cameras.append("top_view")
        return cameras

    # ------------------------------------------------------------------
    # Room-view rendering (free camera with EnclosingWallRenderWrapper)
    # ------------------------------------------------------------------

    def _collect_scene_points(
        self,
        include_objects: bool = True,
        include_robots: bool = True,
        fixture_ids: set[str] | None = None,
    ) -> np.ndarray:
        points = []

        for fixture_id, fixture in self.env.fixtures.items():
            if fixture_ids is not None and fixture_id not in fixture_ids:
                continue
            if hasattr(fixture, "get_bbox_points"):
                try:
                    bbox_points = np.asarray(fixture.get_bbox_points(), dtype=float)
                except Exception:
                    bbox_points = None
                if bbox_points is not None and bbox_points.ndim == 2 and bbox_points.shape[1] >= 3:
                    points.append(bbox_points[:, :3])
                    continue

            if hasattr(fixture, "pos") and fixture.pos is not None:
                points.append(np.asarray(fixture.pos, dtype=float).reshape(1, 3))

        if include_objects and hasattr(self.env, "obj_body_id"):
            for object_id, body_id in self.env.obj_body_id.items():
                try:
                    obj_pos = self.env.sim.data.body_xpos[body_id].copy()
                except Exception:
                    continue
                points.append(np.asarray(obj_pos, dtype=float).reshape(1, 3))

        if include_robots:
            for robot_idx in range(self._num_robots):
                points.append(self._get_robot_position(robot_idx).reshape(1, 3))

        if not points:
            return np.zeros((0, 3), dtype=float)
        return np.concatenate(points, axis=0)

    def _collect_focus_points(self, fixture_radius: float) -> np.ndarray:
        """Collect points around fixtures near the current task workspace."""
        focus_fixture_ids: set[str] = set()
        focus_centers_xy = []

        if hasattr(self.env, "obj_body_id"):
            for body_id in self.env.obj_body_id.values():
                try:
                    obj_pos = self.env.sim.data.body_xpos[body_id].copy()
                except Exception:
                    continue
                focus_centers_xy.append(obj_pos[:2])

        for robot_idx in range(self._num_robots):
            focus_centers_xy.append(self._get_robot_position(robot_idx)[:2])

        if not focus_centers_xy:
            return self._collect_scene_points(include_objects=True, include_robots=True)

        for fixture_id, fixture in self.env.fixtures.items():
            if not hasattr(fixture, "pos") or fixture.pos is None:
                continue
            fixture_xy = np.asarray(fixture.pos[:2], dtype=float)
            if any(
                float(np.linalg.norm(fixture_xy - center_xy)) <= fixture_radius
                for center_xy in focus_centers_xy
            ):
                focus_fixture_ids.add(fixture_id)

        if not focus_fixture_ids:
            return self._collect_scene_points(include_objects=True, include_robots=True)

        return self._collect_scene_points(
            include_objects=True,
            include_robots=True,
            fixture_ids=focus_fixture_ids,
        )

    def _collect_room_view_points(self) -> np.ndarray:
        """Collect a tighter set of points around the active task workspace."""
        return self._collect_focus_points(ROOM_VIEW_FIXTURE_RADIUS)

    def _collect_top_view_points(self) -> np.ndarray:
        """Collect the same focused kitchen footprint used by room_view.

        A broader fixture radius tends to pull in adjacent rooms / hallways on
        larger layouts, which makes the overhead camera zoom out too far.
        """
        return self._collect_room_view_points()

    def _compute_room_cam_config(self, base_cam_config: dict) -> dict:
        """Derive an oblique room camera from the current scene footprint."""
        scene_points = self._collect_room_view_points()
        if scene_points.size == 0:
            return dict(base_cam_config)

        min_xyz = np.min(scene_points, axis=0)
        max_xyz = np.max(scene_points, axis=0)
        center_xyz = 0.5 * (min_xyz + max_xyz)
        z_extent = max_xyz[2] - min_xyz[2]

        fovy_deg = float(getattr(self.env.sim.model.vis.global_, "fovy", 45.0))
        lookat = np.asarray(base_cam_config["lookat"], dtype=float).copy()
        lookat[0] = center_xyz[0]
        lookat[1] = center_xyz[1]
        lookat[2] = max(
            lookat[2],
            float(min_xyz[2] + ROOM_VIEW_Z_LOOKAT_FRACTION * max(z_extent, 1.0)),
        )

        azimuth_rad = np.deg2rad(float(base_cam_config["azimuth"]))
        elevation_rad = np.deg2rad(float(base_cam_config["elevation"]))
        forward = np.array(
            [
                -np.cos(elevation_rad) * np.cos(azimuth_rad),
                -np.cos(elevation_rad) * np.sin(azimuth_rad),
                -np.sin(elevation_rad),
            ],
            dtype=float,
        )
        forward /= np.linalg.norm(forward)
        world_up = np.array([0.0, 0.0, 1.0], dtype=float)
        right = np.cross(forward, world_up)
        if np.linalg.norm(right) < 1e-6:
            right = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        up /= np.linalg.norm(up)

        centered_points = scene_points - lookat
        right_extent = float(np.max(np.abs(centered_points @ right)))
        up_extent = float(np.max(np.abs(centered_points @ up)))

        half_fovy_rad = np.deg2rad(np.clip(fovy_deg, 1.0, 89.0) / 2.0)
        aspect = max(float(self.render_width) / float(self.render_height), 1e-6)
        half_fovx_rad = np.arctan(np.tan(half_fovy_rad) * aspect)
        required_distance_x = right_extent / max(np.tan(half_fovx_rad), 1e-6)
        required_distance_y = up_extent / max(np.tan(half_fovy_rad), 1e-6)
        required_distance = max(required_distance_x, required_distance_y) * ROOM_VIEW_XY_MARGIN

        return dict(
            lookat=lookat.tolist(),
            distance=float(
                max(
                    ROOM_VIEW_MIN_DISTANCE,
                    base_cam_config["distance"] * ROOM_VIEW_BASE_DISTANCE_SCALE,
                    required_distance,
                )
            ),
            azimuth=float(base_cam_config["azimuth"]),
            elevation=float(base_cam_config["elevation"]),
        )

    def _compute_top_cam_config(self, room_cam_config: dict) -> dict:
        """Derive an overhead camera that keeps the active kitchen room in frame."""
        scene_points = self._collect_top_view_points()
        if scene_points.size == 0:
            scene_points = self._collect_scene_points(include_objects=True, include_robots=True)
        if scene_points.size == 0:
            return dict(
                lookat=list(room_cam_config["lookat"]),
                distance=max(room_cam_config["distance"] * 1.8, 8.0),
                azimuth=room_cam_config["azimuth"],
                elevation=-89.0,
            )

        xy_points = scene_points[:, :2]
        min_xy = np.min(xy_points, axis=0)
        max_xy = np.max(xy_points, axis=0)
        center_xy = 0.5 * (min_xy + max_xy)
        max_radius = np.max(np.linalg.norm(xy_points - center_xy, axis=1))

        fovy_deg = float(getattr(self.env.sim.model.vis.global_, "fovy", 45.0))
        half_fovy_rad = np.deg2rad(np.clip(fovy_deg, 1.0, 89.0) / 2.0)
        required_distance = (max_radius / np.tan(half_fovy_rad)) * TOP_VIEW_XY_MARGIN

        lookat = np.asarray(room_cam_config["lookat"], dtype=float).copy()
        lookat[:2] = center_xy
        return dict(
            lookat=lookat.tolist(),
            distance=float(max(TOP_VIEW_MIN_DISTANCE, required_distance)),
            azimuth=float(room_cam_config["azimuth"]),
            elevation=-89.0,
        )

    def _render_free_camera(self, cam_config: dict) -> np.ndarray:
        """Render a free camera with an explicit camera config."""
        render_ctx = self.env.sim._render_context_offscreen
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

    def _render_room_view(self) -> np.ndarray:
        """Render the layout-wide room camera."""
        return self._render_free_camera(self._room_cam_config)

    def _render_top_view(self) -> np.ndarray:
        """Render an overhead free camera focused on the active kitchen room."""
        return self._render_free_camera(self._top_cam_config)

    # ------------------------------------------------------------------
    # Robot positioning
    # ------------------------------------------------------------------

    def _get_robot_position(self, robot_idx: int) -> np.ndarray:
        """Return the current (x, y, z) world position of a robot's mobile base.

        Note: ``robot.robot_model.root_body`` (e.g. ``robot0_base``) is a
        fixed anchor at (10, 10, 0) — **not** the real base.  The actual
        movable platform is the ``mobilebase{idx}_base`` body.
        """
        body_name = f"mobilebase{robot_idx}_base"
        body_id = self.env.sim.model.body_name2id(body_name)
        return self.env.sim.data.body_xpos[body_id].copy()

    def _robots_too_close(self, idx_a: int, idx_b: int) -> bool:
        """Return True if two robots are closer than MIN_ROBOT_SEPARATION."""
        pa = self._get_robot_position(idx_a)
        pb = self._get_robot_position(idx_b)
        return float(np.linalg.norm(pa[:2] - pb[:2])) < MIN_ROBOT_SEPARATION

    def _lookup_named_world_xy(self, name: str) -> np.ndarray | None:
        """Return the world XY position of a named site, geom, or body."""
        sim = self.env.sim
        model = sim.model
        data = sim.data

        for id_lookup, pos_array in (
            (model.site_name2id, data.site_xpos),
            (model.geom_name2id, data.geom_xpos),
            (model.body_name2id, data.body_xpos),
        ):
            try:
                idx = id_lookup(name)
            except Exception:
                continue
            if idx is None or idx < 0:
                continue
            return np.asarray(pos_array[idx][:2], dtype=float).copy()
        return None

    def _scan_fixture_named_world_xy(
        self,
        fixture: Fixture,
        name_filter,
    ) -> list[np.ndarray]:
        """Return XY positions of named sim elements associated with *fixture*."""
        sim = self.env.sim
        model = sim.model
        data = sim.data
        prefixes = []
        for prefix in (getattr(fixture, "naming_prefix", None), getattr(fixture, "name", None)):
            if isinstance(prefix, str) and prefix:
                prefixes.append(prefix)

        def _matches_prefix(name: str) -> bool:
            return any(name.startswith(prefix) for prefix in prefixes)

        matches: list[np.ndarray] = []
        for count, id_to_name, pos_array in (
            (model.nsite, model.site_id2name, data.site_xpos),
            (model.ngeom, model.geom_id2name, data.geom_xpos),
            (model.nbody, model.body_id2name, data.body_xpos),
        ):
            for idx in range(count):
                name = id_to_name(idx)
                if not isinstance(name, str) or not _matches_prefix(name):
                    continue
                if not name_filter(name.lower()):
                    continue
                matches.append(np.asarray(pos_array[idx][:2], dtype=float).copy())
        return matches

    def _get_fixture_front_target_xy(self, fixture_id: str) -> np.ndarray | None:
        """Return the fixture's front working target, preferring handle geometry."""
        fixture = self._fixtures.get(fixture_id)
        if fixture is None:
            return None

        explicit_handle_names: list[str] = []
        for attr_name in ("left_handle_name", "right_handle_name", "handle_name"):
            try:
                candidate = getattr(fixture, attr_name)
            except Exception:
                candidate = None
            if isinstance(candidate, str):
                explicit_handle_names.append(candidate)

        handle_positions = [
            pos for pos in
            (self._lookup_named_world_xy(name) for name in explicit_handle_names)
            if pos is not None
        ]
        if handle_positions:
            return np.mean(np.stack(handle_positions), axis=0)

        handle_positions = self._scan_fixture_named_world_xy(
            fixture,
            lambda name: "handle" in name and ("main" in name or name.endswith("_handle")),
        )
        if handle_positions:
            return np.mean(np.stack(handle_positions), axis=0)

        handle_positions = self._scan_fixture_named_world_xy(
            fixture,
            lambda name: "handle" in name,
        )
        if handle_positions:
            return np.mean(np.stack(handle_positions), axis=0)

        try:
            door_name = getattr(fixture, "door_name")
        except Exception:
            door_name = None
        if isinstance(door_name, str):
            door_pos = self._lookup_named_world_xy(door_name)
            if door_pos is not None:
                return door_pos

        return None

    def _resolve_fixture_target_pos(
        self,
        fixture_id: str,
        ref_object_id: str | None = None,
        ref_pos_override: np.ndarray | None = None,
        require_front: bool = False,
    ) -> np.ndarray | None:
        """Resolve the placement target for a fixture interaction."""
        if ref_pos_override is not None:
            return np.asarray(ref_pos_override, dtype=float)[:2]
        if require_front:
            return self._get_fixture_front_target_xy(fixture_id)
        return self._resolve_ref_object_pos(ref_object_id)

    def _offset_robot_beside_other(
        self,
        robot_idx: int,
        fixture_id: str,
        ref_object_id: str | None = None,
        ref_pos_override: np.ndarray | None = None,
        require_front: bool = False,
    ):
        """Re-place *robot_idx* beside the other robot at the same fixture.

        Re-queries placement with both the other robot's position and the
        current (colliding) position excluded.
        """
        if fixture_id not in self._fixtures:
            return
        fxtr = self._fixtures[fixture_id]
        ref_pos = self._resolve_fixture_target_pos(
            fixture_id,
            ref_object_id=ref_object_id,
            ref_pos_override=ref_pos_override,
            require_front=require_front,
        )

        if self._placement_mode == "continuous" and self._continuous is not None:
            # For continuous: include current robot position as a "robot" to avoid
            robot_positions = []
            for other_idx in range(self._num_robots):
                if other_idx != robot_idx:
                    robot_positions.append(self._get_robot_position(other_idx)[:2])
            # Also add own position to force a different result
            robot_positions.append(self._get_robot_position(robot_idx)[:2])
            result = self._continuous.find_placement(
                fxtr, robot_positions, ref_pos, require_front=require_front,
            )
        else:
            robot_cells = []
            robot_positions = []
            for other_idx in range(self._num_robots):
                if other_idx != robot_idx:
                    other_pos = self._get_robot_position(other_idx)[:2]
                    robot_cells.append(self._occupancy_grid._world_to_grid(other_pos))
                    robot_positions.append(other_pos)
            my_pos = self._get_robot_position(robot_idx)[:2]
            my_cell = self._occupancy_grid._world_to_grid(my_pos)
            if my_cell not in robot_cells:
                robot_cells.append(my_cell)
            result = self._occupancy_grid.find_placement(
                fxtr,
                robot_cells,
                ref_pos,
                robot_positions=robot_positions,
                require_front=require_front,
            )

        if result is not None:
            pos_xy, yaw = result
            if self._continuous is not None:
                if (not self._continuous.is_standable(pos_xy) or
                        self._continuous.is_inside_any_fixture(pos_xy)):
                    return
                if self._occupancy_grid is not None:
                    grid_ok = (
                        self._occupancy_grid.is_free_of_fixtures(pos_xy)
                        if require_front else
                        self._occupancy_grid.is_free(pos_xy)
                    )
                    if not grid_ok:
                        return
            self._set_robot_pose(robot_idx, pos_xy, yaw)

    def _resolve_ref_object_pos(self, ref_object_id: str | None) -> np.ndarray | None:
        """Return the 2D world position of a reference object, or None."""
        if ref_object_id is None:
            return None
        try:
            body_id = self.env.obj_body_id.get(ref_object_id)
            if body_id is not None:
                return self.env.sim.data.body_xpos[body_id][:2].copy()
        except Exception:
            pass
        return None

    def _move_robot_near_fixture(
        self,
        robot_idx: int,
        fixture_id: str,
        ref_object_id: str | None = None,
        ref_pos_override: np.ndarray | None = None,
        require_front: bool = False,
    ) -> bool:
        """Teleport a robot's base near the given fixture, facing it.

        Delegates to the active placement strategy (grid or continuous).
        If, after placement, another robot is too close (< MIN_ROBOT_SEPARATION),
        the moved robot is re-placed to an alternative position.

        Args:
            ref_pos_override: explicit 2D position to prefer (takes precedence
                over ``ref_object_id``).  Used when the target position is
                known before the object is actually placed (e.g.,
                ``place_on_surface`` pre-computes the landing spot).
            require_front: If True, robot must approach from the fixture's
                front face (for interactive fixtures like fridge, cabinet).
                If False, pick the closest valid position on any face.

        Returns:
            True if the robot was successfully placed, False if no valid
            position was found (robot stays at current position).
        """
        if fixture_id not in self._fixtures:
            return False
        fxtr = self._fixtures[fixture_id]
        ref_pos = self._resolve_fixture_target_pos(
            fixture_id,
            ref_object_id=ref_object_id,
            ref_pos_override=ref_pos_override,
            require_front=require_front,
        )

        if self._placement_mode == "continuous" and self._continuous is not None:
            result = self._move_robot_continuous(robot_idx, fxtr, ref_pos, require_front)
        else:
            result = self._move_robot_grid(robot_idx, fxtr, ref_pos, require_front)

        # Validate result with strategy-specific checks.
        # Grid results: grid's own validation is sufficient (flood-fill + cell occupancy).
        # Continuous results: also check grid reachability (flood-fill catches enclosed pockets).
        if result is not None:
            pos_xy, yaw = result
            rejected = False
            if self._placement_mode == "continuous" and self._continuous is not None:
                if (not self._continuous.is_standable(pos_xy) or
                        self._continuous.is_inside_any_fixture(pos_xy)):
                    rejected = True
                # Grid reachability catches enclosed pockets for continuous
                if self._occupancy_grid is not None:
                    grid_ok = (
                        self._occupancy_grid.is_free_of_fixtures(pos_xy)
                        if require_front else
                        self._occupancy_grid.is_free(pos_xy)
                    )
                    if not grid_ok:
                        rejected = True
            if rejected:
                result = None  # reject, try fallback

        placed = False
        if result is not None:
            pos_xy, yaw = result
            self._set_robot_pose(robot_idx, pos_xy, yaw)
            placed = True
        else:
            # Fallback: try the OTHER placement strategy before giving up.
            fallback_result = None
            if self._placement_mode != "continuous" and self._continuous is not None:
                fallback_result = self._move_robot_continuous(robot_idx, fxtr, ref_pos, require_front)
            elif self._placement_mode == "continuous" and self._occupancy_grid is not None:
                fallback_result = self._move_robot_grid(robot_idx, fxtr, ref_pos, require_front)

            # Validate fallback — use the fallback strategy's own validation.
            # Fallback is the OTHER strategy, so apply its specific checks.
            if fallback_result is not None:
                pos_xy, yaw = fallback_result
                rejected = False
                # Fallback from grid mode → continuous was tried
                if self._placement_mode != "continuous" and self._continuous is not None:
                    if (not self._continuous.is_standable(pos_xy) or
                            self._continuous.is_inside_any_fixture(pos_xy)):
                        rejected = True
                    if self._occupancy_grid is not None:
                        grid_ok = (
                            self._occupancy_grid.is_free_of_fixtures(pos_xy)
                            if require_front else
                            self._occupancy_grid.is_free(pos_xy)
                        )
                        if not grid_ok:
                            rejected = True
                # Fallback from continuous mode → grid was tried (grid validates itself)
                if rejected:
                    fallback_result = None
            if fallback_result is not None:
                pos_xy, yaw = fallback_result
                self._set_robot_pose(robot_idx, pos_xy, yaw)
                placed = True
            elif self._continuous is not None:
                # Last resort: try multiple directions and standoffs to find
                # a standable position. Interactive fixtures stay on the front
                # approach line; only surface placement may fall back to sides.
                angle = getattr(fxtr, "rot", 0.0) or 0.0
                directions = [angle + np.pi]
                if not require_front:
                    directions.extend([
                        angle + np.pi / 2,   # left side
                        angle - np.pi / 2,   # right side
                        angle,               # behind (last resort)
                    ])
                for direction in directions:
                    for standoff in (0.6, 0.8, 0.4, 1.0, 1.2, 1.5):
                        fallback_pos = np.array(fxtr.pos[:2], dtype=float)
                        fallback_pos[0] += standoff * np.cos(direction)
                        fallback_pos[1] += standoff * np.sin(direction)
                        grid_ok = True
                        if self._occupancy_grid is not None:
                            grid_ok = (
                                self._occupancy_grid.is_free_of_fixtures(fallback_pos)
                                if require_front else
                                self._occupancy_grid.is_free(fallback_pos)
                            )
                        if (not self._continuous._collides_with_obstacles(fallback_pos, exclude_fixture=fxtr)
                                and self._continuous.is_standable(fallback_pos)
                                and not self._continuous.is_inside_any_fixture(fallback_pos)
                                and grid_ok):
                            self._set_robot_pose(robot_idx, fallback_pos, direction)
                            placed = True
                            break
                    if placed:
                        break

        # --- If overlapping another robot, shift to its side ---
        if placed:
            for other_idx in range(self._num_robots):
                if other_idx == robot_idx:
                    continue
                if self._robots_too_close(robot_idx, other_idx):
                    self._offset_robot_beside_other(
                        robot_idx,
                        fixture_id,
                        ref_object_id=ref_object_id,
                        ref_pos_override=ref_pos,
                        require_front=require_front,
                    )
                    break  # only one correction needed for 2-robot setups

        return placed

    def _move_robot_continuous(
        self,
        robot_idx: int,
        fixture: Fixture,
        ref_pos: np.ndarray | None,
        require_front: bool = False,
    ) -> tuple[np.ndarray, float] | None:
        """Find placement using continuous AABB collision checks."""
        robot_positions = []
        for other_idx in range(self._num_robots):
            if other_idx == robot_idx:
                continue
            robot_positions.append(self._get_robot_position(other_idx)[:2])
        return self._continuous.find_placement(
            fixture, robot_positions, ref_pos, require_front=require_front,
        )

    def _move_robot_grid(
        self,
        robot_idx: int,
        fixture: Fixture,
        ref_pos: np.ndarray | None,
        require_front: bool = False,
    ) -> tuple[np.ndarray, float] | None:
        """Find placement using the occupancy grid."""
        robot_cells = []
        robot_positions = []
        for other_idx in range(self._num_robots):
            if other_idx == robot_idx:
                continue
            other_pos = self._get_robot_position(other_idx)[:2]
            robot_cells.append(self._occupancy_grid._world_to_grid(other_pos))
            robot_positions.append(other_pos)
        return self._occupancy_grid.find_placement(
            fixture, robot_cells, ref_pos, robot_positions=robot_positions,
            require_front=require_front,
        )

    def _find_nearest_surface(self, pos_2d: np.ndarray) -> str | None:
        """Find the nearest counter / placeable surface to a 2-D position."""
        best_id, best_dist = None, float("inf")
        for fid, fxtr in self._fixtures.items():
            ftype = _classify_fixture(fxtr)
            if ftype is None or "counter" not in ftype:
                continue
            d = float(np.linalg.norm(np.array(fxtr.pos[:2]) - pos_2d))
            if d < best_dist:
                best_dist = d
                best_id = fid
        return best_id

    def give_space(
        self,
        robot_idx: int,
        fixture_id: str,
        min_distance: float = 1.5,
    ):
        """Move *robot_idx* to open floor space away from *fixture_id*.

        Grid mode: finds the nearest free cell that is ≥ *min_distance* from
        the fixture center and not occupied by another robot.

        Continuous mode: samples positions on a circle around the fixture at
        increasing radii until a standable, collision-free position is found.
        """
        if fixture_id not in self._fixtures:
            return
        fxtr = self._fixtures[fixture_id]
        fxtr_pos = np.asarray(fxtr.pos[:2], dtype=float)

        # Collect other robot positions
        other_positions = []
        for other_idx in range(self._num_robots):
            if other_idx == robot_idx:
                continue
            other_positions.append(self._get_robot_position(other_idx)[:2])

        best_pos = None
        best_yaw = None
        robot_pos = self._get_robot_position(robot_idx)[:2]

        def _is_valid_candidate(pos):
            """Check a candidate is collision-free and standable."""
            for rp in other_positions:
                if float(np.linalg.norm(pos - rp)) < MIN_ROBOT_SEPARATION:
                    return False
            # Always validate standability if continuous placement is available
            if self._continuous is not None:
                if self._continuous._collides_with_obstacles(pos):
                    return False
                if not self._continuous.is_standable(pos):
                    return False
            return True

        def _try_continuous():
            """Sample on expanding circles around the fixture."""
            nonlocal best_pos, best_yaw
            if self._continuous is None:
                return
            for radius in np.arange(min_distance, min_distance + 3.0, 0.3):
                for angle in np.linspace(0, 2 * np.pi, 16, endpoint=False):
                    candidate = fxtr_pos + radius * np.array([np.cos(angle), np.sin(angle)])
                    if not _is_valid_candidate(candidate):
                        continue
                    best_pos = candidate
                    delta = fxtr_pos - candidate
                    best_yaw = float(np.arctan2(delta[1], delta[0]))
                    return
            return

        def _try_grid():
            """Scan all free cells for the closest one far enough from fixture."""
            nonlocal best_pos, best_yaw
            if self._occupancy_grid is None:
                return
            grid = self._occupancy_grid
            best_dist_to_robot = float("inf")

            for r in range(grid._rows):
                for c in range(grid._cols):
                    if grid._grid[r, c]:
                        continue  # occupied cell
                    cell_pos = grid._grid_to_world(r, c)
                    dist_to_fixture = float(np.linalg.norm(cell_pos - fxtr_pos))
                    if dist_to_fixture < min_distance:
                        continue
                    if not _is_valid_candidate(cell_pos):
                        continue
                    dist_to_robot = float(np.linalg.norm(cell_pos - robot_pos))
                    if dist_to_robot < best_dist_to_robot:
                        best_dist_to_robot = dist_to_robot
                        best_pos = cell_pos
                        delta = fxtr_pos - cell_pos
                        best_yaw = float(np.arctan2(delta[1], delta[0]))

        # Try the preferred strategy first, then fall back to the other
        if self._placement_mode == "continuous":
            _try_continuous()
            if best_pos is None:
                _try_grid()
        else:
            _try_grid()
            if best_pos is None:
                _try_continuous()

        if best_pos is not None:
            self._set_robot_pose(robot_idx, best_pos, best_yaw)

    def hand_off_object(
        self,
        robot_idx: int,
        object_id: str,
        to_robot_idx: int,
    ):
        """Place *object_id* in front of the receiving robot and teleport the
        delivering robot to stand beside them.

        1. Find the nearest counter/surface to *to_robot_idx*'s position.
        2. Teleport the object onto that surface.
        3. Teleport *robot_idx* next to *to_robot_idx* (side-offset).
        """
        receiving_pos = self._get_robot_position(to_robot_idx)[:2]
        surface = self._find_nearest_surface(receiving_pos)
        if surface is None:
            return  # nothing we can place on

        # Place the object on that surface
        if object_id:
            self.move_object(object_id, surface)

        # Teleport delivering robot beside receiving robot at the same fixture
        self._move_robot_near_fixture(robot_idx, surface)
        # _move_robot_near_fixture already handles offset if too close

    def _set_robot_yaw(self, robot_idx: int, yaw: float | None):
        """Set the mobile base yaw joint for the robot."""
        yaw_jnt = f"mobilebase{robot_idx}_joint_mobile_yaw"
        try:
            addr = self.env.sim.model.get_joint_qpos_addr(yaw_jnt)
            if yaw is not None:
                # ori from compute_robot_base_placement_pose is an euler [0,0,yaw]
                # The yaw joint is relative to the robot's anchor orientation
                anchor_ori = getattr(self.env, "init_robot_base_ori_anchors", [None] * (robot_idx + 1))[robot_idx]
                if anchor_ori is not None:
                    self.env.sim.data.qpos[addr] = yaw - anchor_ori[2]
                else:
                    self.env.sim.data.qpos[addr] = 0.0
            else:
                self.env.sim.data.qpos[addr] = 0.0
            self.env.sim.forward()
        except Exception:
            pass

    def _set_robot_pose(self, robot_idx: int, pos_xy: np.ndarray, yaw: float | None):
        """Set robot position and yaw atomically.

        The yaw joint rotates the body offset relative to the joint anchor,
        which shifts body_xpos.  To achieve the desired world position *and*
        yaw, we: (1) set yaw, (2) set position, (3) correct for the
        yaw-induced position drift by re-setting position.
        """
        self._set_robot_yaw(robot_idx, yaw)
        pos_3d = np.array([pos_xy[0], pos_xy[1], 0.0])
        EnvUtils.set_robot_to_position(self.env, pos_3d, robot_idx=robot_idx)
        # Correct for yaw-induced body offset: read back actual position
        # and adjust if it doesn't match the target.
        actual = self._get_robot_position(robot_idx)[:2]
        error = pos_xy - actual
        if np.linalg.norm(error) > 0.01:
            corrected = pos_3d.copy()
            corrected[:2] += error
            EnvUtils.set_robot_to_position(self.env, corrected, robot_idx=robot_idx)

    def _get_step_fixture(self, action: str, args: dict) -> str | None:
        """Extract the fixture a step interacts with."""
        if action == "move_object":
            return args.get("to")
        elif action == "interact":
            return args.get("fixture") or args.get("fixture_id")
        elif action == "navigate":
            return args.get("fixture") or args.get("fixture_id")
        elif action == "move_away":
            return None  # hand_off_object handles its own positioning
        elif action == "give_space":
            return None  # give_space handles its own positioning
        return None

    # ------------------------------------------------------------------
    # Scene description
    # ------------------------------------------------------------------

    def get_scene_description(self) -> dict:
        """Extract a JSON-serializable description of the scene for the LLM."""
        if self._scene is not None:
            return self._scene

        fixtures_info = {}
        fixture_positions = {}

        for name, fxtr in self._fixtures.items():
            ftype = _classify_fixture(fxtr)
            if ftype is None:
                continue

            pos = fxtr.pos.tolist() if hasattr(fxtr, "pos") and fxtr.pos is not None else [0, 0, 0]
            fixture_positions[name] = np.array(pos[:2])

            interactions = _get_interactions(fxtr)
            can_place = any(
                fixture_is_type(fxtr, ft)
                for ft in _PLACEABLE_FIXTURE_TYPES
                if ft in FixtureType.__members__.values()
            )

            fixtures_info[name] = FixtureInfo(
                fixture_id=name,
                fixture_type=ftype,
                position=[round(p, 3) for p in pos],
                interactions=interactions,
                can_place_objects=can_place,
            )

        # Compute nearby fixtures (within 1.0m)
        for name, info in fixtures_info.items():
            if name not in fixture_positions:
                continue
            pos_a = fixture_positions[name]
            nearby = []
            for other_name, pos_b in fixture_positions.items():
                if other_name == name:
                    continue
                if np.linalg.norm(pos_a - pos_b) < 1.0:
                    nearby.append(other_name)
            info.nearby_fixtures = nearby

        # Objects — use ep_meta for rich type info when available
        objects_info = {}
        obj_cfg_map = {}
        if hasattr(self.env, "get_ep_meta"):
            ep_meta = self.env.get_ep_meta()
            for cfg in ep_meta.get("object_cfgs", []):
                obj_cfg_map[cfg["name"]] = cfg

        if hasattr(self.env, "objects") and self.env.objects:
            for obj_name in self.env.objects:
                obj_pos = self.env.sim.data.body_xpos[
                    self.env.obj_body_id[obj_name]
                ]
                location = self._find_object_fixture(obj_pos)

                cfg = obj_cfg_map.get(obj_name, {})
                info = cfg.get("info", {})
                obj_type = str(info.get("cat", obj_name))

                objects_info[obj_name] = ObjectInfo(
                    object_id=obj_name,
                    object_type=obj_type,
                    location=location,
                )
                self._object_locations[obj_name] = location

        # Robots
        robots_info = []
        for i, robot in enumerate(self.env.robots):
            robot_pos = self.env.sim.data.body_xpos[
                self.env.sim.model.body_name2id(robot.robot_model.root_body)
            ]
            robots_info.append({
                "robot_id": f"agent_{i}",
                "position": [round(float(p), 3) for p in robot_pos],
            })

        # Task instruction
        task_lang = None
        if hasattr(self.env, "get_ep_meta"):
            task_lang = self.env.get_ep_meta().get("lang")

        self._scene = {
            "task": task_lang,
            "fixtures": {k: v.to_dict() for k, v in fixtures_info.items()},
            "objects": {k: v.to_dict() for k, v in objects_info.items()},
            "cameras": self.camera_names,
            "robots": robots_info,
        }
        return self._scene

    def _find_object_fixture(self, obj_pos: np.ndarray) -> str:
        """Find the fixture closest to an object's position (2D)."""
        best_name = "unknown"
        best_dist = float("inf")
        for name, fxtr in self._fixtures.items():
            if not hasattr(fxtr, "pos") or fxtr.pos is None:
                continue
            dist = np.linalg.norm(obj_pos[:2] - fxtr.pos[:2])
            if dist < best_dist:
                best_dist = dist
                best_name = name
        return best_name

    def _set_object_location(self, object_id: str, fixture_id: str):
        """Update cached fixture grounding for an object."""
        self._object_locations[object_id] = fixture_id
        if self._scene is None:
            return
        object_info = self._scene.get("objects", {}).get(object_id)
        if isinstance(object_info, dict):
            object_info["location"] = fixture_id

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self) -> dict[str, np.ndarray]:
        """Render all configured cameras and return {camera_name: image}."""
        images = {}
        for cam_name in self.camera_names:
            if cam_name == "room_view":
                images[cam_name] = self._render_room_view()
                continue
            if cam_name == "top_view":
                images[cam_name] = self._render_top_view()
                continue
            try:
                frame = self.env.sim.render(
                    height=self.render_height,
                    width=self.render_width,
                    camera_name=cam_name,
                )[::-1]  # flip vertical (MuJoCo convention)
                images[cam_name] = frame
            except Exception:
                continue
        return images

    # ------------------------------------------------------------------
    # Primitives
    # ------------------------------------------------------------------

    def _fixture_local_to_world(
        self,
        target_fxtr: Fixture,
        local_offset: np.ndarray,
    ) -> np.ndarray:
        """Convert a fixture-local offset into a world position."""
        offset = np.asarray(local_offset, dtype=float)
        if hasattr(target_fxtr, "rot") and target_fxtr.rot is not None:
            angle = float(target_fxtr.rot)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rot_offset = np.array(
                [
                    cos_a * offset[0] - sin_a * offset[1],
                    sin_a * offset[0] + cos_a * offset[1],
                    offset[2],
                ],
                dtype=float,
            )
        else:
            rot_offset = offset
        return np.asarray(target_fxtr.pos, dtype=float) + rot_offset

    def _world_to_fixture_local(
        self,
        target_fxtr: Fixture,
        world_xy: np.ndarray,
    ) -> np.ndarray:
        """Project a world XY position into a fixture's local frame."""
        world_xy = np.asarray(world_xy, dtype=float)
        offset = world_xy - np.asarray(target_fxtr.pos[:2], dtype=float)
        if not hasattr(target_fxtr, "rot") or target_fxtr.rot is None:
            return offset
        angle = float(target_fxtr.rot)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        return np.array(
            [
                cos_a * offset[0] + sin_a * offset[1],
                -sin_a * offset[0] + cos_a * offset[1],
            ],
            dtype=float,
        )

    def _sample_axis_values(self, center: float, half_span: float) -> np.ndarray:
        """Sample positions along one fixture-local axis."""
        if half_span <= 1e-6:
            return np.array([center], dtype=float)

        axis_min = center - half_span
        axis_max = center + half_span
        span = axis_max - axis_min
        approx_count = int(np.floor(span / _OBJECT_PLACEMENT_STEP)) + 1
        count = min(_OBJECT_PLACEMENT_MAX_AXIS_SAMPLES, max(2, approx_count))
        return np.linspace(axis_min, axis_max, num=count, dtype=float)

    def _get_object_placement_metadata(self, object_id: str) -> dict[str, np.ndarray | float]:
        """Return bbox-derived placement metadata for an object."""
        obj = self.env.objects[object_id]
        qpos = self.env.sim.data.get_joint_qpos(obj.joints[0]).copy()
        quat_wxyz = qpos[3:7]
        quat_xyzw = T.convert_quat(quat_wxyz, to="xyzw")
        bbox_points = np.asarray(
            obj.get_bbox_points(trans=np.zeros(3, dtype=float), rot=quat_xyzw),
            dtype=float,
        )
        span = np.max(bbox_points, axis=0) - np.min(bbox_points, axis=0)
        try:
            xy_radius = float(obj.horizontal_radius)
        except Exception:
            xy_radius = 0.5 * float(max(span[0], span[1]))
        return {
            "quat_wxyz": quat_wxyz,
            "quat_xyzw": quat_xyzw,
            "size": span,
            "xy_radius": max(xy_radius, 0.5 * float(max(span[0], span[1]))),
        }

    def _get_fixture_reset_regions(
        self,
        target_fxtr: Fixture,
        min_size: np.ndarray | None = None,
    ) -> list[dict]:
        """Return placement regions for a fixture, filtered by minimum size."""
        regions: list[dict] = []
        try:
            all_regions = target_fxtr.get_reset_regions(env=self.env)
        except Exception:
            all_regions = None

        if isinstance(all_regions, dict):
            for region_name, region in all_regions.items():
                region_size = np.asarray(region.get("size", (0.1, 0.1)), dtype=float)
                region_height = region.get("height")
                if min_size is not None:
                    if min_size[0] > max(region_size) and min_size[1] > max(region_size):
                        continue
                    if (
                        region_height is not None
                        and len(min_size) == 3
                        and min_size[2] > float(region_height)
                    ):
                        continue
                region_dict = dict(region)
                region_dict["name"] = region_name
                regions.append(region_dict)

        if regions:
            return regions

        try:
            fallback = target_fxtr.sample_reset_region(env=self.env, min_size=min_size)
        except Exception:
            fallback = {"offset": (0.0, 0.0, 0.0), "size": (0.1, 0.1)}
        fallback_region = dict(fallback)
        fallback_region.setdefault("name", "fallback")
        return [fallback_region]

    def _iter_object_target_candidates(
        self,
        target_fxtr: Fixture,
        object_id: str,
        preferred_xy: np.ndarray | None = None,
        rng_offset: np.ndarray | None = None,
    ) -> list[np.ndarray]:
        """Generate candidate world positions for placing an object on a fixture."""
        metadata = self._get_object_placement_metadata(object_id)
        object_size = np.asarray(metadata["size"], dtype=float)
        min_size = object_size + 2.0 * _OBJECT_PLACEMENT_MARGIN
        xy_radius = float(metadata["xy_radius"])
        preferred_local = None
        if preferred_xy is not None:
            preferred_local = self._world_to_fixture_local(target_fxtr, preferred_xy)

        candidates: list[np.ndarray] = []
        seen: set[tuple[float, float, float]] = set()

        for region in self._get_fixture_reset_regions(target_fxtr, min_size=min_size):
            offset = np.asarray(region.get("offset", (0.0, 0.0, 0.0)), dtype=float)
            size = np.asarray(region.get("size", (0.1, 0.1)), dtype=float)
            usable_half = 0.5 * size - (xy_radius + _OBJECT_PLACEMENT_MARGIN)
            if np.any(usable_half < -1e-6):
                continue
            usable_half = np.maximum(usable_half, 0.0)

            local_candidates = [offset.copy()]

            if rng_offset is not None:
                jittered = offset.copy()
                jittered[:2] += np.asarray(rng_offset[:2], dtype=float)
                if np.all(np.abs(jittered[:2] - offset[:2]) <= usable_half + 1e-6):
                    local_candidates.append(jittered)

            if preferred_local is not None:
                projected = offset.copy()
                projected[0] = float(np.clip(
                    preferred_local[0],
                    offset[0] - usable_half[0],
                    offset[0] + usable_half[0],
                ))
                projected[1] = float(np.clip(
                    preferred_local[1],
                    offset[1] - usable_half[1],
                    offset[1] + usable_half[1],
                ))
                local_candidates.append(projected)

            x_values = self._sample_axis_values(float(offset[0]), float(usable_half[0]))
            y_values = self._sample_axis_values(float(offset[1]), float(usable_half[1]))
            for x in x_values:
                for y in y_values:
                    local = offset.copy()
                    local[0] = float(x)
                    local[1] = float(y)
                    local_candidates.append(local)

            for local in local_candidates:
                key = tuple(np.round(local, 4))
                if key in seen:
                    continue
                seen.add(key)
                world = self._fixture_local_to_world(target_fxtr, local)
                world[2] += 0.02
                candidates.append(world)

        if candidates:
            return candidates

        fallback = self._fixture_local_to_world(
            target_fxtr,
            np.array([0.0, 0.0, 0.0], dtype=float),
        )
        fallback[2] += 0.02
        return [fallback]

    def _build_object_collision_context(
        self,
        object_id: str,
        target_fixture_id: str,
        ignored_object_ids: set[str] | None = None,
    ) -> dict:
        """Pre-compute static scene geometry used during one placement search."""
        metadata = self._get_object_placement_metadata(object_id)
        obj = self.env.objects[object_id]
        ignored = set() if ignored_object_ids is None else set(ignored_object_ids)

        object_obstacles = []
        for other_id, other_obj in self.env.objects.items():
            if other_id == object_id or other_id in ignored:
                continue
            other_qpos = self.env.sim.data.get_joint_qpos(other_obj.joints[0]).copy()
            other_pos = other_qpos[:3]
            try:
                other_radius = float(other_obj.horizontal_radius)
            except Exception:
                other_radius = 0.0
            object_obstacles.append(
                (
                    other_obj,
                    other_pos,
                    T.convert_quat(other_qpos[3:7], to="xyzw"),
                    other_radius,
                )
            )

        fixture_obstacles = []
        for fixture_id, fixture in self._fixtures.items():
            if fixture_id == target_fixture_id:
                continue
            fixture_pos = np.asarray(fixture.pos, dtype=float)
            try:
                fixture_radius = float(fixture.horizontal_radius)
            except Exception:
                fixture_radius = 0.0
            fixture_obstacles.append((fixture, fixture_pos, fixture_radius))

        return {
            "object": obj,
            "obj_quat": metadata["quat_xyzw"],
            "obj_radius": float(metadata["xy_radius"]),
            "object_obstacles": object_obstacles,
            "fixture_obstacles": fixture_obstacles,
        }

    def _candidate_overlaps_scene(
        self,
        candidate_pos: np.ndarray,
        collision_context: dict,
    ) -> bool:
        """Return True if the candidate intersects precomputed scene geometry."""
        obj = collision_context["object"]
        obj_quat = collision_context["obj_quat"]
        obj_radius = float(collision_context["obj_radius"])

        for other_obj, other_pos, other_quat, other_radius in collision_context["object_obstacles"]:
            if np.linalg.norm(other_pos[:2] - candidate_pos[:2]) > obj_radius + other_radius + 0.30:
                continue
            try:
                if OU.objs_intersect(
                    obj,
                    candidate_pos,
                    obj_quat,
                    other_obj,
                    other_pos,
                    other_quat,
                ):
                    return True
            except Exception:
                continue

        for fixture, fixture_pos, fixture_radius in collision_context["fixture_obstacles"]:
            if np.linalg.norm(fixture_pos[:2] - candidate_pos[:2]) > obj_radius + fixture_radius + 0.30:
                continue
            try:
                if OU.objs_intersect(
                    obj,
                    candidate_pos,
                    obj_quat,
                    fixture,
                    fixture_pos,
                    None,
                ):
                    return True
            except Exception:
                continue

        return False

    def _compute_object_target_pos(
        self,
        target_fxtr: Fixture,
        rng_offset: np.ndarray | None = None,
        object_id: str | None = None,
        preferred_xy: np.ndarray | None = None,
        ignored_object_ids: set[str] | None = None,
    ) -> np.ndarray:
        """Compute a collision-aware world position on *target_fxtr*'s surface."""
        if object_id is None:
            try:
                region = target_fxtr.sample_reset_region(env=self.env)
            except Exception:
                region = {"offset": (0.0, 0.0, 0.0), "size": (0.1, 0.1)}

            offset = np.array(region["offset"], dtype=float)
            if rng_offset is not None:
                offset[0] += rng_offset[0]
                offset[1] += rng_offset[1]

            target_pos = self._fixture_local_to_world(target_fxtr, offset)
            target_pos[2] += 0.02
            return target_pos

        preferred = None if preferred_xy is None else np.asarray(preferred_xy, dtype=float)[:2]
        candidates = self._iter_object_target_candidates(
            target_fxtr,
            object_id,
            preferred_xy=preferred,
            rng_offset=rng_offset,
        )
        collision_context = self._build_object_collision_context(
            object_id,
            target_fxtr.name,
            ignored_object_ids=ignored_object_ids,
        )

        target_xy = preferred if preferred is not None else np.asarray(target_fxtr.pos[:2], dtype=float)
        valid: list[tuple[float, np.ndarray]] = []
        for candidate in candidates:
            if not self._validate_object_on_fixture(candidate, target_fxtr.name):
                continue
            if self._candidate_overlaps_scene(candidate, collision_context):
                continue
            score = float(np.linalg.norm(candidate[:2] - target_xy))
            valid.append((score, candidate.copy()))

        if valid:
            return min(valid, key=lambda item: item[0])[1]

        return candidates[0].copy()

    def _validate_object_on_fixture(
        self, obj_pos: np.ndarray, fixture_id: str,
    ) -> bool:
        """Return True if *obj_pos* is geometrically inside *fixture_id*."""
        fxtr = self._fixtures.get(fixture_id)
        if fxtr is None:
            return False
        try:
            return OU.point_in_fixture(obj_pos, fxtr, only_2d=True)
        except Exception:
            # Fallback: accept if within 0.3 m of the fixture centre
            return float(np.linalg.norm(obj_pos[:2] - fxtr.pos[:2])) < 0.3

    def _find_contained_objects(self, container_id: str) -> list[str]:
        """Find objects physically inside/on top of *container_id*."""
        contained = []
        body_id = self.env.obj_body_id.get(container_id)
        if body_id is None:
            return contained
        container_pos = self.env.sim.data.body_xpos[body_id].copy()
        container_obj = self.env.objects[container_id]
        radius = getattr(container_obj, "horizontal_radius", 0.10) * 1.2
        for other_id in self.env.objects:
            if other_id == container_id:
                continue
            other_body_id = self.env.obj_body_id.get(other_id)
            if other_body_id is None:
                continue
            other_pos = self.env.sim.data.body_xpos[other_body_id].copy()
            xy_dist = float(np.linalg.norm(other_pos[:2] - container_pos[:2]))
            z_diff = other_pos[2] - container_pos[2]
            if xy_dist < radius and -0.05 < z_diff < 0.20:
                contained.append(other_id)
        return contained

    def move_object(
        self,
        object_id: str,
        to_fixture: str,
        target_pos: np.ndarray | None = None,
        preferred_xy: np.ndarray | None = None,
    ):
        """Teleport an object to a fixture's surface.

        If the object contains other objects (e.g. a bowl with slices),
        those are moved along with it.

        Placement is collision-aware: candidates are sampled from fixture
        reset regions, filtered against existing objects / nearby fixtures,
        and ranked by distance to ``preferred_xy`` when provided.
        """
        if object_id not in self.env.objects:
            raise ValueError(
                f"Unknown object '{object_id}'. "
                f"Available: {list(self.env.objects.keys())}"
            )
        if to_fixture not in self._fixtures:
            raise ValueError(
                f"Unknown fixture '{to_fixture}'. "
                f"Available: {list(self._fixtures.keys())}"
            )

        target_fxtr = self._fixtures[to_fixture]
        obj = self.env.objects[object_id]
        current_qpos = self.env.sim.data.get_joint_qpos(obj.joints[0])
        current_quat = current_qpos[3:7]
        old_pos = current_qpos[:3].copy()

        # Snapshot contained objects before moving
        contained = self._find_contained_objects(object_id)
        if target_pos is None:
            target_pos = self._compute_object_target_pos(
                target_fxtr,
                object_id=object_id,
                preferred_xy=preferred_xy,
                ignored_object_ids=set(contained),
            )

        self.env.sim.data.set_joint_qpos(
            obj.joints[0],
            np.concatenate([target_pos, current_quat]),
        )

        # Move contained objects by the same delta
        if contained:
            delta = target_pos - old_pos
            for child_id in contained:
                child_obj = self.env.objects[child_id]
                child_qpos = self.env.sim.data.get_joint_qpos(child_obj.joints[0])
                child_qpos[:3] += delta
                self.env.sim.data.set_joint_qpos(child_obj.joints[0], child_qpos)

        self.env.sim.forward()

        self._set_object_location(object_id, to_fixture)
        for child_id in contained:
            self._set_object_location(child_id, to_fixture)

    def interact(self, fixture_id: str, action: str):
        """Change a fixture's state: open, close, turn_on, turn_off."""
        if fixture_id not in self._fixtures:
            raise ValueError(
                f"Unknown fixture '{fixture_id}'. "
                f"Available: {list(self._fixtures.keys())}"
            )

        fxtr = self._fixtures[fixture_id]

        if action == "open":
            fxtr.open_door(env=self.env)
        elif action == "close":
            fxtr.close_door(env=self.env)
        elif action == "turn_on":
            if hasattr(fxtr, "_joint_infos"):
                non_door = [
                    j for j in fxtr._joint_infos
                    if "door" not in j.lower() and "drawer" not in j.lower()
                ]
                if non_door:
                    fxtr.set_joint_state(
                        min=0.9, max=1.0, env=self.env, joint_names=non_door
                    )
        elif action == "turn_off":
            if hasattr(fxtr, "_joint_infos"):
                non_door = [
                    j for j in fxtr._joint_infos
                    if "door" not in j.lower() and "drawer" not in j.lower()
                ]
                if non_door:
                    fxtr.set_joint_state(
                        min=0.0, max=0.0, env=self.env, joint_names=non_door
                    )
        else:
            raise ValueError(
                f"Unknown action '{action}'. "
                f"Supported: open, close, turn_on, turn_off"
            )

        self.env.sim.forward()

    # ------------------------------------------------------------------
    # Trajectory execution
    # ------------------------------------------------------------------

    def run(self, trajectory: dict | list) -> TrajectoryResult:
        """
        Execute a trajectory and capture visual observations.

        For each step:
          1. Teleport the acting robot near the target fixture
          2. Render before images
          3. Execute the primitive (move_object / interact / communicate)
          4. Render after images
        """
        if isinstance(trajectory, dict):
            steps = trajectory.get("steps", [])
        else:
            steps = trajectory

        scene = self.get_scene_description()

        # Do a zero-action step to fully initialize the render context
        # (required for free camera rendering, same as two_robot_video_sample)
        low, _ = self.env.action_spec
        self.env.step(np.zeros_like(low))

        initial_obs = self.render()
        step_results = []

        for i, step in enumerate(steps):
            agent_id = step.get("agent_id", "agent_0")
            action = step.get("action") or step.get("tool_name", "")
            args = step.get("args") or step.get("tool_args", {})
            move_target_pos = None

            # Parse robot index from agent_id
            robot_idx = int(agent_id.replace("agent_", ""))

            # Teleport robot near the fixture it will interact with.
            # For move_object, pre-compute the drop position so the robot
            # stands near where the object will actually land.
            target_fixture = self._get_step_fixture(action, args)
            if target_fixture is not None:
                ref_override = None
                if action == "move_object" and target_fixture in self._fixtures:
                    move_target_pos = self._compute_object_target_pos(
                        self._fixtures[target_fixture],
                        object_id=args["object"],
                    )
                    ref_override = move_target_pos[:2]
                self._move_robot_near_fixture(
                    robot_idx, target_fixture, ref_pos_override=ref_override,
                )

            # Render before
            before = self.render()

            # Execute primitive
            if action == "move_object":
                self.move_object(
                    object_id=args["object"],
                    to_fixture=args["to"],
                    target_pos=move_target_pos,
                )
            elif action == "interact":
                fixture_id = args.get("fixture") or args.get("fixture_id")
                act = args.get("action")
                if fixture_id is None or act is None:
                    raise ValueError(
                        f"Step {i}: 'interact' requires 'fixture' and 'action' in args"
                    )
                self.interact(fixture_id, act)
            elif action == "navigate":
                # Robot was already teleported above via _move_robot_near_fixture
                pass
            elif action == "move_away":
                to_agent = args.get("to_agent", "")
                obj_id = args.get("object", "")
                to_robot = int(to_agent.replace("agent_", "")) if to_agent else (1 - robot_idx)
                self.hand_off_object(robot_idx, obj_id, to_robot)
            elif action == "give_space":
                fixture_id = args.get("fixture") or args.get("fixture_id", "")
                self.give_space(robot_idx, fixture_id)
            elif action == "communicate":
                pass  # no sim change
            elif action == "wait":
                pass  # no sim change, just observe
            else:
                raise ValueError(
                    f"Step {i}: unknown action '{action}'. "
                    f"Expected: move_object, interact, navigate, give_space, communicate, wait"
                )

            # Render after
            after = self.render()

            step_results.append(StepResult(
                step_index=i,
                agent_id=agent_id,
                action=action,
                args=step.get("args") or step.get("tool_args", {}),
                before=before,
                after=after,
            ))

        return TrajectoryResult(
            initial_obs=initial_obs,
            steps=step_results,
            scene=scene,
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self):
        """Close the environment."""
        self.env.close()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
