"""
Executable simulator tool layer for symbolic kitchen manipulation.

This is a pragmatic executor over the existing ``TrajectoryRunner`` substrate:
robot navigation uses the runner's working-pose placement, object transport is
represented by a held-object cache plus explicit object pose updates, and
fixture manipulation is performed by direct simulator state changes.

These tools are "real" in the sense that they mutate a live RoboCasa
environment. They are not low-level controller policies.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any
from copy import deepcopy

import imageio
import numpy as np
import robosuite.utils.transform_utils as T

from robocasa.models.fixtures.coffee_machine import CoffeeMachine
from robocasa.models.fixtures.electric_kettle import ElectricKettle
from robocasa.models.fixtures.microwave import Microwave
from robocasa.models.fixtures.sink import Sink
from robocasa.models.fixtures.toaster import Toaster
import robocasa.utils.object_utils as OU
from robocasa.models.fixtures import FixtureType
from robocasa.models.fixtures.fixture_utils import fixture_is_type
from robocasa.utils.placement import (
    MAX_FRONT_WORKING_LATERAL_OFFSET,
    get_front_alignment_metrics,
    get_fixture_aabb,
)
from robocasa.utils.sim_tool_specs import SIM_TOOL_SPEC_BY_NAME
from robocasa.utils.trajectory_runner import TrajectoryRunner

# Large fixtures with a door/opening that MUST be approached from the front.
# For these fixtures, require_front=True ensures the robot lines up with the
# opening rather than standing on a blocked side.
_REQUIRE_FRONT_TYPES = {
    FixtureType.CABINET,
    FixtureType.CABINET_SINGLE_DOOR,
    FixtureType.CABINET_DOUBLE_DOOR,
    FixtureType.CABINET_WITH_DOOR,
    FixtureType.FRIDGE,
    FixtureType.MICROWAVE,
    FixtureType.OVEN,
    FixtureType.DISHWASHER,
    FixtureType.TOP_DRAWER,
    FixtureType.DRAWER,
}

# Small countertop appliances: robot should approach the fixture center (not
# an object inside it), but does NOT need require_front — any face is fine.
# Their AABBs are tiny, so front-face-only sampling often lands the robot in
# a bad spot on the counter.
_COUNTERTOP_APPLIANCE_TYPES = {
    FixtureType.TOASTER,
    FixtureType.TOASTER_OVEN,
    FixtureType.COFFEE_MACHINE,
    FixtureType.BLENDER,
    FixtureType.STAND_MIXER,
    FixtureType.ELECTRIC_KETTLE,
}

# Union: all fixtures where the robot targets the fixture center, not an
# object's position inside it.
_APPROACH_CENTER_TYPES = _REQUIRE_FRONT_TYPES | _COUNTERTOP_APPLIANCE_TYPES

_FRONT_READY_MIN_GAP = 0.05
_FRONT_READY_MAX_GAP = 0.75
_FRONT_READY_MAX_CENTER_DISTANCE = 1.0


def _is_approach_center(fixture) -> bool:
    """Return True if the robot should approach the fixture center, not an object inside it."""
    return any(fixture_is_type(fixture, ft) for ft in _APPROACH_CENTER_TYPES)


def _require_front(fixture) -> bool:
    """Return True if the robot must approach from the fixture's front face."""
    return any(fixture_is_type(fixture, ft) for ft in _REQUIRE_FRONT_TYPES)


@dataclass
class ToolResult:
    tool_name: str
    success: bool
    details: dict[str, Any]


class SimToolExecutor:
    """Dispatch simulator tool calls against a live RoboCasa environment."""

    _HELD_Z_OFFSET = 0.0
    _DEMO_TASK_BY_NAME = {
        "cooperative_hotdog_setup": "HotDogSetup",
        "sandwich_station": "PrepareSandwichStation",
    }

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
        placement: str = "grid",
        cell_size: float = 0.05,
        align_to_wall: bool = True,
        standoff: float = 0.40,
        sample_spacing: float = 0.08,
        robot_radius: float = 0.18,
        robot_spawn: str = "trajectory",
        full_scene_view: bool = True,
    ):
        os.environ["MUJOCO_GL"] = gl_backend
        self.runner = TrajectoryRunner(
            task_name=task_name,
            robots=robots,
            layout=layout,
            style=style,
            seed=seed,
            camera_names=camera_names,
            render_width=render_width,
            render_height=render_height,
            gl_backend=gl_backend,
            placement=placement,
            cell_size=cell_size,
            align_to_wall=align_to_wall,
            standoff=standoff,
            sample_spacing=sample_spacing,
            robot_radius=robot_radius,
            full_scene_view=full_scene_view,
        )
        self.env = self.runner.env
        self._held_objects: dict[int, str] = {}
        self._clean_map_labels: bool = True
        self._robot_spawn: str = robot_spawn

        # Place all robots at the task's init_robot_base_ref (ground truth
        # starting position).  The env spawns robots at (10,10,0) by default;
        # this moves them to the fixture the task designates as the start.
        # In "trajectory" mode this is still called so the initial_map.png
        # shows sim ground truth; robots are repositioned later in
        # load_initial_state when trajectory agent locations are applied.
        self._place_robots_at_spawn()

    def _place_robots_at_spawn(self):
        """Move all robots to init_robot_base_ref — the task's ground truth spawn."""
        scene = self.get_scene_description()
        spawn_fixture = scene.get("init_robot_base_ref")
        if not spawn_fixture or spawn_fixture not in self.runner._fixtures:
            return
        for i in range(self.runner._num_robots):
            self.runner._move_robot_near_fixture(i, spawn_fixture)
        # Belt-and-suspenders: verify every robot ended up inside kitchen
        for i in range(self.runner._num_robots):
            self.runner._rescue_robot_to_kitchen(i)

    # ------------------------------------------------------------------
    # Scene / state helpers
    # ------------------------------------------------------------------

    def close(self):
        self.runner.close()

    def get_scene_description(self) -> dict[str, Any]:
        return self.runner.get_scene_description()

    def render(self) -> dict[str, np.ndarray]:
        return self.runner.render()

    def save_placement_map(
        self, output_dir: str | Path, prefix: str = "placement",
        clean_labels: bool = True,
    ) -> Path:
        """Render the 2D placement map and save it to *output_dir*.

        Uses the grid view for grid mode, continuous view for continuous mode,
        or side-by-side if both are available.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return self._save_map_image(
            output_dir / f"{prefix}_map.png", clean_labels=clean_labels,
        )

    def _save_map_image(
        self, image_path: str | Path, clean_labels: bool = True,
    ) -> Path:
        """Render the placement map and save it to an explicit output path.

        *clean_labels*: when True (default), fixture labels are shortened
        (strip ``_group``, ``_main``, dedupe repeated segments).
        Set False to show full raw fixture ids.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from robocasa.utils.placement_map import draw_grid_map, draw_continuous_map

        path = Path(image_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = self.runner._placement_mode
        if mode == "grid":
            fig, ax = plt.subplots(1, 1, figsize=(20, 16))
            draw_grid_map(ax, self.runner, clean_labels=clean_labels)
        elif mode == "continuous":
            fig, ax = plt.subplots(1, 1, figsize=(20, 16))
            draw_continuous_map(ax, self.runner, clean_labels=clean_labels)
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(36, 16))
            draw_grid_map(ax1, self.runner, clean_labels=clean_labels)
            draw_continuous_map(ax2, self.runner, clean_labels=clean_labels)

        fig.tight_layout()
        fig.savefig(path, dpi=300)
        plt.close(fig)
        return path

    def save_scene_frames(
        self, output_dir: str | Path, prefix: str = "initial"
    ) -> dict[str, Path]:
        """Render the current scene and save one image per camera."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        frames = self.render()
        saved = {}
        for camera_name, image in frames.items():
            camera_dir = output_dir / camera_name
            camera_dir.mkdir(parents=True, exist_ok=True)
            path = camera_dir / f"{prefix}.jpg"
            imageio.imwrite(path, image, quality=85)
            saved[camera_name] = path
        return saved

    def _parse_agent_idx(self, agent_id: str | int) -> int:
        if isinstance(agent_id, int):
            return agent_id
        agent_str = str(agent_id)
        if agent_str.startswith("agent_"):
            return int(agent_str.replace("agent_", ""))
        return int(agent_str)

    def _render_camera(self, camera_name: str) -> np.ndarray:
        if camera_name == "room_view":
            return self.runner._render_room_view()
        if camera_name == "top_view":
            return self.runner._render_top_view()
        return self.env.sim.render(
            height=self.runner.render_height,
            width=self.runner.render_width,
            camera_name=camera_name,
        )[::-1]

    def _save_image(
        self, image: np.ndarray, image_path: str | Path, *, is_map: bool = False
    ) -> Path:
        path = Path(image_path)
        # Use JPEG for camera renders (much smaller), keep PNG for maps
        if not is_map and path.suffix.lower() == ".png":
            path = path.with_suffix(".jpg")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() in {".jpg", ".jpeg"}:
            imageio.imwrite(path, image, quality=85)
        else:
            imageio.imwrite(path, image)
        return path

    def _camera_name_for_agent_view(
        self, agent_id: str | int, view: str
    ) -> tuple[int, str]:
        robot_idx = self._parse_agent_idx(agent_id)
        view_name = str(view).strip().lower()
        view_aliases = {
            "wrist": "eye_in_hand",
            "eye_in_hand": "eye_in_hand",
            "agentview_center": "agentview_center",
            "agentview_left": "agentview_left",
            "agentview_right": "agentview_right",
            "robotview": "robotview",
        }
        suffix = view_aliases.get(view_name, view_name)
        if suffix.startswith("robot"):
            return robot_idx, suffix
        return robot_idx, f"robot{robot_idx}_{suffix}"

    def _set_fixture_machine_state(self, fixture_id: str, started: bool):
        fixture = self._require_fixture(fixture_id)
        started = bool(started)
        if isinstance(fixture, (CoffeeMachine, Microwave)):
            fixture._turned_on = started
        elif isinstance(fixture, ElectricKettle):
            fixture.set_power_state(self.env, power_on=started)
        else:
            return
        self.env.sim.forward()

    def load_initial_state(
        self, initial_state: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Apply a normalized initial state to the live simulator."""
        if not initial_state:
            return {"loaded": False}

        fixtures = initial_state.get("fixtures", {})
        objects = initial_state.get("objects", {})
        agents = initial_state.get("agents", {})
        machine_state = initial_state.get("machine_state", {})
        held_assignments: dict[int, str] = {}

        self._held_objects.clear()

        # Skip fixture part states (open/close) from the trajectory's
        # initial_state.  The sim's _setup_scene already set the correct
        # fixture states (e.g. opening the cabinet so the mug is accessible).
        # The trajectory's initial_state reflects the LLM planner's
        # assumptions, which may conflict with the sim (e.g. closing a
        # cabinet that must start open).  The trajectory's own steps will
        # open/close fixtures as needed.

        for fixture_id, machine_cfg in machine_state.items():
            if "started" in machine_cfg:
                self._set_fixture_machine_state(
                    fixture_id, bool(machine_cfg["started"])
                )

        held_object_ids = set()
        for agent_id, agent_state in agents.items():
            held_object = agent_state.get("held_object")
            if held_object is None:
                continue
            robot_idx = self._parse_agent_idx(agent_id)
            held_assignments[robot_idx] = held_object
            held_object_ids.add(held_object)

        # Object placements from the scene description — where the sim
        # originally placed each object.  Used to skip redundant moves.
        scene = self.get_scene_description()
        sim_object_placements = scene.get("object_placements", {})

        for object_id, object_state in objects.items():
            if object_id in held_object_ids:
                continue
            location = object_state.get("location")
            if isinstance(location, str):
                # Location is another object (e.g. slices inside a bowl) —
                # the sim already placed them correctly, skip.
                if location in objects:
                    continue
                # Skip if the sim already placed this object at the target
                # fixture — re-placing would resample the position and may
                # put the object somewhere unexpected (e.g. wrong shelf).
                if sim_object_placements.get(object_id) == location:
                    continue
                self.runner.move_object(object_id, location)

        if self._robot_spawn == "trajectory":
            # Navigate each robot to the trajectory's stated initial location.
            # These are the LLM planner's logical assumptions (not sim ground
            # truth), but the user explicitly requested trajectory-based spawn.
            for agent_id, agent_state in agents.items():
                location = agent_state.get("location")
                if isinstance(location, str) and location in self.runner._fixtures:
                    robot_idx = self._parse_agent_idx(agent_id)
                    self.runner._move_robot_near_fixture(robot_idx, location)
            # Final safety: verify every robot is inside kitchen
            for i in range(self.runner._num_robots):
                self.runner._rescue_robot_to_kitchen(i)
        # else: "sim" mode — robots already placed at init_robot_base_ref
        # by _place_robots_at_spawn() during __init__.

        for robot_idx, object_id in held_assignments.items():
            self._require_object(object_id)
            self._held_objects[robot_idx] = object_id
            self._sync_held_object(robot_idx)
            agent_key = f"agent_{robot_idx}"
            agent_state = agents.get(agent_key, {})
            location = agent_state.get("location")
            if isinstance(location, str) and location in self.runner._fixtures:
                self.runner._set_object_location(object_id, location)

        return {
            "loaded": True,
            "agents": sorted(agents.keys()),
            "objects": sorted(objects.keys()),
            "fixtures": sorted(fixtures.keys()),
            "held_objects": {
                f"robot{robot_idx}": object_id
                for robot_idx, object_id in sorted(self._held_objects.items())
            },
        }

    def run_tool_plan(
        self,
        tool_calls: list[dict[str, Any]],
        output_dir: str | Path,
        fps: int = 2,
        skip_videos: bool = False,
    ) -> dict[str, Any]:
        """
        Execute a list of tool calls, letting ``get_image`` steps produce images.

        Images are saved only when the plan contains ``get_image`` tool calls
        with ``image_paths``.  When ``skip_videos`` is False, per-camera MP4
        videos are also generated by rendering all cameras before and after
        every action step (giving a complete visual record of the trajectory).
        """
        tool_calls = self.ground_plan_template(tool_calls)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        camera_names = list(self.render().keys())

        # Video writers — render before/after for every action step
        writers: dict[str, Any] = {}
        if not skip_videos:
            writers = {
                cam: imageio.get_writer(str(output_dir / f"{cam}.mp4"), fps=fps)
                for cam in camera_names
            }

        def _write_video_frame():
            if writers:
                frames = self.render()
                for cam, image in frames.items():
                    if cam in writers:
                        writers[cam].append_data(image)

        metadata = {
            "task": self.get_scene_description().get("task"),
            "steps": [],
            "cameras": camera_names,
        }

        try:
            # Initial frame
            _write_video_frame()

            for step_idx, tool_call in enumerate(tool_calls):
                tool_name = tool_call["tool"]
                robot_idx = tool_call.get("robot_idx", 0)
                args = tool_call.get("args", {})

                # Before frame (for action steps only — get_image doesn't change state)
                is_action = tool_name != "get_image"
                if is_action:
                    _write_video_frame()

                result = self.execute(tool_name, robot_idx=robot_idx, **args)

                # After frame
                if is_action:
                    _write_video_frame()

                # Record robot positions for diagnostics
                robot_positions = {}
                for ri in range(self.runner._num_robots):
                    rp = self.runner._get_robot_position(ri)
                    robot_positions[f"robot{ri}"] = [
                        round(float(rp[0]), 3),
                        round(float(rp[1]), 3),
                        round(float(rp[2]), 3),
                    ]

                step_meta = {
                    "step_index": step_idx,
                    "tool": tool_name,
                    "robot_idx": robot_idx,
                    "args": args,
                    "success": result.success,
                    "details": result.details,
                    "robot_positions": robot_positions,
                }
                # Propagate image paths produced by get_image
                if result.details.get("image_paths"):
                    step_meta["image_paths"] = result.details["image_paths"]

                metadata["steps"].append(step_meta)
        finally:
            for writer in writers.values():
                writer.close()

        with open(output_dir / "plan.json", "w") as f:
            json.dump(tool_calls, f, indent=2)
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        return metadata

    def _require_fixture(self, fixture_id: str):
        if fixture_id not in self.runner._fixtures:
            raise ValueError(f"Unknown fixture_id: {fixture_id!r}")
        return self.runner._fixtures[fixture_id]

    def _require_object(self, object_id: str):
        if object_id not in self.env.objects:
            raise ValueError(f"Unknown object_id: {object_id!r}")
        return self.env.objects[object_id]

    def _get_object_pose(self, object_id: str) -> tuple[np.ndarray, np.ndarray]:
        obj = self._require_object(object_id)
        qpos = self.env.sim.data.get_joint_qpos(obj.joints[0]).copy()
        return qpos[:3], qpos[3:7]

    def _set_object_pose(
        self,
        object_id: str,
        pos: np.ndarray | list[float],
        quat: np.ndarray | list[float] | None = None,
    ):
        obj = self._require_object(object_id)
        old_pos, current_quat = self._get_object_pose(object_id)
        contained = self._find_contained_objects(object_id)
        quat_arr = current_quat if quat is None else np.asarray(quat, dtype=float)
        target_pos = np.asarray(pos, dtype=float)
        self.env.sim.data.set_joint_qpos(
            obj.joints[0],
            np.concatenate([target_pos, quat_arr]),
        )
        self.env.sim.forward()

        if contained:
            delta = target_pos - old_pos
            if np.linalg.norm(delta) > 1e-9:
                for child_id in contained:
                    child_pos, child_quat = self._get_object_pose(child_id)
                    self._set_object_pose(child_id, child_pos + delta, child_quat)

    def _get_robot_eef_pos(self, robot_idx: int) -> np.ndarray:
        site_id = self.env.robots[robot_idx].eef_site_id["right"]
        return self.env.sim.data.site_xpos[site_id].copy()

    def _find_contained_objects(self, container_id: str) -> list[str]:
        """Find objects physically inside/on top of *container_id*.

        Only returns objects that are smaller than (or equal to)
        *container_id* so that picking up an item inside a bowl/tray
        does NOT drag the bowl/tray along with it.
        """
        contained = []
        container_pos, _ = self._get_object_pose(container_id)
        container_obj = self.env.objects[container_id]
        container_radius = getattr(container_obj, "horizontal_radius", 0.10)
        search_radius = container_radius * 1.2
        for other_id in self.env.objects:
            if other_id == container_id:
                continue
            other_obj = self.env.objects[other_id]
            other_radius = getattr(other_obj, "horizontal_radius", 0.10)
            # Skip objects that are larger — they are parents, not children.
            if other_radius > container_radius:
                continue
            other_pos, _ = self._get_object_pose(other_id)
            xy_dist = float(np.linalg.norm(other_pos[:2] - container_pos[:2]))
            z_diff = other_pos[2] - container_pos[2]
            # Object must be close horizontally and roughly at the same height
            # as the container.  Allows slight negative z_diff for objects
            # inside concave containers (slices resting at the bottom of a bowl).
            if xy_dist < search_radius and -0.05 <= z_diff < 0.20:
                contained.append(other_id)
        return contained

    def _sync_held_object(self, robot_idx: int):
        object_id = self._held_objects.get(robot_idx)
        if object_id is None:
            return

        eef_pos = self._get_robot_eef_pos(robot_idx)
        held_pos = eef_pos.copy()
        held_pos[2] += self._HELD_Z_OFFSET
        print(
            f"[_sync_held] robot{robot_idx} holds {object_id}: "
            f"eef=({eef_pos[0]:.3f}, {eef_pos[1]:.3f}, {eef_pos[2]:.3f}) "
            f"-> held=({held_pos[0]:.3f}, {held_pos[1]:.3f}, {held_pos[2]:.3f})"
        )
        # _set_object_pose already moves contained objects (e.g. slices
        # inside a bowl) by the same delta — no extra handling needed.
        self._set_object_pose(object_id, held_pos)

    def _held_by_robot(self, object_id: str) -> int | None:
        for robot_idx, held_object in self._held_objects.items():
            if held_object == object_id:
                return robot_idx
        return None

    def _place_on_object_center(self, object_id: str, support_object_id: str):
        support_obj = self._require_object(support_object_id)
        support_body_id = self.env.obj_body_id[support_object_id]
        support_pos = self.env.sim.data.body_xpos[support_body_id].copy()
        support_quat_xyzw = T.convert_quat(
            self.env.sim.data.body_xquat[support_body_id].copy(),
            to="xyzw",
        )
        support_points = support_obj.get_bbox_points(
            trans=support_pos,
            rot=support_quat_xyzw,
        )
        support_top_z = max(point[2] for point in support_points)

        obj = self._require_object(object_id)
        obj_qpos = self.env.sim.data.get_joint_qpos(obj.joints[0]).copy()
        obj_quat_wxyz = obj_qpos[3:7]
        obj_quat_xyzw = T.convert_quat(obj_quat_wxyz, to="xyzw")
        obj_points_at_origin = obj.get_bbox_points(
            trans=np.zeros(3, dtype=float),
            rot=obj_quat_xyzw,
        )
        obj_bottom_z = min(point[2] for point in obj_points_at_origin)

        target_pos = support_pos.copy()
        target_pos[2] = support_top_z - obj_bottom_z + 0.01
        self._set_object_pose(object_id, target_pos, obj_quat_wxyz)

    def _set_named_joint(self, fixture, joint_name: str, value: float):
        fixture.set_joint_state(
            min=value,
            max=value,
            env=self.env,
            joint_names=[joint_name],
        )
        self.env.sim.forward()

    def _resolve_joint_name(self, fixture, token: str) -> str:
        token = token.lower()

        if hasattr(fixture, "_joint_names") and token in fixture._joint_names:
            return fixture._joint_names[token]

        if hasattr(fixture, "_joint_infos"):
            for joint_name in fixture._joint_infos:
                if token in joint_name.lower():
                    return joint_name

        raise ValueError(f"Unknown part/control {token!r} for fixture {fixture.name!r}")

    def _get_scene_object_location(self, object_id: str) -> str | None:
        cached_location = self.runner._object_locations.get(object_id)
        if (
            isinstance(cached_location, str)
            and cached_location in self.runner._fixtures
        ):
            return cached_location

        scene = self.get_scene_description()
        object_info = scene.get("objects", {}).get(object_id, {})
        location = object_info.get("location")
        if isinstance(location, str) and location in scene.get("fixtures", {}):
            return location

        try:
            obj_pos, _ = self._get_object_pose(object_id)
        except Exception:
            return None
        inferred = self.runner._find_object_fixture(obj_pos)
        if inferred in self.runner._fixtures:
            return inferred
        return None

    def _get_task_class_name(self) -> str:
        env = self.env
        while hasattr(env, "env"):
            env = env.env
        return env.__class__.__name__

    def _find_fixture_by_type(self, fixture_types: set[str]) -> str:
        scene = self.get_scene_description()
        for fixture_id, fixture_info in scene.get("fixtures", {}).items():
            if fixture_info.get("fixture_type") in fixture_types:
                return fixture_id
        raise ValueError(f"Could not find fixture with type in {sorted(fixture_types)}")

    # ------------------------------------------------------------------
    # Spatial relation helpers
    # ------------------------------------------------------------------

    def _find_placeable_surface_near_fixture(self, reference_fixture_id: str) -> str:
        """Find the nearest placeable surface to a reference fixture.

        If the reference fixture is itself placeable (e.g., a counter), returns it.
        Otherwise searches nearby fixtures and falls back to the closest counter.

        This indirection is needed because the reference in a spatial relation
        is often non-placeable (e.g., "near toaster_oven" — the toaster isn't a
        surface, but the counter it sits on is).
        """
        scene = self.get_scene_description()
        fixtures = scene.get("fixtures", {})
        ref_info = fixtures.get(reference_fixture_id)
        if ref_info is None:
            raise ValueError(f"Unknown fixture: {reference_fixture_id!r}")

        if ref_info.get("can_place_objects", False):
            return reference_fixture_id

        # Use parent_fixture (containment-based) when available — this is
        # the counter the fixture actually sits on.
        parent_id = ref_info.get("parent_fixture")
        if parent_id and parent_id in fixtures:
            parent_info = fixtures[parent_id]
            if parent_info.get("can_place_objects", False):
                return parent_id

        # Fallback: nearest placeable surface by center distance.
        ref_pos = np.asarray(ref_info["position"][:2], dtype=float)
        best_id, best_dist = None, float("inf")
        for fixture_id, info in fixtures.items():
            if not info.get("can_place_objects", False):
                continue
            dist = float(
                np.linalg.norm(np.asarray(info["position"][:2], dtype=float) - ref_pos)
            )
            if dist < best_dist:
                best_dist = dist
                best_id = fixture_id

        if best_id is None:
            raise ValueError(
                f"No placeable surface found near {reference_fixture_id!r}"
            )
        return best_id

    def _find_nearest_fixture_for_object(
        self,
        object_id: str,
        preferred_fixture_types: list[str] | set[str] | tuple[str, ...] | None = None,
        require_placeable: bool = False,
    ) -> str:
        object_pos, _ = self._get_object_pose(object_id)
        scene = self.get_scene_description()
        fixtures = scene.get("fixtures", {})
        preferred_fixture_types = self._normalize_preferred_fixture_types(
            preferred_fixture_types
        )

        def candidate_ids() -> list[str]:
            candidates = []
            for fixture_id, fixture_info in fixtures.items():
                if (
                    preferred_fixture_types is not None
                    and fixture_info.get("fixture_type") not in preferred_fixture_types
                ):
                    continue
                if require_placeable and not fixture_info.get(
                    "can_place_objects", False
                ):
                    continue
                candidates.append(fixture_id)
            return candidates

        candidates = candidate_ids()
        if not candidates:
            raise ValueError(
                f"Could not find candidate fixtures for {object_id!r} "
                f"with types {sorted(preferred_fixture_types or set())}"
            )

        return min(
            candidates,
            key=lambda fixture_id: float(
                np.linalg.norm(
                    object_pos[:2]
                    - np.asarray(fixtures[fixture_id]["position"][:2], dtype=float)
                )
            ),
        )

    def _resolve_object_anchor_fixture(
        self,
        object_id: str,
        preferred_fixture_types: list[str] | set[str] | tuple[str, ...] | None = None,
        require_placeable: bool = False,
    ) -> str:
        scene = self.get_scene_description()
        fixtures = scene.get("fixtures", {})
        object_pos, _ = self._get_object_pose(object_id)
        preferred_fixture_types = self._normalize_preferred_fixture_types(
            preferred_fixture_types
        )

        def matches(fixture_id: str) -> bool:
            fixture_info = fixtures.get(fixture_id)
            if fixture_info is None:
                return False
            if (
                preferred_fixture_types is not None
                and fixture_info.get("fixture_type") not in preferred_fixture_types
            ):
                return False
            if require_placeable and not fixture_info.get("can_place_objects", False):
                return False
            return True

        explicit_location = self._get_scene_object_location(object_id)
        if explicit_location is not None and matches(explicit_location):
            return explicit_location

        candidate_ids = []
        if explicit_location is not None:
            candidate_ids.extend(
                fixtures.get(explicit_location, {}).get("nearby_fixtures", [])
            )
        candidate_ids.extend(fixtures.keys())

        deduped_candidate_ids = []
        seen = set()
        for fixture_id in candidate_ids:
            if fixture_id in seen or not matches(fixture_id):
                continue
            seen.add(fixture_id)
            deduped_candidate_ids.append(fixture_id)

        if not deduped_candidate_ids:
            return self._find_nearest_fixture_for_object(
                object_id,
                preferred_fixture_types=preferred_fixture_types,
                require_placeable=require_placeable,
            )

        def candidate_score(fixture_id: str) -> tuple[int, int, float]:
            fixture = self.runner._fixtures.get(fixture_id)
            contains_object = False
            if fixture is not None:
                try:
                    contains_object = bool(
                        OU.point_in_fixture(object_pos, fixture, only_2d=True)
                    )
                except Exception:
                    contains_object = False

            fixture_type = fixtures[fixture_id].get("fixture_type")
            type_priority = self._get_fixture_type_priority(
                fixture_type,
                preferred_fixture_types,
            )
            distance = float(
                np.linalg.norm(
                    object_pos[:2]
                    - np.asarray(fixtures[fixture_id]["position"][:2], dtype=float)
                )
            )
            return (
                0 if contains_object else 1,
                type_priority,
                distance,
            )

        return min(deduped_candidate_ids, key=candidate_score)

    def _normalize_preferred_fixture_types(
        self,
        preferred_fixture_types: list[str] | set[str] | tuple[str, ...] | None,
    ) -> list[str] | None:
        if preferred_fixture_types is None:
            return None
        if isinstance(preferred_fixture_types, list):
            return preferred_fixture_types
        return list(preferred_fixture_types)

    def _get_fixture_type_priority(
        self,
        fixture_type: str | None,
        preferred_fixture_types: list[str] | None,
    ) -> int:
        if preferred_fixture_types is None or fixture_type is None:
            return 0
        try:
            return preferred_fixture_types.index(fixture_type)
        except ValueError:
            return len(preferred_fixture_types)

    def _infer_source_fixture(
        self,
        object_id: str,
        preferred_fixture_types: list[str] | set[str] | tuple[str, ...] | None = None,
    ) -> str:
        return self._resolve_object_anchor_fixture(
            object_id,
            preferred_fixture_types=preferred_fixture_types,
        )

    def _move_robot_near_object_anchor(
        self,
        robot_idx: int,
        object_id: str,
        preferred_fixture_types: list[str] | set[str] | tuple[str, ...] | None = None,
        require_placeable: bool = False,
    ) -> str:
        fixture_id = self._resolve_object_anchor_fixture(
            object_id,
            preferred_fixture_types=preferred_fixture_types,
            require_placeable=require_placeable,
        )
        fixture = self.runner._fixtures[fixture_id]
        if _is_approach_center(fixture):
            self.runner._move_robot_near_fixture(
                robot_idx,
                fixture_id,
                require_front=True,
            )
        else:
            self.runner._move_robot_near_fixture(
                robot_idx,
                fixture_id,
                ref_object_id=object_id,
            )
        self._sync_held_object(robot_idx)
        return fixture_id

    def _semantic_ref(self, resolver: str, **kwargs) -> dict[str, Any]:
        return {"$ref": resolver, **kwargs}

    def _build_hotdog_setup_demo_template(self) -> list[dict[str, Any]]:
        return [
            {
                "tool": "communicate",
                "robot_idx": 0,
                "args": {
                    "to": "agent_1",
                    "message": "I will stage the bun and condiment at the dining table.",
                },
            },
            {
                "tool": "communicate",
                "robot_idx": 1,
                "args": {
                    "to": "agent_0",
                    "message": "I will bring the sausage from the fridge once the plate is ready.",
                },
            },
            {
                "tool": "navigate_to_fixture",
                "robot_idx": 0,
                "args": {
                    "fixture_id": self._semantic_ref(
                        "source_fixture",
                        object_id="hotdog_bun",
                    )
                },
            },
            {
                "tool": "pick_up_object",
                "robot_idx": 0,
                "args": {
                    "object_id": "hotdog_bun",
                    "source_id": self._semantic_ref(
                        "source_fixture",
                        object_id="hotdog_bun",
                    ),
                },
            },
            {
                "tool": "navigate_to_fixture",
                "robot_idx": 1,
                "args": {
                    "fixture_id": self._semantic_ref(
                        "source_fixture",
                        object_id="sausage",
                        preferred_fixture_types=["fridge"],
                    )
                },
            },
            {
                "tool": "open_hinged_part",
                "robot_idx": 1,
                "args": {
                    "target_id": self._semantic_ref(
                        "source_fixture",
                        object_id="sausage",
                        preferred_fixture_types=["fridge"],
                    ),
                    "part_id": "hinged",
                },
            },
            {
                "tool": "navigate_to_fixture",
                "robot_idx": 0,
                "args": {
                    "fixture_id": self._semantic_ref(
                        "object_anchor_fixture",
                        object_id="plate",
                        preferred_fixture_types=[
                            "dining_counter",
                            "island",
                            "counter_non_dining",
                        ],
                        require_placeable=True,
                    )
                },
            },
            {
                "tool": "place_on_object",
                "robot_idx": 0,
                "args": {
                    "object_id": "hotdog_bun",
                    "support_object_id": "plate",
                    "anchor_fixture_id": self._semantic_ref(
                        "object_anchor_fixture",
                        object_id="plate",
                        preferred_fixture_types=[
                            "dining_counter",
                            "island",
                            "counter_non_dining",
                        ],
                        require_placeable=True,
                    ),
                },
            },
            {
                "tool": "communicate",
                "robot_idx": 0,
                "args": {
                    "to": "agent_1",
                    "message": "Plate is staged on the dining table. Bring the sausage now.",
                },
            },
            {
                "tool": "pick_up_object",
                "robot_idx": 1,
                "args": {
                    "object_id": "sausage",
                    "source_id": self._semantic_ref(
                        "source_fixture",
                        object_id="sausage",
                        preferred_fixture_types=["fridge"],
                    ),
                },
            },
            {
                "tool": "navigate_to_fixture",
                "robot_idx": 1,
                "args": {
                    "fixture_id": self._semantic_ref(
                        "object_anchor_fixture",
                        object_id="plate",
                        preferred_fixture_types=[
                            "dining_counter",
                            "island",
                            "counter_non_dining",
                        ],
                        require_placeable=True,
                    )
                },
            },
            {
                "tool": "place_on_object",
                "robot_idx": 1,
                "args": {
                    "object_id": "sausage",
                    "support_object_id": "plate",
                    "anchor_fixture_id": self._semantic_ref(
                        "object_anchor_fixture",
                        object_id="plate",
                        preferred_fixture_types=[
                            "dining_counter",
                            "island",
                            "counter_non_dining",
                        ],
                        require_placeable=True,
                    ),
                },
            },
            {
                "tool": "communicate",
                "robot_idx": 1,
                "args": {
                    "to": "agent_0",
                    "message": "Sausage is placed. The dining table is clear for the condiment.",
                },
            },
            {
                "tool": "navigate_to_fixture",
                "robot_idx": 0,
                "args": {
                    "fixture_id": self._semantic_ref(
                        "source_fixture",
                        object_id="condiment",
                    )
                },
            },
            {
                "tool": "pick_up_object",
                "robot_idx": 0,
                "args": {
                    "object_id": "condiment",
                    "source_id": self._semantic_ref(
                        "source_fixture",
                        object_id="condiment",
                    ),
                },
            },
            {
                "tool": "navigate_to_fixture",
                "robot_idx": 0,
                "args": {
                    "fixture_id": self._semantic_ref(
                        "object_anchor_fixture",
                        object_id="plate",
                        preferred_fixture_types=[
                            "dining_counter",
                            "island",
                            "counter_non_dining",
                        ],
                        require_placeable=True,
                    )
                },
            },
            {
                "tool": "place_on_surface",
                "robot_idx": 0,
                "args": {
                    "object_id": "condiment",
                    "support_id": self._semantic_ref(
                        "object_anchor_fixture",
                        object_id="plate",
                        preferred_fixture_types=[
                            "dining_counter",
                            "island",
                            "counter_non_dining",
                        ],
                        require_placeable=True,
                    ),
                },
            },
            {
                "tool": "communicate",
                "robot_idx": 0,
                "args": {
                    "to": "agent_1",
                    "message": "Hot dog setup complete.",
                },
            },
        ]

    def _build_sandwich_station_demo_template(self) -> list[dict[str, Any]]:
        """Demo plan for PrepareSandwichStation (cooperative 2-robot).

        Robot 0: handles the ingredient bowl (fridge → counter near toaster_oven).
        Robot 1: handles the baguette (fridge → counter near toaster_oven).

        Coordination: R0 opens fridge and grabs bowl, R1 grabs baguette
        while fridge is still open, R1 closes fridge, both place on counter.
        """
        fridge_ref = self._semantic_ref(
            "source_fixture",
            object_id="ingredient_bowl",
            preferred_fixture_types=["fridge"],
        )
        counter_ref = self._semantic_ref(
            "nearest_fixture",
            anchor_fixture_type="toaster_oven",
            preferred_fixture_types=["counter", "counter_non_dining"],
            require_placeable=True,
        )

        return [
            # --- Phase 1: Both robots approach the fridge ---
            {
                "tool": "communicate",
                "robot_idx": 0,
                "args": {
                    "to": "agent_1",
                    "message": "I'll open the fridge and grab the ingredient bowl. "
                    "You grab the baguette after me.",
                },
            },
            {
                "tool": "communicate",
                "robot_idx": 1,
                "args": {
                    "to": "agent_0",
                    "message": "Got it. I'll grab the baguette and close the fridge.",
                },
            },
            # R0 opens fridge
            {
                "tool": "navigate_to_fixture",
                "robot_idx": 0,
                "args": {"fixture_id": fridge_ref},
            },
            {
                "tool": "open_hinged_part",
                "robot_idx": 0,
                "args": {"target_id": fridge_ref, "part_id": "hinged"},
            },
            # R0 picks ingredient bowl
            {
                "tool": "pick_up_object",
                "robot_idx": 0,
                "args": {
                    "object_id": "ingredient_bowl",
                    "source_id": fridge_ref,
                },
            },
            # --- Phase 2: R0 clears the fridge area, R1 takes over ---
            # R0 moves away so R1 has room at the fridge
            {
                "tool": "give_space",
                "robot_idx": 0,
                "args": {"fixture_id": fridge_ref},
            },
            # R1 picks baguette from still-open fridge
            {
                "tool": "pick_up_object",
                "robot_idx": 1,
                "args": {
                    "object_id": "baguette",
                    "source_id": self._semantic_ref(
                        "source_fixture",
                        object_id="baguette",
                        preferred_fixture_types=["fridge"],
                    ),
                },
            },
            # R1 closes fridge (last one out)
            {
                "tool": "close_hinged_part",
                "robot_idx": 1,
                "args": {"target_id": fridge_ref, "part_id": "hinged"},
            },
            # --- Phase 3: Both place on counter near toaster oven ---
            # R0 places bowl
            {
                "tool": "navigate_to_fixture",
                "robot_idx": 0,
                "args": {"fixture_id": counter_ref},
            },
            {
                "tool": "place_on_surface",
                "robot_idx": 0,
                "args": {
                    "object_id": "ingredient_bowl",
                    "support_id": counter_ref,
                },
            },
            {
                "tool": "communicate",
                "robot_idx": 0,
                "args": {
                    "to": "agent_1",
                    "message": "Bowl is placed. Your turn to place the baguette.",
                },
            },
            # R1 places baguette
            {
                "tool": "navigate_to_fixture",
                "robot_idx": 1,
                "args": {"fixture_id": counter_ref},
            },
            {
                "tool": "place_on_surface",
                "robot_idx": 1,
                "args": {
                    "object_id": "baguette",
                    "support_id": counter_ref,
                },
            },
            {
                "tool": "communicate",
                "robot_idx": 1,
                "args": {
                    "to": "agent_0",
                    "message": "Sandwich station is ready.",
                },
            },
        ]

    def _resolve_nearest_fixture(
        self,
        anchor_fixture_type: str,
        preferred_fixture_types: list[str] | set[str] | tuple[str, ...] | None = None,
        require_placeable: bool = False,
    ) -> str:
        """Find the fixture of *preferred_fixture_types* nearest to the first fixture of *anchor_fixture_type*."""
        scene = self.get_scene_description()
        fixtures = scene.get("fixtures", {})

        # Find the anchor fixture by type
        anchor_id = None
        for fid, finfo in fixtures.items():
            if finfo.get("fixture_type") == anchor_fixture_type:
                anchor_id = fid
                break
        if anchor_id is None:
            raise ValueError(f"No fixture of type {anchor_fixture_type!r} found")
        anchor_pos = np.asarray(fixtures[anchor_id]["position"][:2], dtype=float)
        preferred = self._normalize_preferred_fixture_types(preferred_fixture_types)

        best_id, best_dist = None, float("inf")
        for fid, finfo in fixtures.items():
            if fid == anchor_id:
                continue
            ftype = finfo.get("fixture_type")
            if preferred is not None and ftype not in preferred:
                continue
            if require_placeable and not finfo.get("can_place_objects", False):
                continue
            d = float(
                np.linalg.norm(
                    np.asarray(finfo["position"][:2], dtype=float) - anchor_pos
                )
            )
            if d < best_dist:
                best_dist = d
                best_id = fid

        if best_id is None:
            raise ValueError(
                f"No fixture of types {preferred} found near {anchor_fixture_type!r}"
            )
        return best_id

    def _resolve_semantic_ref(self, ref: dict[str, Any]) -> Any:
        resolver = ref.get("$ref")
        if resolver == "source_fixture":
            preferred_fixture_types = ref.get("preferred_fixture_types")
            return self._infer_source_fixture(
                ref["object_id"],
                preferred_fixture_types=preferred_fixture_types,
            )
        if resolver == "object_anchor_fixture":
            preferred_fixture_types = ref.get("preferred_fixture_types")
            return self._resolve_object_anchor_fixture(
                ref["object_id"],
                preferred_fixture_types=preferred_fixture_types,
                require_placeable=bool(ref.get("require_placeable", False)),
            )
        if resolver == "nearest_fixture":
            preferred_fixture_types = ref.get("preferred_fixture_types")
            return self._resolve_nearest_fixture(
                ref["anchor_fixture_type"],
                preferred_fixture_types=preferred_fixture_types,
                require_placeable=bool(ref.get("require_placeable", False)),
            )
        raise ValueError(f"Unknown semantic resolver {resolver!r}")

    def _ground_value(self, value: Any) -> Any:
        if isinstance(value, dict):
            if "$ref" in value:
                return self._resolve_semantic_ref(value)
            return {
                key: self._ground_value(subvalue) for key, subvalue in value.items()
            }
        if isinstance(value, list):
            return [self._ground_value(item) for item in value]
        return value

    def ground_plan_template(
        self, tool_calls: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        grounded_tool_calls = deepcopy(tool_calls)
        for tool_call in grounded_tool_calls:
            tool_call["args"] = self._ground_value(tool_call.get("args", {}))
        return grounded_tool_calls

    def build_demo_plan_template(self, demo_plan_name: str) -> list[dict[str, Any]]:
        normalized_name = demo_plan_name.strip().lower().replace("-", "_")

        if normalized_name == "cooperative_hotdog_setup":
            if self._get_task_class_name() != "HotDogSetup":
                raise ValueError(
                    "The cooperative_hotdog_setup demo plan requires --task HotDogSetup"
                )
            return self._build_hotdog_setup_demo_template()

        if normalized_name == "sandwich_station":
            if self._get_task_class_name() != "PrepareSandwichStation":
                raise ValueError(
                    "The sandwich_station demo plan requires --task PrepareSandwichStation"
                )
            return self._build_sandwich_station_demo_template()

        raise ValueError(
            f"Unknown demo plan {demo_plan_name!r}. "
            f"Available: {sorted(self._DEMO_TASK_BY_NAME)}"
        )

    def build_demo_plan(self, demo_plan_name: str) -> list[dict[str, Any]]:
        return self.ground_plan_template(self.build_demo_plan_template(demo_plan_name))

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def get_parts(self, fixture_id: str) -> list[str]:
        fixture = self._require_fixture(fixture_id)
        parts = []

        if hasattr(fixture, "door_joint_names"):
            for joint_name in fixture.door_joint_names:
                if "slide" in joint_name.lower() or "drawer" in joint_name.lower():
                    parts.append("sliding")
                else:
                    parts.append("hinged")
                parts.append(joint_name)

        if hasattr(fixture, "_joint_names"):
            for key in fixture._joint_names:
                if "lid" in key or "head" in key:
                    parts.append(key)

        return sorted(set(parts))

    def get_controls(self, fixture_id: str) -> list[str]:
        fixture = self._require_fixture(fixture_id)

        if hasattr(fixture, "_joint_names"):
            return sorted(fixture._joint_names.keys())
        if isinstance(fixture, CoffeeMachine):
            return sorted(fixture._start_button_names)
        if isinstance(fixture, Microwave):
            return ["start_button", "stop_button"]
        return []

    # ------------------------------------------------------------------
    # Primitive tools
    # ------------------------------------------------------------------

    def get_image(
        self,
        views: list[str] | tuple[str, ...] | str | None = None,
        image_paths: list[str] | tuple[str, ...] | str | None = None,
        agent_id: str | int | None = None,
        robot_idx: int = 0,
        view: str | None = None,
        image_path: str | None = None,
    ) -> ToolResult:
        if views is None:
            if view is None:
                raise ValueError("get_image requires views or view.")
            view_names = [str(view)]
        elif isinstance(views, str):
            view_names = [views]
        else:
            view_names = [str(view_name) for view_name in views]

        if not view_names:
            raise ValueError("get_image requires at least one view.")

        if image_paths is None:
            if image_path is None:
                raise ValueError("get_image requires image_paths or image_path.")
            requested_paths = [str(image_path)]
        elif isinstance(image_paths, (str, Path)):
            requested_paths = [str(image_paths)]
        else:
            requested_paths = [str(path) for path in image_paths]

        if len(requested_paths) != len(view_names):
            raise ValueError(
                "get_image requires image_paths to match the number of requested views."
            )

        resolved_agent_id = agent_id if agent_id is not None else f"agent_{robot_idx}"
        resolved_robot_idx = self._parse_agent_idx(resolved_agent_id)
        saved_paths: list[str] = []
        camera_names: list[str] = []

        normalized_view_names: list[str] = []
        for view_name, requested_path in zip(view_names, requested_paths):
            normalized_view = str(view_name).strip().lower()
            normalized_view_names.append(normalized_view)
            if normalized_view == "map":
                saved_path = self._save_map_image(
                    requested_path, clean_labels=self._clean_map_labels,
                )
                camera_name = "map"
            elif normalized_view in {"room_view", "top_view"}:
                camera_name = normalized_view
                image = self._render_camera(camera_name)
                saved_path = self._save_image(image, requested_path)
            else:
                _, camera_name = self._camera_name_for_agent_view(
                    resolved_agent_id,
                    normalized_view,
                )
                image = self._render_camera(camera_name)
                saved_path = self._save_image(image, requested_path)

            camera_names.append(camera_name)
            saved_paths.append(str(saved_path))

        details = {
            "robot_idx": resolved_robot_idx,
            "agent_id": str(resolved_agent_id),
            "views": normalized_view_names,
            "camera_names": camera_names,
            "image_paths": saved_paths,
        }
        if len(normalized_view_names) == 1:
            details["view"] = normalized_view_names[0]
            details["camera_name"] = camera_names[0]
            details["image_path"] = saved_paths[0]
        return ToolResult("get_image", True, details)

    def navigate_to_fixture(self, fixture_id: str, robot_idx: int = 0) -> ToolResult:
        fixture = self._require_fixture(fixture_id)
        placed = self._move_robot_near_fixture_with_retries(
            robot_idx,
            fixture_id,
            require_front=_require_front(fixture),
        )
        self._sync_held_object(robot_idx)
        return ToolResult(
            tool_name="navigate_to_fixture",
            success=placed,
            details={"fixture_id": fixture_id, "robot_idx": robot_idx},
        )

    def open_hinged_part(
        self,
        target_id: str,
        part_id: str,
        robot_idx: int = 0,
    ) -> ToolResult:
        fixture = self._require_fixture(target_id)
        if not self._robot_near_fixture(robot_idx, target_id):
            moved = self._move_robot_near_fixture_with_retries(
                robot_idx,
                target_id,
                require_front=True,
            )
            if moved:
                self._sync_held_object(robot_idx)
        if part_id == "hinged" and hasattr(fixture, "open_door"):
            fixture.open_door(env=self.env)
        else:
            joint_name = self._resolve_joint_name(fixture, part_id)
            self._set_named_joint(fixture, joint_name, 1.0)
        return ToolResult(
            "open_hinged_part", True, {"target_id": target_id, "part_id": part_id}
        )

    def close_hinged_part(
        self,
        target_id: str,
        part_id: str,
        robot_idx: int = 0,
    ) -> ToolResult:
        fixture = self._require_fixture(target_id)
        if not self._robot_near_fixture(robot_idx, target_id):
            moved = self._move_robot_near_fixture_with_retries(
                robot_idx,
                target_id,
                require_front=True,
            )
            if moved:
                self._sync_held_object(robot_idx)
        if part_id == "hinged" and hasattr(fixture, "close_door"):
            fixture.close_door(env=self.env)
        else:
            joint_name = self._resolve_joint_name(fixture, part_id)
            self._set_named_joint(fixture, joint_name, 0.0)
        return ToolResult(
            "close_hinged_part", True, {"target_id": target_id, "part_id": part_id}
        )

    def open_sliding_part(
        self,
        target_id: str,
        part_id: str,
        robot_idx: int = 0,
    ) -> ToolResult:
        fixture = self._require_fixture(target_id)
        if not self._robot_near_fixture(robot_idx, target_id):
            moved = self._move_robot_near_fixture_with_retries(
                robot_idx,
                target_id,
                require_front=True,
            )
            if moved:
                self._sync_held_object(robot_idx)
        joint_name = self._resolve_joint_name(
            fixture, part_id if part_id != "sliding" else "slide"
        )
        self._set_named_joint(fixture, joint_name, 1.0)
        return ToolResult(
            "open_sliding_part", True, {"target_id": target_id, "part_id": part_id}
        )

    def close_sliding_part(
        self,
        target_id: str,
        part_id: str,
        robot_idx: int = 0,
    ) -> ToolResult:
        fixture = self._require_fixture(target_id)
        if not self._robot_near_fixture(robot_idx, target_id):
            moved = self._move_robot_near_fixture_with_retries(
                robot_idx,
                target_id,
                require_front=True,
            )
            if moved:
                self._sync_held_object(robot_idx)
        joint_name = self._resolve_joint_name(
            fixture, part_id if part_id != "sliding" else "slide"
        )
        self._set_named_joint(fixture, joint_name, 0.0)
        return ToolResult(
            "close_sliding_part", True, {"target_id": target_id, "part_id": part_id}
        )

    def _robot_near_fixture(
        self, robot_idx: int, fixture_id: str, threshold: float = 1.5
    ) -> bool:
        """Return True if the robot is ready to interact with the fixture."""
        fxtr = self.runner._fixtures.get(fixture_id)
        if fxtr is None:
            return False
        pos = self.runner._get_robot_position(robot_idx)[:2]
        if _is_approach_center(fxtr):
            target_xy = self.runner._get_fixture_front_target_xy(fixture_id)
            metrics = get_front_alignment_metrics(fxtr, pos, target_xy=target_xy)
            if metrics is None:
                return False
            fixture_center = np.asarray(fxtr.pos[:2], dtype=float)
            return bool(
                metrics["on_front_face"]
                and metrics["within_span"]
                and metrics["lateral_offset"] <= MAX_FRONT_WORKING_LATERAL_OFFSET
                and _FRONT_READY_MIN_GAP <= metrics["front_gap"] <= _FRONT_READY_MAX_GAP
                and float(np.linalg.norm(pos - fixture_center))
                <= _FRONT_READY_MAX_CENTER_DISTANCE
            )
        fxtr_pos = np.asarray(fxtr.pos[:2], dtype=float)
        return float(np.linalg.norm(pos - fxtr_pos)) < threshold

    def _safe_compute_object_target_pos(
        self,
        support_id: str,
        object_id: str,
        preferred_xy: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute a placement target, falling back to legacy sampling on geometry errors."""
        support_fxtr = self.runner._fixtures[support_id]
        try:
            return self.runner._compute_object_target_pos(
                support_fxtr,
                object_id=object_id,
                preferred_xy=preferred_xy,
            )
        except Exception:
            target_pos = self.runner._compute_object_target_pos(support_fxtr)
            if preferred_xy is not None:
                target_pos[:2] = np.asarray(preferred_xy, dtype=float)[:2]
            return target_pos

    def _fixture_clearance_radius(self, fixture_id: str) -> float:
        """Return a radius around the fixture where teammates likely block access."""
        fixture = self.runner._fixtures.get(fixture_id)
        if fixture is None:
            return 1.5
        aabb = get_fixture_aabb(fixture)
        if aabb is None:
            return 1.5
        half_extent = 0.5 * np.linalg.norm(aabb[1] - aabb[0])
        return max(1.5, float(half_extent + 0.8))

    def _clear_fixture_blockers(self, robot_idx: int, fixture_id: str) -> bool:
        """Move nearby teammate robots away from a fixture before retrying placement."""
        fixture = self.runner._fixtures.get(fixture_id)
        if fixture is None:
            return False
        fixture_pos = np.asarray(fixture.pos[:2], dtype=float)
        clearance_radius = self._fixture_clearance_radius(fixture_id)
        moved_any = False
        for other_idx in range(self.runner._num_robots):
            if other_idx == robot_idx:
                continue
            other_pos = self.runner._get_robot_position(other_idx)[:2]
            if float(np.linalg.norm(other_pos - fixture_pos)) >= clearance_radius:
                continue
            self.runner.give_space(other_idx, fixture_id)
            self._sync_held_object(other_idx)
            moved_any = True
        return moved_any

    def _move_robot_near_fixture_with_retries(
        self,
        robot_idx: int,
        fixture_id: str,
        ref_object_id: str | None = None,
        ref_pos_override: np.ndarray | None = None,
        require_front: bool = False,
    ) -> bool:
        """Place a robot near a fixture, clearing teammate blockers if needed."""
        placed = self.runner._move_robot_near_fixture(
            robot_idx,
            fixture_id,
            ref_object_id=ref_object_id,
            ref_pos_override=ref_pos_override,
            require_front=require_front,
        )
        if placed or not require_front:
            return placed
        if not self._clear_fixture_blockers(robot_idx, fixture_id):
            return placed
        return self.runner._move_robot_near_fixture(
            robot_idx,
            fixture_id,
            ref_object_id=ref_object_id,
            ref_pos_override=ref_pos_override,
            require_front=require_front,
        )

    def pick_up_object(
        self,
        object_id: str,
        source_id: str,
        robot_idx: int = 0,
    ) -> ToolResult:
        self._require_object(object_id)
        self._require_fixture(source_id)

        current_holder = self._held_by_robot(object_id)
        if current_holder is not None and current_holder != robot_idx:
            raise ValueError(
                f"Object {object_id!r} is already held by robot {current_holder}"
            )

        # Skip navigation only if the robot is already in a usable working
        # pose for the fixture.
        if not self._robot_near_fixture(robot_idx, source_id):
            source_fxtr = self.runner._fixtures[source_id]
            if _is_approach_center(source_fxtr):
                # Interactive fixture (fridge, cabinet) — must approach from front
                moved = self._move_robot_near_fixture_with_retries(
                    robot_idx,
                    source_id,
                    require_front=True,
                )
            else:
                # Surface — pre-compute object position, pick closest face
                obj_pos, _ = self._get_object_pose(object_id)
                ref_pos = obj_pos[:2].copy()
                moved = self._move_robot_near_fixture_with_retries(
                    robot_idx,
                    source_id,
                    ref_pos_override=ref_pos,
                )
            if not moved:
                return ToolResult(
                    "pick_up_object",
                    False,
                    {"object_id": object_id, "source_id": source_id, "robot_idx": robot_idx},
                )
            self._sync_held_object(robot_idx)
        obj_pos_before, _ = self._get_object_pose(object_id)
        print(
            f"[pick_up] robot{robot_idx} picking {object_id} from {source_id}: "
            f"obj_before=({obj_pos_before[0]:.3f}, {obj_pos_before[1]:.3f}, {obj_pos_before[2]:.3f})"
        )
        self._held_objects[robot_idx] = object_id
        self._sync_held_object(robot_idx)
        obj_pos_after, _ = self._get_object_pose(object_id)
        print(
            f"[pick_up] after sync: obj=({obj_pos_after[0]:.3f}, {obj_pos_after[1]:.3f}, {obj_pos_after[2]:.3f})"
        )
        return ToolResult(
            "pick_up_object",
            True,
            {"object_id": object_id, "source_id": source_id, "robot_idx": robot_idx},
        )

    def place_on_surface(
        self,
        object_id: str,
        support_id: str,
        robot_idx: int = 0,
    ) -> ToolResult:
        self._require_object(object_id)
        self._require_fixture(support_id)
        holder = self._held_by_robot(object_id)
        if holder not in {None, robot_idx}:
            raise ValueError(f"Object {object_id!r} is held by robot {holder}")

        # Pre-compute where the object will land so the robot stands near it.
        target_pos = self._safe_compute_object_target_pos(support_id, object_id)
        self.runner._move_robot_near_fixture(
            robot_idx,
            support_id,
            ref_pos_override=target_pos[:2],
        )
        self.runner.move_object(object_id, support_id, target_pos=target_pos)
        self._held_objects.pop(robot_idx, None)
        return ToolResult(
            "place_on_surface",
            True,
            {"object_id": object_id, "support_id": support_id, "robot_idx": robot_idx},
        )

    def place_in_receptacle(
        self,
        object_id: str,
        receptacle_id: str,
        robot_idx: int = 0,
    ) -> ToolResult:
        self._require_object(object_id)
        holder = self._held_by_robot(object_id)
        if holder not in {None, robot_idx}:
            raise ValueError(f"Object {object_id!r} is held by robot {holder}")

        if receptacle_id in self.runner._fixtures:
            target_pos = self._safe_compute_object_target_pos(receptacle_id, object_id)
            self.runner._move_robot_near_fixture(
                robot_idx,
                receptacle_id,
                ref_pos_override=target_pos[:2],
            )
            self.runner.move_object(object_id, receptacle_id, target_pos=target_pos)
        else:
            self._require_object(receptacle_id)
            contained = self._find_contained_objects(object_id)
            self._place_on_object_center(object_id, receptacle_id)
            anchor_fixture_id = self._get_scene_object_location(receptacle_id)
            if anchor_fixture_id is not None:
                self.runner._set_object_location(object_id, anchor_fixture_id)
                for child_id in contained:
                    self.runner._set_object_location(child_id, anchor_fixture_id)

        self._held_objects.pop(robot_idx, None)
        return ToolResult(
            "place_in_receptacle",
            True,
            {
                "object_id": object_id,
                "receptacle_id": receptacle_id,
                "robot_idx": robot_idx,
            },
        )

    def place_next_to(
        self,
        object_id: str,
        reference_object_id: str,
        robot_idx: int = 0,
    ) -> ToolResult:
        """Place an object adjacent to another object or fixture on a surface.

        ``reference_object_id`` may refer to a sim object *or* a fixture
        (e.g. ``toaster_oven_main_group``).  When it's a fixture, the object
        is placed on the nearest placeable surface adjacent to that fixture.
        """
        self._require_object(object_id)
        holder = self._held_by_robot(object_id)
        if holder not in {None, robot_idx}:
            raise ValueError(f"Object {object_id!r} is held by robot {holder}")

        # Determine if the reference is a fixture or an object.
        ref_is_fixture = reference_object_id in self.runner._fixtures

        if ref_is_fixture:
            fixture = self.runner._fixtures[reference_object_id]
            ref_pos = np.asarray(fixture.pos, dtype=float)
            # Use the fixture's AABB to compute extent for offset.
            aabb = get_fixture_aabb(fixture)
            if aabb is not None:
                ref_extent_x = float(aabb[1][0] - aabb[0][0])
                ref_extent_y = float(aabb[1][1] - aabb[0][1])
            else:
                ref_extent_x = ref_extent_y = 0.2
            # Find the surface the fixture sits on.
            support_fixture_id = self._find_placeable_surface_near_fixture(
                reference_object_id
            )
        else:
            self._require_object(reference_object_id)
            ref_pos, _ = self._get_object_pose(reference_object_id)
            ref_obj = self._require_object(reference_object_id)
            ref_body_id = self.env.obj_body_id[reference_object_id]
            ref_body_pos = self.env.sim.data.body_xpos[ref_body_id].copy()
            ref_quat_xyzw = T.convert_quat(
                self.env.sim.data.body_xquat[ref_body_id].copy(), to="xyzw"
            )
            ref_bbox = ref_obj.get_bbox_points(trans=ref_body_pos, rot=ref_quat_xyzw)
            ref_extent_x = max(p[0] for p in ref_bbox) - min(p[0] for p in ref_bbox)
            ref_extent_y = max(p[1] for p in ref_bbox) - min(p[1] for p in ref_bbox)
            # Find what fixture the reference object is on.
            support_fixture_id = self._get_scene_object_location(reference_object_id)
            if support_fixture_id is None:
                support_fixture_id = self.runner._find_object_fixture(ref_pos)

        self._require_fixture(support_fixture_id)
        self.runner._move_robot_near_fixture(robot_idx, support_fixture_id)

        # Offset along the longer axis to place beside.
        offset_axis = 0 if ref_extent_x > ref_extent_y else 1
        offset_magnitude = max(ref_extent_x, ref_extent_y) * 0.5 + 0.05

        preferred_positions = []
        for direction in (1.0, -1.0):
            preferred_xy = ref_pos[:2].copy()
            preferred_xy[offset_axis] += direction * offset_magnitude
            preferred_positions.append(preferred_xy)

        scored_targets: list[tuple[float, np.ndarray]] = []
        for preferred_xy in preferred_positions:
            target_pos = self._safe_compute_object_target_pos(
                support_fixture_id,
                object_id,
                preferred_xy=preferred_xy,
            )
            score = float(np.linalg.norm(target_pos[:2] - preferred_xy[:2]))
            scored_targets.append((score, target_pos))
        target_pos = min(scored_targets, key=lambda item: item[0])[1]

        self.runner._move_robot_near_fixture(
            robot_idx,
            support_fixture_id,
            ref_pos_override=target_pos[:2],
        )
        self.runner.move_object(object_id, support_fixture_id, target_pos=target_pos)

        self._held_objects.pop(robot_idx, None)
        return ToolResult(
            "place_next_to",
            True,
            {
                "object_id": object_id,
                "reference_object_id": reference_object_id,
                "support_fixture_id": support_fixture_id,
                "robot_idx": robot_idx,
            },
        )

    def place_under(
        self,
        object_id: str,
        reference_fixture_id: str,
        robot_idx: int = 0,
    ) -> ToolResult:
        """Place an object directly beneath a reference fixture.

        For dispenser-type fixtures (CoffeeMachine, Sink) the object is
        positioned at the dispenser output site.  For all other fixtures
        (e.g. cabinet, hood) the object is placed on the nearest surface
        below, with XY aligned to the reference fixture.
        """
        self._require_object(object_id)
        fixture = self._require_fixture(reference_fixture_id)
        holder = self._held_by_robot(object_id)
        if holder not in {None, robot_idx}:
            raise ValueError(f"Object {object_id!r} is held by robot {holder}")

        self.runner._move_robot_near_fixture(robot_idx, reference_fixture_id)

        if isinstance(fixture, CoffeeMachine):
            site_name = f"{fixture.naming_prefix}receptacle_place_site"
            site_id = self.env.sim.model.site_name2id(site_name)
            target_pos = self.env.sim.data.site_xpos[site_id].copy()
            contained = self._find_contained_objects(object_id)
            self._set_object_pose(object_id, target_pos)
            support_fixture_id = self._find_placeable_surface_near_fixture(
                reference_fixture_id
            )
            self.runner._set_object_location(object_id, support_fixture_id)
            for child_id in contained:
                self.runner._set_object_location(child_id, support_fixture_id)
        elif isinstance(fixture, Sink):
            water_site_name = fixture.water_site.get("name")
            site_id = self.env.sim.model.site_name2id(water_site_name)
            target_pos = self.env.sim.data.site_xpos[site_id].copy()
            contained = self._find_contained_objects(object_id)
            self._set_object_pose(object_id, target_pos)
            support_fixture_id = self._find_placeable_surface_near_fixture(
                reference_fixture_id
            )
            self.runner._set_object_location(object_id, support_fixture_id)
            for child_id in contained:
                self.runner._set_object_location(child_id, support_fixture_id)
        else:
            # Generic: project fixture XY, find the surface below.
            fxtr_pos = np.asarray(fixture.pos, dtype=float)
            support_fixture_id = self._find_placeable_surface_near_fixture(
                reference_fixture_id
            )
            target_pos = self._safe_compute_object_target_pos(
                support_fixture_id,
                object_id,
                preferred_xy=fxtr_pos[:2],
            )
            self.runner.move_object(
                object_id,
                support_fixture_id,
                target_pos=target_pos,
                preferred_xy=fxtr_pos[:2],
            )

        self._held_objects.pop(robot_idx, None)
        return ToolResult(
            "place_under",
            True,
            {
                "object_id": object_id,
                "reference_fixture_id": reference_fixture_id,
                "robot_idx": robot_idx,
            },
        )

    def place_on_object(
        self,
        object_id: str,
        support_object_id: str,
        anchor_fixture_id: str | None = None,
        robot_idx: int = 0,
    ) -> ToolResult:
        self._require_object(object_id)
        self._require_object(support_object_id)
        holder = self._held_by_robot(object_id)
        if holder not in {None, robot_idx}:
            raise ValueError(f"Object {object_id!r} is held by robot {holder}")

        # Pre-compute the support object's position — this is where the
        # object will land, so the robot should stand near it (not the
        # fixture center).  Mirrors how place_on_surface pre-computes the
        # drop position via ref_pos_override.
        support_pos, _ = self._get_object_pose(support_object_id)
        ref_pos = support_pos[:2].copy()

        if anchor_fixture_id is None:
            anchor_fixture_id = self._resolve_object_anchor_fixture(
                support_object_id,
                preferred_fixture_types=[
                    "dining_counter",
                    "island",
                    "counter_non_dining",
                ],
                require_placeable=True,
            )
        self._require_fixture(anchor_fixture_id)
        self.runner._move_robot_near_fixture(
            robot_idx,
            anchor_fixture_id,
            ref_pos_override=ref_pos,
        )
        self._sync_held_object(robot_idx)
        contained = self._find_contained_objects(object_id)
        self._place_on_object_center(object_id, support_object_id)
        self.runner._set_object_location(object_id, anchor_fixture_id)
        for child_id in contained:
            self.runner._set_object_location(child_id, anchor_fixture_id)
        self._held_objects.pop(robot_idx, None)
        return ToolResult(
            "place_on_object",
            True,
            {
                "object_id": object_id,
                "support_object_id": support_object_id,
                "anchor_fixture_id": anchor_fixture_id,
                "robot_idx": robot_idx,
            },
        )

    def press_button(
        self,
        target_id: str,
        control_id: str,
        robot_idx: int = 0,
    ) -> ToolResult:
        fixture = self._require_fixture(target_id)
        moved = self._move_robot_near_fixture_with_retries(
            robot_idx,
            target_id,
            require_front=_require_front(fixture),
        )
        if moved:
            self._sync_held_object(robot_idx)

        if isinstance(fixture, Microwave):
            fixture._turned_on = control_id == "start_button"
        elif isinstance(fixture, CoffeeMachine):
            fixture._turned_on = True
        elif isinstance(fixture, ElectricKettle):
            if control_id == "lid_button":
                fixture.set_lid(self.env, lid_val=1.0, gradual=False)
            else:
                raise ValueError(f"Unsupported kettle button {control_id!r}")
        else:
            joint_name = self._resolve_joint_name(fixture, control_id)
            self._set_named_joint(fixture, joint_name, 1.0)

        if hasattr(fixture, "update_state"):
            fixture.update_state(self.env)
        self.env.sim.forward()
        return ToolResult(
            "press_button", True, {"target_id": target_id, "control_id": control_id}
        )

    def press_lever(
        self,
        target_id: str,
        control_id: str,
        robot_idx: int = 0,
    ) -> ToolResult:
        fixture = self._require_fixture(target_id)
        moved = self._move_robot_near_fixture_with_retries(
            robot_idx,
            target_id,
            require_front=_require_front(fixture),
        )
        if moved:
            self._sync_held_object(robot_idx)

        if isinstance(fixture, Toaster):
            slot_pair = 0
            if "_" in control_id:
                _, slot_pair_str = control_id.rsplit("_", 1)
                slot_pair = int(slot_pair_str)
            fixture.set_lever(self.env, slot_pair=slot_pair, value=1.0)
        elif isinstance(fixture, ElectricKettle):
            fixture.set_power_state(self.env, power_on=True)
        else:
            joint_name = self._resolve_joint_name(fixture, control_id)
            self._set_named_joint(fixture, joint_name, 1.0)

        if hasattr(fixture, "update_state"):
            fixture.update_state(self.env)
        self.env.sim.forward()
        return ToolResult(
            "press_lever", True, {"target_id": target_id, "control_id": control_id}
        )

    def set_rotary_control(
        self,
        target_id: str,
        control_id: str,
        goal: str,
        robot_idx: int = 0,
    ) -> ToolResult:
        fixture = self._require_fixture(target_id)
        moved = self._move_robot_near_fixture_with_retries(
            robot_idx,
            target_id,
            require_front=True,
        )
        if moved:
            self._sync_held_object(robot_idx)

        goal_lower = str(goal).lower()
        if goal_lower in {"on", "open", "high", "max", "1", "true"}:
            value = 1.0
        elif goal_lower in {"off", "close", "low", "min", "0", "false"}:
            value = 0.0
        else:
            value = float(goal)

        if isinstance(fixture, Toaster) and control_id.startswith("knob"):
            slot_pair = 0
            if "_" in control_id:
                _, slot_pair_str = control_id.rsplit("_", 1)
                slot_pair = int(slot_pair_str)
            fixture.set_doneness_knob(self.env, slot_pair=slot_pair, value=value)
        else:
            joint_name = self._resolve_joint_name(fixture, control_id)
            self._set_named_joint(fixture, joint_name, value)

        if hasattr(fixture, "update_state"):
            fixture.update_state(self.env)
        self.env.sim.forward()
        return ToolResult(
            "set_rotary_control",
            True,
            {"target_id": target_id, "control_id": control_id, "goal": goal},
        )

    def communicate(
        self,
        to: str,
        message: str,
        robot_idx: int = 0,
    ) -> ToolResult:
        return ToolResult(
            "communicate",
            True,
            {
                "robot_idx": robot_idx,
                "to": to,
                "message": message,
            },
        )

    def give_space(self, fixture_id: str, robot_idx: int = 0) -> ToolResult:
        """Move the robot to open floor space away from *fixture_id*.

        Delegates to ``TrajectoryRunner.give_space`` which finds a free grid
        cell (grid mode) or a standable position on expanding circles
        (continuous mode) that is ≥1.5 m from the fixture.
        """
        self._require_fixture(fixture_id)
        self.runner.give_space(robot_idx, fixture_id)
        self._sync_held_object(robot_idx)
        return ToolResult(
            "give_space",
            True,
            {
                "fixture_id": fixture_id,
                "robot_idx": robot_idx,
            },
        )

    def wait(self, robot_idx: int = 0) -> ToolResult:
        return ToolResult(
            "wait",
            True,
            {"robot_idx": robot_idx},
        )

    # ------------------------------------------------------------------
    # Generic dispatch
    # ------------------------------------------------------------------

    def execute(self, tool_name: str, robot_idx: int = 0, **kwargs) -> ToolResult:
        if tool_name not in SIM_TOOL_SPEC_BY_NAME:
            raise ValueError(f"Unknown tool {tool_name!r}")

        method = getattr(self, tool_name, None)
        if method is None:
            raise NotImplementedError(f"No executor method defined for {tool_name!r}")
        return method(robot_idx=robot_idx, **kwargs)


def _main():
    parser = argparse.ArgumentParser(
        description="Execute simulator tools and save scene frames / videos."
    )
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--robots", type=int, default=2)
    parser.add_argument(
        "--layout",
        type=int,
        default=None,
        help=(
            "Kitchen layout id. If omitted, uses trajectory scene_parameters.layout "
            "when present, otherwise defaults to 11."
        ),
    )
    parser.add_argument(
        "--style",
        type=int,
        default=None,
        help=(
            "Kitchen style id. If omitted, uses trajectory scene_parameters.style "
            "when present, otherwise defaults to 34."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Environment seed. If omitted, uses trajectory scene_parameters.seed "
            "when present, otherwise defaults to 42."
        ),
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--plan",
        type=str,
        default=None,
        help="Path to a JSON tool plan. If omitted, only current-scene frames are saved.",
    )
    parser.add_argument(
        "--demo-plan",
        type=str,
        default=None,
        help=(
            "Name of a built-in tool plan. Available: cooperative_hotdog_setup, sandwich_station. "
            "Mutually exclusive with --plan and --trajectory."
        ),
    )
    parser.add_argument(
        "--trajectory",
        type=str,
        default=None,
        help=(
            "Path to a full external trajectory JSON. Mutually exclusive with "
            "--plan and --demo-plan."
        ),
    )
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument(
        "--skip-videos",
        action="store_true",
        default=False,
        help="Skip MP4 video generation to reduce output size.",
    )
    parser.add_argument(
        "--raw-map-labels",
        action="store_true",
        default=False,
        help="Show full raw fixture ids on the map instead of cleaned-up labels.",
    )
    parser.add_argument(
        "--robot-spawn",
        choices=["sim", "trajectory"],
        default="trajectory",
        help=(
            "Robot initial placement source. 'trajectory' (default): place each robot "
            "at the location specified in the trajectory's initial_state. 'sim': place "
            "all robots at init_robot_base_ref (task ground truth)."
        ),
    )
    parser.add_argument("--gl-backend", type=str, default="osmesa")
    parser.add_argument(
        "--placement",
        choices=["grid", "continuous"],
        default="grid",
        help="Robot placement strategy: grid (occupancy grid) or continuous (AABB-based).",
    )
    parser.add_argument(
        "--cell-size",
        type=float,
        default=0.05,
        help="Grid cell size in meters (only used when --placement=grid). Default: 0.05",
    )
    parser.add_argument(
        "--align-to-wall",
        action="store_true",
        default=False,
        help="Align grid origin so cell boundaries fall on y=0 and x=0 (wall edges).",
    )
    parser.add_argument(
        "--standoff",
        type=float,
        default=0.30,
        help="Distance from fixture face to robot center (meters). Default: 0.30",
    )
    parser.add_argument(
        "--sample-spacing",
        type=float,
        default=0.12,
        help="Spacing between candidate samples along fixture faces (continuous mode). Default: 0.12",
    )
    parser.add_argument(
        "--robot-radius",
        type=float,
        default=0.18,
        help="Robot collision radius for placement (meters). Default: 0.18",
    )
    args = parser.parse_args()

    provided_inputs = [
        args.plan is not None,
        args.demo_plan is not None,
        args.trajectory is not None,
    ]
    if sum(provided_inputs) > 1:
        parser.error("Use only one of --plan, --demo-plan, or --trajectory.")

    trajectory_payload = None
    if args.trajectory is not None:
        with open(args.trajectory, "r") as f:
            trajectory_payload = json.load(f)

    def _resolve_scene_parameter(name: str, fallback: int) -> int:
        explicit_value = getattr(args, name)
        if explicit_value is not None:
            return explicit_value
        if isinstance(trajectory_payload, dict):
            scene_parameters = trajectory_payload.get("scene_parameters")
            if isinstance(scene_parameters, dict):
                scene_value = scene_parameters.get(name)
                if scene_value is not None:
                    return int(scene_value)
            top_level_value = trajectory_payload.get(name)
            if top_level_value is not None:
                return int(top_level_value)
        return fallback

    layout = _resolve_scene_parameter("layout", 11)
    style = _resolve_scene_parameter("style", 34)
    seed = _resolve_scene_parameter("seed", 42)

    if args.task is None:
        if args.demo_plan is not None:
            demo_key = args.demo_plan.strip().lower().replace("-", "_")
            task_name = SimToolExecutor._DEMO_TASK_BY_NAME.get(demo_key)
            if task_name is None:
                parser.error(
                    "Unknown demo plan. Available: cooperative_hotdog_setup, sandwich_station"
                )
        elif (
            trajectory_payload is not None
            and trajectory_payload.get("composite_task") is not None
        ):
            task_name = trajectory_payload["composite_task"]
        else:
            task_name = "MicrowaveThawing"
    else:
        task_name = args.task

    executor = SimToolExecutor(
        task_name=task_name,
        robots=args.robots,
        layout=layout,
        style=style,
        seed=seed,
        render_width=args.width,
        render_height=args.height,
        gl_backend=args.gl_backend,
        placement=args.placement,
        cell_size=args.cell_size,
        align_to_wall=args.align_to_wall,
        standoff=args.standoff,
        sample_spacing=args.sample_spacing,
        robot_radius=args.robot_radius,
        robot_spawn=args.robot_spawn,
    )
    # --raw-map-labels disables fixture label cleanup on the map
    executor._clean_map_labels = not args.raw_map_labels

    try:
        if args.plan is None and args.demo_plan is None and args.trajectory is None:
            # No trajectory/plan — save map and frames now
            map_path = executor.save_placement_map(
                args.output_dir, prefix="initial",
                clean_labels=executor._clean_map_labels,
            )
            print(f"Placement map: {map_path}")
            saved = executor.save_scene_frames(args.output_dir, prefix="initial")
            print(json.dumps({k: str(v) for k, v in saved.items()}, indent=2))
        else:
            if args.trajectory is not None:
                from robocasa.utils.trajectory_adapter import execute_trajectory
                import shutil

                # Copy original trajectory JSON to output dir
                out_path = Path(args.output_dir)
                out_path.mkdir(parents=True, exist_ok=True)
                shutil.copy2(args.trajectory, out_path / "original_trajectory.json")

                metadata = execute_trajectory(
                    executor=executor,
                    trajectory=trajectory_payload,
                    output_dir=args.output_dir,
                    fps=args.fps,
                    skip_videos=args.skip_videos,
                )
                # The adapter already saves initial_map.png right after
                # load_initial_state (before trajectory steps run), so the
                # map reflects the true initial robot/object positions.
                # Do NOT re-save here — that would overwrite with post-
                # execution positions.
            elif args.demo_plan is not None:
                tool_calls = executor.build_demo_plan(args.demo_plan)
                metadata = executor.run_tool_plan(
                    tool_calls=tool_calls,
                    output_dir=args.output_dir,
                    fps=args.fps,
                )
            else:
                with open(args.plan, "r") as f:
                    tool_calls = json.load(f)
                metadata = executor.run_tool_plan(
                    tool_calls=tool_calls,
                    output_dir=args.output_dir,
                    fps=args.fps,
                )
            # Save map after execution for demo_plan/plan paths
            if args.trajectory is None:
                map_path = executor.save_placement_map(
                    args.output_dir, prefix="initial",
                    clean_labels=executor._clean_map_labels,
                )
                print(f"Placement map: {map_path}")
            print(json.dumps(metadata, indent=2))
    finally:
        executor.close()


__all__ = ["SimToolExecutor", "ToolResult"]


if __name__ == "__main__":
    _main()
