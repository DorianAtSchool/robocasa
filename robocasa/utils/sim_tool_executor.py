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
from robocasa.models.fixtures.toaster import Toaster
from robocasa.utils.sim_tool_specs import SIM_TOOL_SPEC_BY_NAME
from robocasa.utils.trajectory_runner import TrajectoryRunner


@dataclass
class ToolResult:
    tool_name: str
    success: bool
    details: dict[str, Any]


class SimToolExecutor:
    """Dispatch simulator tool calls against a live RoboCasa environment."""

    _HELD_Z_OFFSET = 0.12
    _DEMO_TASK_BY_NAME = {
        "cooperative_hotdog_setup": "HotDogSetup",
    }

    def __init__(
        self,
        task_name: str = "Kitchen",
        robots: int = 2,
        layout: int | None = None,
        style: int | None = None,
        seed: int | None = None,
        render_width: int = 512,
        render_height: int = 512,
        gl_backend: str = "osmesa",
    ):
        os.environ["MUJOCO_GL"] = gl_backend
        self.runner = TrajectoryRunner(
            task_name=task_name,
            robots=robots,
            layout=layout,
            style=style,
            seed=seed,
            render_width=render_width,
            render_height=render_height,
            gl_backend=gl_backend,
        )
        self.env = self.runner.env
        self._held_objects: dict[int, str] = {}

    # ------------------------------------------------------------------
    # Scene / state helpers
    # ------------------------------------------------------------------

    def close(self):
        self.runner.close()

    def get_scene_description(self) -> dict[str, Any]:
        return self.runner.get_scene_description()

    def render(self) -> dict[str, np.ndarray]:
        return self.runner.render()

    def save_scene_frames(self, output_dir: str | Path, prefix: str = "initial") -> dict[str, Path]:
        """Render the current scene and save one image per camera."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        frames = self.render()
        saved = {}
        for camera_name, image in frames.items():
            camera_dir = output_dir / camera_name
            camera_dir.mkdir(parents=True, exist_ok=True)
            path = camera_dir / f"{prefix}.png"
            imageio.imwrite(path, image)
            saved[camera_name] = path
        return saved

    def run_tool_plan(
        self,
        tool_calls: list[dict[str, Any]],
        output_dir: str | Path,
        fps: int = 2,
    ) -> dict[str, Any]:
        """
        Execute a list of tool calls and save before / after frames plus per-camera videos.

        Expected plan format:
        [
          {"tool": "navigate_to_fixture", "robot_idx": 0, "args": {"fixture_id": "..."}},
          {"tool": "pick_up_object", "args": {"object_id": "...", "source_id": "..."}}
        ]
        """
        tool_calls = self.ground_plan_template(tool_calls)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        frames_dir = output_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        initial_frames = self.render()
        camera_names = list(initial_frames.keys())
        writers = {
            camera_name: imageio.get_writer(
                output_dir / f"{camera_name}.mp4",
                fps=fps,
            )
            for camera_name in camera_names
        }

        metadata = {
            "task": self.get_scene_description().get("task"),
            "steps": [],
            "cameras": camera_names,
        }

        def record_frame_bundle(tag: str, frames: dict[str, np.ndarray]):
            for camera_name, image in frames.items():
                camera_dir = frames_dir / camera_name
                camera_dir.mkdir(parents=True, exist_ok=True)
                imageio.imwrite(camera_dir / f"{tag}.png", image)
                writers[camera_name].append_data(image)

        try:
            record_frame_bundle("initial", initial_frames)

            for step_idx, tool_call in enumerate(tool_calls):
                tool_name = tool_call["tool"]
                robot_idx = tool_call.get("robot_idx", 0)
                args = tool_call.get("args", {})

                before = self.render()
                record_frame_bundle(f"step_{step_idx:03d}_before", before)

                result = self.execute(tool_name, robot_idx=robot_idx, **args)

                after = self.render()
                record_frame_bundle(f"step_{step_idx:03d}_after", after)

                metadata["steps"].append(
                    {
                        "step_index": step_idx,
                        "tool": tool_name,
                        "robot_idx": robot_idx,
                        "args": args,
                        "success": result.success,
                        "details": result.details,
                    }
                )
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
        _, current_quat = self._get_object_pose(object_id)
        quat_arr = current_quat if quat is None else np.asarray(quat, dtype=float)
        self.env.sim.data.set_joint_qpos(
            obj.joints[0],
            np.concatenate([np.asarray(pos, dtype=float), quat_arr]),
        )
        self.env.sim.forward()

    def _get_robot_eef_pos(self, robot_idx: int) -> np.ndarray:
        site_id = self.env.robots[robot_idx].eef_site_id["right"]
        return self.env.sim.data.site_xpos[site_id].copy()

    def _sync_held_object(self, robot_idx: int):
        object_id = self._held_objects.get(robot_idx)
        if object_id is None:
            return
        eef_pos = self._get_robot_eef_pos(robot_idx)
        held_pos = eef_pos.copy()
        held_pos[2] += self._HELD_Z_OFFSET
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
        scene = self.get_scene_description()
        object_info = scene.get("objects", {}).get(object_id, {})
        location = object_info.get("location")
        if isinstance(location, str) and location in scene.get("fixtures", {}):
            return location
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

    def _find_nearest_fixture_for_object(
        self,
        object_id: str,
        preferred_fixture_types: set[str] | None = None,
        require_placeable: bool = False,
    ) -> str:
        object_pos, _ = self._get_object_pose(object_id)
        scene = self.get_scene_description()
        fixtures = scene.get("fixtures", {})

        def candidate_ids() -> list[str]:
            candidates = []
            for fixture_id, fixture_info in fixtures.items():
                if (
                    preferred_fixture_types is not None
                    and fixture_info.get("fixture_type") not in preferred_fixture_types
                ):
                    continue
                if require_placeable and not fixture_info.get("can_place_objects", False):
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
                    object_pos[:2] - np.asarray(fixtures[fixture_id]["position"][:2], dtype=float)
                )
            ),
        )

    def _resolve_object_anchor_fixture(
        self,
        object_id: str,
        preferred_fixture_types: set[str] | None = None,
        require_placeable: bool = False,
    ) -> str:
        scene = self.get_scene_description()
        fixtures = scene.get("fixtures", {})

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

        if explicit_location is not None:
            for nearby_fixture_id in fixtures.get(explicit_location, {}).get("nearby_fixtures", []):
                if matches(nearby_fixture_id):
                    return nearby_fixture_id

        return self._find_nearest_fixture_for_object(
            object_id,
            preferred_fixture_types=preferred_fixture_types,
            require_placeable=require_placeable,
        )

    def _infer_source_fixture(
        self,
        object_id: str,
        preferred_fixture_types: set[str] | None = None,
    ) -> str:
        return self._resolve_object_anchor_fixture(
            object_id,
            preferred_fixture_types=preferred_fixture_types,
        )

    def _move_robot_near_object_anchor(
        self,
        robot_idx: int,
        object_id: str,
        preferred_fixture_types: set[str] | None = None,
        require_placeable: bool = False,
    ) -> str:
        fixture_id = self._resolve_object_anchor_fixture(
            object_id,
            preferred_fixture_types=preferred_fixture_types,
            require_placeable=require_placeable,
        )
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
                "args": {"object_id": "hotdog_bun", "support_object_id": "plate"},
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
                "tool": "wait",
                "robot_idx": 0,
                "args": {},
            },
            {
                "tool": "place_on_object",
                "robot_idx": 1,
                "args": {"object_id": "sausage", "support_object_id": "plate"},
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
                "tool": "wait",
                "robot_idx": 1,
                "args": {},
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

    def _resolve_semantic_ref(self, ref: dict[str, Any]) -> Any:
        resolver = ref.get("$ref")
        if resolver == "source_fixture":
            preferred_fixture_types = ref.get("preferred_fixture_types")
            preferred_fixture_types_set = (
                set(preferred_fixture_types) if preferred_fixture_types is not None else None
            )
            return self._infer_source_fixture(
                ref["object_id"],
                preferred_fixture_types=preferred_fixture_types_set,
            )
        if resolver == "object_anchor_fixture":
            preferred_fixture_types = ref.get("preferred_fixture_types")
            preferred_fixture_types_set = (
                set(preferred_fixture_types) if preferred_fixture_types is not None else None
            )
            return self._resolve_object_anchor_fixture(
                ref["object_id"],
                preferred_fixture_types=preferred_fixture_types_set,
                require_placeable=bool(ref.get("require_placeable", False)),
            )
        raise ValueError(f"Unknown semantic resolver {resolver!r}")

    def _ground_value(self, value: Any) -> Any:
        if isinstance(value, dict):
            if "$ref" in value:
                return self._resolve_semantic_ref(value)
            return {key: self._ground_value(subvalue) for key, subvalue in value.items()}
        if isinstance(value, list):
            return [self._ground_value(item) for item in value]
        return value

    def ground_plan_template(self, tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
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

    def navigate_to_fixture(self, fixture_id: str, robot_idx: int = 0) -> ToolResult:
        self._require_fixture(fixture_id)
        self.runner._move_robot_near_fixture(robot_idx, fixture_id)
        self._sync_held_object(robot_idx)
        return ToolResult(
            tool_name="navigate_to_fixture",
            success=True,
            details={"fixture_id": fixture_id, "robot_idx": robot_idx},
        )

    def open_hinged_part(
        self,
        target_id: str,
        part_id: str,
        robot_idx: int = 0,
    ) -> ToolResult:
        fixture = self._require_fixture(target_id)
        self.runner._move_robot_near_fixture(robot_idx, target_id)
        if part_id == "hinged" and hasattr(fixture, "open_door"):
            fixture.open_door(env=self.env)
        else:
            joint_name = self._resolve_joint_name(fixture, part_id)
            self._set_named_joint(fixture, joint_name, 1.0)
        return ToolResult("open_hinged_part", True, {"target_id": target_id, "part_id": part_id})

    def close_hinged_part(
        self,
        target_id: str,
        part_id: str,
        robot_idx: int = 0,
    ) -> ToolResult:
        fixture = self._require_fixture(target_id)
        self.runner._move_robot_near_fixture(robot_idx, target_id)
        if part_id == "hinged" and hasattr(fixture, "close_door"):
            fixture.close_door(env=self.env)
        else:
            joint_name = self._resolve_joint_name(fixture, part_id)
            self._set_named_joint(fixture, joint_name, 0.0)
        return ToolResult("close_hinged_part", True, {"target_id": target_id, "part_id": part_id})

    def open_sliding_part(
        self,
        target_id: str,
        part_id: str,
        robot_idx: int = 0,
    ) -> ToolResult:
        fixture = self._require_fixture(target_id)
        self.runner._move_robot_near_fixture(robot_idx, target_id)
        joint_name = self._resolve_joint_name(fixture, part_id if part_id != "sliding" else "slide")
        self._set_named_joint(fixture, joint_name, 1.0)
        return ToolResult("open_sliding_part", True, {"target_id": target_id, "part_id": part_id})

    def close_sliding_part(
        self,
        target_id: str,
        part_id: str,
        robot_idx: int = 0,
    ) -> ToolResult:
        fixture = self._require_fixture(target_id)
        self.runner._move_robot_near_fixture(robot_idx, target_id)
        joint_name = self._resolve_joint_name(fixture, part_id if part_id != "sliding" else "slide")
        self._set_named_joint(fixture, joint_name, 0.0)
        return ToolResult("close_sliding_part", True, {"target_id": target_id, "part_id": part_id})

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
            raise ValueError(f"Object {object_id!r} is already held by robot {current_holder}")

        self.runner._move_robot_near_fixture(robot_idx, source_id)
        self._held_objects[robot_idx] = object_id
        self._sync_held_object(robot_idx)
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

        self.runner._move_robot_near_fixture(robot_idx, support_id)
        self.runner.move_object(object_id, support_id)
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
            self.runner._move_robot_near_fixture(robot_idx, receptacle_id)
            self.runner.move_object(object_id, receptacle_id)
        else:
            self._require_object(receptacle_id)
            self._place_on_object_center(object_id, receptacle_id)

        self._held_objects.pop(robot_idx, None)
        return ToolResult(
            "place_in_receptacle",
            True,
            {"object_id": object_id, "receptacle_id": receptacle_id, "robot_idx": robot_idx},
        )

    def place_on_object(
        self,
        object_id: str,
        support_object_id: str,
        robot_idx: int = 0,
    ) -> ToolResult:
        self._require_object(object_id)
        self._require_object(support_object_id)
        holder = self._held_by_robot(object_id)
        if holder not in {None, robot_idx}:
            raise ValueError(f"Object {object_id!r} is held by robot {holder}")

        self._move_robot_near_object_anchor(
            robot_idx,
            support_object_id,
            preferred_fixture_types={"dining_counter", "island", "counter_non_dining"},
            require_placeable=True,
        )
        self._place_on_object_center(object_id, support_object_id)
        self._held_objects.pop(robot_idx, None)
        return ToolResult(
            "place_on_object",
            True,
            {
                "object_id": object_id,
                "support_object_id": support_object_id,
                "robot_idx": robot_idx,
            },
        )

    def place_under_dispenser(
        self,
        object_id: str,
        dispenser_id: str,
        robot_idx: int = 0,
    ) -> ToolResult:
        self._require_object(object_id)
        fixture = self._require_fixture(dispenser_id)
        holder = self._held_by_robot(object_id)
        if holder not in {None, robot_idx}:
            raise ValueError(f"Object {object_id!r} is held by robot {holder}")

        self.runner._move_robot_near_fixture(robot_idx, dispenser_id)

        if isinstance(fixture, CoffeeMachine):
            site_name = f"{fixture.naming_prefix}receptacle_place_site"
            site_id = self.env.sim.model.site_name2id(site_name)
            target_pos = self.env.sim.data.site_xpos[site_id].copy()
            self._set_object_pose(object_id, target_pos)
        else:
            self.runner.move_object(object_id, dispenser_id)

        self._held_objects.pop(robot_idx, None)
        return ToolResult(
            "place_under_dispenser",
            True,
            {"object_id": object_id, "dispenser_id": dispenser_id, "robot_idx": robot_idx},
        )

    def press_button(
        self,
        target_id: str,
        control_id: str,
        robot_idx: int = 0,
    ) -> ToolResult:
        fixture = self._require_fixture(target_id)
        self.runner._move_robot_near_fixture(robot_idx, target_id)

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

        self.env.sim.forward()
        return ToolResult("press_button", True, {"target_id": target_id, "control_id": control_id})

    def press_lever(
        self,
        target_id: str,
        control_id: str,
        robot_idx: int = 0,
    ) -> ToolResult:
        fixture = self._require_fixture(target_id)
        self.runner._move_robot_near_fixture(robot_idx, target_id)

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

        self.env.sim.forward()
        return ToolResult("press_lever", True, {"target_id": target_id, "control_id": control_id})

    def set_rotary_control(
        self,
        target_id: str,
        control_id: str,
        goal: str,
        robot_idx: int = 0,
    ) -> ToolResult:
        fixture = self._require_fixture(target_id)
        self.runner._move_robot_near_fixture(robot_idx, target_id)

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
    parser.add_argument("--layout", type=int, default=11)
    parser.add_argument("--style", type=int, default=34)
    parser.add_argument("--seed", type=int, default=42)
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
            "Name of a built-in tool plan. Available: cooperative_hotdog_setup. "
            "Mutually exclusive with --plan."
        ),
    )
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--gl-backend", type=str, default="osmesa")
    args = parser.parse_args()

    if args.plan is not None and args.demo_plan is not None:
        parser.error("Use either --plan or --demo-plan, not both.")

    if args.task is None:
        if args.demo_plan is not None:
            demo_key = args.demo_plan.strip().lower().replace("-", "_")
            task_name = SimToolExecutor._DEMO_TASK_BY_NAME.get(demo_key)
            if task_name is None:
                parser.error(
                    "Unknown demo plan. Available: cooperative_hotdog_setup"
                )
        else:
            task_name = "MicrowaveThawing"
    else:
        task_name = args.task

    executor = SimToolExecutor(
        task_name=task_name,
        robots=args.robots,
        layout=args.layout,
        style=args.style,
        seed=args.seed,
        render_width=args.width,
        render_height=args.height,
        gl_backend=args.gl_backend,
    )

    try:
        if args.plan is None and args.demo_plan is None:
            saved = executor.save_scene_frames(args.output_dir, prefix="initial")
            print(json.dumps({k: str(v) for k, v in saved.items()}, indent=2))
        else:
            if args.demo_plan is not None:
                tool_calls = executor.build_demo_plan(args.demo_plan)
            else:
                with open(args.plan, "r") as f:
                    tool_calls = json.load(f)
            metadata = executor.run_tool_plan(
                tool_calls=tool_calls,
                output_dir=args.output_dir,
                fps=args.fps,
            )
            print(json.dumps(metadata, indent=2))
    finally:
        executor.close()


__all__ = ["SimToolExecutor", "ToolResult"]


if __name__ == "__main__":
    _main()
