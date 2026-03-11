"""
Trajectory runner for multi-agent task visualization.

Builds a Kitchen environment, extracts the scene description (fixtures, objects),
and executes trajectories defined as sequences of three primitives:

  - move_object(object, from, to)
  - interact(fixture, action, ...)
  - communicate(to, message)

Each step produces before/after visual observations from multiple cameras.

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
from robosuite.controllers import load_composite_controller_config
from robosuite.environments.base import make

import robocasa  # noqa: F401  — registers envs
from robocasa.models.fixtures import FixtureType
from robocasa.models.fixtures.fixture import Fixture
from robocasa.models.fixtures.fixture_utils import fixture_is_type


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


def _classify_fixture(fixture: Fixture) -> str | None:
    """Return the most specific FixtureType name for a fixture, or None."""
    # Check specific types before general ones to get the most useful label.
    # Order matters: e.g. DINING_COUNTER before COUNTER, CABINET_SINGLE_DOOR before CABINET.
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
class TrajectoryResult:
    """Full result from running a trajectory."""
    initial_obs: dict[str, np.ndarray]
    steps: list[StepResult]
    scene: dict  # the scene description used

    def save(self, output_dir: str | Path, format: str = "png"):
        """Save all images and the scene description."""
        import imageio

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Initial observations
        for cam_name, img in self.initial_obs.items():
            imageio.imwrite(str(output_dir / f"initial_{cam_name}.{format}"), img)
        # Per-step
        for step in self.steps:
            step.save_images(output_dir, format=format)
        # Scene metadata
        with open(output_dir / "scene.json", "w") as f:
            json.dump(self.scene, f, indent=2)


# ---------------------------------------------------------------------------
# TrajectoryRunner
# ---------------------------------------------------------------------------

class TrajectoryRunner:
    """
    Builds a Kitchen env, extracts scene info, and executes trajectories
    by teleporting objects and toggling fixture states.

    Args:
        task_name: RoboCasa env name (default "Kitchen" for bare scene).
        robots: Number of robots (1 or 2).
        layout: Kitchen layout ID.
        style: Kitchen style ID.
        seed: RNG seed for reproducibility.
        camera_names: Cameras to render. None = auto-detect.
        render_width: Image width in pixels.
        render_height: Image height in pixels.
        gl_backend: MuJoCo GL backend ("osmesa", "egl", "glfw").
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
    ):
        os.environ.setdefault("MUJOCO_GL", gl_backend)

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
        self.env.reset()

        self.render_width = render_width
        self.render_height = render_height

        # Discover cameras
        if camera_names is None:
            self.camera_names = self._discover_cameras()
        else:
            self.camera_names = camera_names

        # Build fixture index: fixture_id -> Fixture object
        self._fixtures: dict[str, Fixture] = dict(self.env.fixtures)

        # Build scene description (cached)
        self._scene: dict | None = None

        # Track object locations: object_id -> fixture_id
        self._object_locations: dict[str, str] = {}

    def _discover_cameras(self) -> list[str]:
        """Find useful cameras in the scene."""
        all_cams = [
            self.env.sim.model.camera_id2name(i)
            for i in range(self.env.sim.model.ncam)
        ]
        # Filter to robot views and agentviews
        useful_prefixes = ("robot0_", "robot1_", "agentview")
        useful = [c for c in all_cams if any(c.startswith(p) for p in useful_prefixes)]
        # Always include a room-level view if available
        for fallback in ("kitchen_overview", "frontview"):
            if fallback in all_cams:
                useful.append(fallback)
                break
        return useful if useful else all_cams[:4]

    # ------------------------------------------------------------------
    # Scene description
    # ------------------------------------------------------------------

    def get_scene_description(self) -> dict:
        """
        Extract a JSON-serializable description of the scene for the LLM.

        Returns dict with:
          - fixtures: {fixture_id: {type, position, interactions, can_place_objects, nearby}}
          - objects: {object_id: {type, location}}
          - cameras: [camera_name, ...]
          - robots: [{robot_id, position}, ...]
        """
        if self._scene is not None:
            return self._scene

        fixtures_info = {}
        fixture_positions = {}

        for name, fxtr in self._fixtures.items():
            ftype = _classify_fixture(fxtr)
            if ftype is None:
                continue  # skip unclassifiable fixtures (walls, floors, etc.)

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
                location = self._find_object_fixture(obj_name, obj_pos)

                # Extract category from ep_meta object_cfgs
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

    def _find_object_fixture(self, obj_name: str, obj_pos: np.ndarray) -> str:
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

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self) -> dict[str, np.ndarray]:
        """Render all configured cameras and return {camera_name: image}."""
        images = {}
        for cam_name in self.camera_names:
            try:
                frame = self.env.sim.render(
                    height=self.render_height,
                    width=self.render_width,
                    camera_name=cam_name,
                )[::-1]  # flip vertical (MuJoCo convention)
                images[cam_name] = frame
            except Exception:
                continue  # skip cameras that fail to render
        return images

    # ------------------------------------------------------------------
    # Primitives
    # ------------------------------------------------------------------

    def move_object(self, object_id: str, from_fixture: str, to_fixture: str):
        """Teleport an object from one fixture's surface to another's."""
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

        # Sample a position on the target fixture's surface
        try:
            region = target_fxtr.sample_reset_region(env=self.env)
        except Exception:
            # Fallback: use fixture center position, slightly above
            region = {
                "offset": (0.0, 0.0, 0.0),
                "size": (0.1, 0.1),
            }

        # Compute world position from fixture-relative offset
        fxtr_pos = target_fxtr.pos
        offset = np.array(region["offset"])

        # Rotate offset by fixture orientation
        if hasattr(target_fxtr, "rot") and target_fxtr.rot is not None:
            angle = target_fxtr.rot
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rot_offset = np.array([
                cos_a * offset[0] - sin_a * offset[1],
                sin_a * offset[0] + cos_a * offset[1],
                offset[2],
            ])
        else:
            rot_offset = offset

        target_pos = fxtr_pos + rot_offset
        # Small upward offset to place on surface
        target_pos[2] += 0.02

        obj = self.env.objects[object_id]
        # Keep current orientation
        current_qpos = self.env.sim.data.get_joint_qpos(obj.joints[0])
        current_quat = current_qpos[3:7]

        self.env.sim.data.set_joint_qpos(
            obj.joints[0],
            np.concatenate([target_pos, current_quat]),
        )
        self.env.sim.forward()

        self._object_locations[object_id] = to_fixture

    def interact(self, fixture_id: str, action: str, **kwargs):
        """
        Change a fixture's state.

        Supported actions:
          - "open": open door/drawer
          - "close": close door/drawer
          - "turn_on": activate appliance (set joint to max)
          - "turn_off": deactivate appliance (set joint to min)
        """
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
            # Set all non-door joints to max
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

        Args:
            trajectory: Either a list of step dicts, or a full trajectory dict
                        with a "steps" key. Each step must have:
                          - action: "move_object" | "interact" | "communicate"
                          - args: dict of arguments for the action

        Returns:
            TrajectoryResult with initial observations and per-step before/after images.
        """
        if isinstance(trajectory, dict):
            steps = trajectory.get("steps", [])
        else:
            steps = trajectory

        scene = self.get_scene_description()

        # Reset env to get clean initial state
        self.env.reset()
        self.env.sim.forward()

        initial_obs = self.render()
        step_results = []

        for i, step in enumerate(steps):
            action = step.get("action") or step.get("tool_name", "")
            args = step.get("args") or step.get("tool_args", {})

            # Render before
            before = self.render()

            # Execute primitive
            if action == "move_object":
                self.move_object(
                    object_id=args["object"],
                    from_fixture=args["from"],
                    to_fixture=args["to"],
                )
            elif action == "interact":
                fixture_id = args.get("fixture") or args.get("fixture_id")
                act = args.get("action")
                if fixture_id is None or act is None:
                    raise ValueError(
                        f"Step {i}: 'interact' requires 'fixture' and 'action' in args"
                    )
                extra = {
                    k: v for k, v in args.items()
                    if k not in ("fixture", "fixture_id", "action")
                }
                self.interact(fixture_id, act, **extra)
            elif action == "communicate":
                pass  # no sim change, just capture the state
            else:
                raise ValueError(
                    f"Step {i}: unknown action '{action}'. "
                    f"Expected: move_object, interact, communicate"
                )

            # Render after
            after = self.render()

            step_results.append(StepResult(
                step_index=i,
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

    def __exit__(self, *args):
        self.close()
