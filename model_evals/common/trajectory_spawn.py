"""Trajectory lookup and robot placement for RoboCasa VLA evaluation."""

from __future__ import annotations

import json
import logging
import pathlib
import re

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_TRAJECTORY_INIT_ROOT = (
    "/home/dorian/Projects/robocasa/data_generation/task_level/data/diversity_analysis/"
    "sampling_methods_data_52Tasks_30Trajectories/sampling_methods_data_52Tasks_30Trajectories/"
    "verbalized"
)


# ---------------------------------------------------------------------------
# Trajectory-based initialization
# ---------------------------------------------------------------------------

def load_trajectory_init(path: str | None) -> dict | None:
    if path is None:
        return None
    with open(path, "r") as f:
        return json.load(f)


def camel_to_snake(name: str) -> str:
    with_underscores = re.sub(r"(?<!^)(?=[A-Z])", "_", name)
    return with_underscores.lower()


def trajectory_path_for_env(root: str | None, env_name: str, trajectory_index: int) -> str | None:
    if root is None:
        return None

    root_path = pathlib.Path(root)
    trajectory_name = f"traj_{trajectory_index:06d}.json"
    direct_names = [
        camel_to_snake(env_name),
        env_name.lower(),
        env_name.replace("_", "").lower(),
    ]
    for task_dir_name in direct_names:
        candidate = root_path / task_dir_name / "trajectories" / trajectory_name
        if candidate.exists():
            return str(candidate)

    normalized_env = env_name.replace("_", "").lower()
    for candidate in root_path.glob(f"*/trajectories/{trajectory_name}"):
        try:
            with open(candidate, "r") as f:
                composite_task = json.load(f).get("composite_task", "")
        except Exception:
            continue
        if str(composite_task).replace("_", "").lower() == normalized_env:
            return str(candidate)

    return None


def load_trajectory_for_env(
    *,
    env_name: str,
    trajectory_json: str | None,
    trajectory_root: str | None,
    trajectory_index: int,
) -> tuple[dict | None, str | None]:
    path = trajectory_json or trajectory_path_for_env(trajectory_root, env_name, trajectory_index)
    if path is None:
        return None, None
    return load_trajectory_init(path), path


def list_trajectory_task_jobs(root: str, trajectory_indices: list[int]) -> list[dict]:
    jobs = []
    root_path = pathlib.Path(root)
    for task_dir in sorted(path for path in root_path.iterdir() if path.is_dir()):
        for trajectory_index in trajectory_indices:
            trajectory_path = task_dir / "trajectories" / f"traj_{trajectory_index:06d}.json"
            if not trajectory_path.exists():
                logger.warning("Missing trajectory file, skipping: %s", trajectory_path)
                continue
            trajectory = load_trajectory_init(str(trajectory_path))
            env_name = trajectory.get("composite_task")
            if not env_name:
                logger.warning("Trajectory has no composite_task, skipping: %s", trajectory_path)
                continue
            jobs.append(
                {
                    "env_name": env_name,
                    "trajectory_index": trajectory_index,
                    "trajectory_path": str(trajectory_path),
                    "trajectory_init": trajectory,
                    "log_label": f"traj_{trajectory_index:06d}",
                }
            )
    return jobs


def unwrap_robocasa_env(env):
    cur = env
    while hasattr(cur, "env"):
        next_env = cur.env
        if next_env is cur:
            break
        if next_env.__class__.__module__.startswith("robocasa.environments."):
            return next_env
        cur = next_env
    return cur


def unwrap_robocasa_gym_env(env):
    cur = env
    while hasattr(cur, "env"):
        if cur.__class__.__module__.startswith("robocasa.wrappers.gym_wrapper"):
            return cur
        next_env = cur.env
        if next_env is cur:
            break
        cur = next_env
    return env if hasattr(env, "get_observation") else None


def refresh_observation(env):
    inner_env = unwrap_robocasa_env(env)
    raw_obs = inner_env._get_observations(force_update=True)
    gym_env = unwrap_robocasa_gym_env(env)
    if gym_env is not None:
        return gym_env.get_observation(raw_obs)
    return raw_obs


def fixture_type_matches(fixture, fixture_type: str) -> bool:
    normalized_type = fixture_type.replace("_", "").lower()
    fixture_name = getattr(fixture, "name", "").replace("_", "").lower()
    class_name = type(fixture).__name__.replace("_", "").lower()
    return normalized_type in fixture_name or normalized_type in class_name


def object_cfg_matches(cfg: dict, object_symbol: str, trajectory: dict) -> bool:
    if cfg.get("name") == object_symbol:
        return True

    object_type = (
        trajectory.get("initial_state", {})
        .get("objects", {})
        .get(object_symbol, {})
        .get("object_type")
    )
    if object_type is None:
        return False

    obj_groups = cfg.get("obj_groups", ())
    if isinstance(obj_groups, str):
        obj_groups = (obj_groups,)
    return cfg.get("name") == object_type or object_type in obj_groups


def resolve_fixture_from_object_cfg(env, trajectory: dict, object_symbol: str):
    inner_env = unwrap_robocasa_env(env)
    for cfg in getattr(inner_env, "object_cfgs", []):
        if not object_cfg_matches(cfg, object_symbol, trajectory):
            continue
        placement = cfg.get("placement", {})
        fixture = placement.get("fixture")
        if fixture is None:
            return None
        if hasattr(fixture, "pos"):
            return fixture
        try:
            resolved = inner_env.get_fixture(fixture)
            if resolved is not None:
                return resolved
        except Exception:
            return None
    return None


def resolve_trajectory_fixture(env, trajectory: dict, symbolic_fixture_id: str):
    inner_env = unwrap_robocasa_env(env)

    fixture = getattr(inner_env, symbolic_fixture_id, None)
    if fixture is not None and hasattr(fixture, "pos"):
        return fixture

    try:
        fixture = inner_env.get_fixture(symbolic_fixture_id)
        if fixture is not None:
            return fixture
    except Exception:
        pass

    symbols = trajectory.get("grounding_map", {}).get("symbols", {})
    symbol = symbols.get(symbolic_fixture_id, {})

    object_symbol = symbol.get("object_symbol")
    if object_symbol is not None:
        fixture = resolve_fixture_from_object_cfg(env, trajectory, object_symbol)
        if fixture is not None:
            return fixture

    fixture_type = symbol.get("fixture_type")
    if fixture_type is not None:
        for fixture in inner_env.fixtures.values():
            if fixture_type_matches(fixture, fixture_type):
                return fixture

    preferred_types = symbol.get("preferred_fixture_types", [])
    for preferred_type in preferred_types:
        for fixture in inner_env.fixtures.values():
            if fixture_type_matches(fixture, preferred_type):
                return fixture

    available = sorted(str(name) for name in inner_env.fixtures.keys())
    raise ValueError(
        f"Could not resolve trajectory fixture '{symbolic_fixture_id}'. "
        f"Available RoboCasa fixtures: {available}"
    )


def get_robot_position(env, robot_idx: int) -> np.ndarray:
    inner_env = unwrap_robocasa_env(env)
    body_name = f"mobilebase{robot_idx}_base"
    body_id = inner_env.sim.model.body_name2id(body_name)
    return inner_env.sim.data.body_xpos[body_id].copy()


def set_robot_yaw(env, robot_idx: int, yaw: float | None):
    inner_env = unwrap_robocasa_env(env)
    yaw_jnt = f"mobilebase{robot_idx}_joint_mobile_yaw"
    try:
        addr = inner_env.sim.model.get_joint_qpos_addr(yaw_jnt)
        if yaw is None:
            inner_env.sim.data.qpos[addr] = 0.0
        else:
            anchor_ori = getattr(
                inner_env, "init_robot_base_ori_anchors", [None] * (robot_idx + 1)
            )[robot_idx]
            inner_env.sim.data.qpos[addr] = yaw - anchor_ori[2] if anchor_ori is not None else yaw
        inner_env.sim.forward()
    except Exception:
        logger.warning("Unable to set robot%d yaw; continuing with existing yaw.", robot_idx)


def set_robot_pose(env, robot_idx: int, pos_xy: np.ndarray, yaw: float | None):
    from robocasa.utils import env_utils as EnvUtils

    inner_env = unwrap_robocasa_env(env)
    set_robot_yaw(env, robot_idx, yaw)
    pos_3d = np.array([pos_xy[0], pos_xy[1], 0.0])
    EnvUtils.set_robot_to_position(inner_env, pos_3d, robot_idx=robot_idx)

    actual = get_robot_position(env, robot_idx)[:2]
    error = np.asarray(pos_xy, dtype=float)[:2] - actual
    if np.linalg.norm(error) > 0.01:
        corrected = pos_3d.copy()
        corrected[:2] += error
        EnvUtils.set_robot_to_position(inner_env, corrected, robot_idx=robot_idx)
    inner_env.sim.forward()


def placement_fixture_candidates(env, fixture) -> list:
    inner_env = unwrap_robocasa_env(env)
    candidates = [fixture]
    if not hasattr(fixture, "pos"):
        return candidates

    target_pos = np.asarray(fixture.pos[:2], dtype=float)
    support_keywords = (
        "counter", "island", "cabinet", "stack", "drawer", "fridge", "dishwasher",
        "sink", "stove", "oven", "microwave", "table",
    )
    nearby = []
    for other in getattr(inner_env, "fixtures", {}).values():
        if other is fixture or not hasattr(other, "pos"):
            continue
        name = getattr(other, "name", "").lower()
        if not any(keyword in name for keyword in support_keywords):
            continue
        dist = float(np.linalg.norm(np.asarray(other.pos[:2], dtype=float) - target_pos))
        nearby.append((dist, other))
    candidates.extend(other for _, other in sorted(nearby, key=lambda item: item[0])[:8])

    seen = set()
    unique = []
    for candidate in candidates:
        key = id(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def place_robot_near_fixture(env, robot_idx: int, fixture):
    from robocasa.utils.occupancy_grid import OccupancyGrid

    inner_env = unwrap_robocasa_env(env)
    fixtures = dict(inner_env.fixtures)
    occupancy_grid = OccupancyGrid(fixtures)

    robot_cells = []
    robot_positions = []
    for other_idx in range(len(inner_env.robots)):
        if other_idx == robot_idx:
            continue
        other_pos = get_robot_position(env, other_idx)[:2]
        robot_cells.append(occupancy_grid._world_to_grid(other_pos))
        robot_positions.append(other_pos)

    requested_fixture = getattr(fixture, "name", str(fixture))
    attempted = []
    for candidate_fixture in placement_fixture_candidates(env, fixture):
        candidate_name = getattr(candidate_fixture, "name", str(candidate_fixture))
        result = occupancy_grid.find_placement(
            candidate_fixture,
            robot_cells,
            ref_object_pos=None,
            robot_positions=robot_positions,
            require_front=False,
        )
        attempted.append(candidate_name)
        if result is None:
            continue

        pos_xy, yaw = result
        set_robot_pose(env, robot_idx, pos_xy, yaw)
        placed_pos = get_robot_position(env, robot_idx)
        return {
            "robot_idx": robot_idx,
            "fixture": candidate_name,
            "requested_fixture": requested_fixture,
            "position": np.asarray(placed_pos).tolist(),
            "yaw": None if yaw is None else float(yaw),
            "placement_backend": "trajectory_runner_occupancy_grid",
            "placement_attempted_fixtures": attempted,
        }

    raise ValueError(
        f"Could not find trajectory-style placement near fixture '{requested_fixture}' "
        f"for robot{robot_idx}. Tried: {attempted}"
    )


def object_cfg_fixture(env, cfg: dict):
    inner_env = unwrap_robocasa_env(env)
    fixture = cfg.get("placement", {}).get("fixture")
    if fixture is None:
        return None
    if hasattr(fixture, "pos"):
        return fixture
    try:
        return inner_env.get_fixture(fixture)
    except Exception:
        return None


def object_cfg_text(cfg: dict) -> str:
    groups = cfg.get("obj_groups", ())
    if isinstance(groups, str):
        groups = (groups,)
    return " ".join(str(x) for x in (cfg.get("name"), *groups) if x).lower()


def unique_fixture_candidates(candidates: list[tuple[object, str, str]]) -> list[tuple[object, str, str]]:
    seen = set()
    unique = []
    for fixture, source, detail in candidates:
        if fixture is None or not hasattr(fixture, "pos"):
            continue
        key = id(fixture)
        if key in seen:
            continue
        seen.add(key)
        unique.append((fixture, source, detail))
    return unique


def fixture_candidates_from_task_keywords(env_name: str, env) -> list[tuple[object, str, str]]:
    inner_env = unwrap_robocasa_env(env)
    name = camel_to_snake(env_name).lower()
    keyword_attrs = [
        ("dishwasher", ("dishwasher",)),
        ("sink", ("sink", "basin")),
        ("coffee_machine", ("coffee",)),
        ("microwave", ("microwave",)),
        ("oven", ("oven",)),
        ("toaster_oven", ("toaster_oven",)),
        ("toaster", ("toast", "toaster")),
        ("stove", ("stove", "heat", "searing", "stir", "kettle", "boil")),
        ("cabinet", ("cabinet", "stack_bowls")),
        ("drawer", ("drawer", "cutting_station")),
        ("fridge", ("fridge", "lettuce", "leftovers")),
        ("dining_counter", ("serve", "straw", "table")),
        ("counter", ("cutting", "station", "prepare", "setup")),
    ]
    candidates = []
    for attr, keywords in keyword_attrs:
        if not any(keyword in name for keyword in keywords):
            continue
        fixture = getattr(inner_env, attr, None)
        if fixture is not None and hasattr(fixture, "pos"):
            candidates.append((fixture, "fallback_final_target_fixture", attr))
    return candidates


def choose_fallback_fixture(env_name: str, env):
    inner_env = unwrap_robocasa_env(env)
    object_cfgs = list(getattr(inner_env, "object_cfgs", []) or [])
    candidates: list[tuple[object, str, str]] = []

    candidates.extend(fixture_candidates_from_task_keywords(env_name, env))
    target_keywords = (
        "receptacle", "container", "tupperware", "plate", "bowl", "cup", "glass",
        "mug", "tray", "rack", "dish", "pan", "pot", "board", "basket", "platter",
        "serving", "dining", "sink", "dishwasher",
    )
    for cfg in object_cfgs:
        text = object_cfg_text(cfg)
        if any(keyword in text for keyword in target_keywords):
            candidates.append(
                (object_cfg_fixture(env, cfg), "fallback_final_target_fixture", cfg.get("name", text))
            )

    for cfg in object_cfgs:
        if cfg.get("init_robot_here") is True:
            candidates.append(
                (object_cfg_fixture(env, cfg), "fallback_primary_source_fixture", cfg.get("name", "init_robot_here"))
            )
    for cfg in object_cfgs:
        candidates.append(
            (object_cfg_fixture(env, cfg), "fallback_primary_source_fixture", cfg.get("name", "object_cfg"))
        )

    init_ref = getattr(inner_env, "init_robot_base_ref", None)
    if init_ref is not None:
        candidates.append((init_ref, "fallback_init_robot_base_ref", getattr(init_ref, "name", "init_robot_base_ref")))

    robot0_pos = get_robot_position(env, 0)[:2]
    nearest_fixture = None
    nearest_dist = float("inf")
    for fixture in getattr(inner_env, "fixtures", {}).values():
        if not hasattr(fixture, "pos"):
            continue
        dist = float(np.linalg.norm(np.asarray(fixture.pos[:2], dtype=float) - robot0_pos))
        if dist < nearest_dist:
            nearest_fixture = fixture
            nearest_dist = dist
    candidates.append((nearest_fixture, "fallback_nearest_robot0", f"distance={nearest_dist:.3f}"))

    work_surface_keywords = ("counter", "island", "dining")
    surface_fixtures = [
        fixture for fixture in getattr(inner_env, "fixtures", {}).values()
        if hasattr(fixture, "pos")
        and any(keyword in getattr(fixture, "name", "").lower() for keyword in work_surface_keywords)
    ]
    if surface_fixtures:
        positions = np.stack([np.asarray(fixture.pos[:2], dtype=float) for fixture in surface_fixtures])
        center = positions.mean(axis=0)
        central = min(
            surface_fixtures,
            key=lambda fixture: float(np.linalg.norm(np.asarray(fixture.pos[:2], dtype=float) - center)),
        )
        candidates.append((central, "fallback_central_work_surface", getattr(central, "name", "surface")))

    return unique_fixture_candidates(candidates)


def apply_fallback_initialization(env_name: str, env, robot_idx: int = 1) -> list[dict]:
    errors = []
    for fixture, source, detail in choose_fallback_fixture(env_name, env):
        try:
            placement = place_robot_near_fixture(env, robot_idx, fixture)
        except Exception as exc:
            errors.append({
                "source": source,
                "detail": detail,
                "fixture": getattr(fixture, "name", str(fixture)),
                "error": str(exc),
            })
            continue
        placement.update({
            "agent_id": f"fallback_robot_{robot_idx}",
            "symbolic_fixture": detail,
            "spawn_source": source,
            "fallback_attempt_errors": errors,
        })
        return [placement]
    raise ValueError(f"Fallback spawning failed for {env_name}. Attempts: {errors}")


def apply_trajectory_initialization(
    env,
    trajectory: dict | None,
    mode: str,
    other_agent: str,
) -> list[dict]:
    if trajectory is None or mode == "none":
        return []

    agents = trajectory.get("initial_state", {}).get("agents", {})
    if not isinstance(agents, dict):
        raise ValueError("Trajectory initial_state.agents must be an object.")

    if mode == "passive_other":
        agent_to_robot = {other_agent: 1}
    elif mode == "all_agents":
        agent_to_robot = {agent_id: idx for idx, agent_id in enumerate(sorted(agents))}
    else:
        raise ValueError(f"Unsupported trajectory init mode: {mode}")

    placements = []
    robot_count = len(unwrap_robocasa_env(env).robots)
    for agent_id, robot_idx in agent_to_robot.items():
        if robot_idx >= robot_count:
            raise ValueError(
                f"Cannot place {agent_id} on robot{robot_idx}; env only has {robot_count} robot(s)."
            )
        agent_state = agents.get(agent_id)
        if not isinstance(agent_state, dict) or "location" not in agent_state:
            raise ValueError(f"Trajectory does not define initial location for {agent_id}.")
        symbolic_fixture_id = agent_state["location"]
        fixture = resolve_trajectory_fixture(env, trajectory, symbolic_fixture_id)
        placement = place_robot_near_fixture(env, robot_idx, fixture)
        placement.update({
            "agent_id": agent_id,
            "symbolic_fixture": symbolic_fixture_id,
            "spawn_source": "trajectory",
        })
        placements.append(placement)

    return placements
