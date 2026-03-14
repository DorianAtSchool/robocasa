#!/usr/bin/env python3
"""
Compose single-robot atomic demos into a 2-robot trajectory.

Takes two lerobot dataset demos (one per robot), creates a 2-robot Kitchen env
with the shared layout, replays both demos simultaneously with actions, and
saves video output.

Usage:
  # Auto-discover a matching pair from available datasets:
  python experiments/compose_demos.py

  # Manually specify demos:
  python experiments/compose_demos.py \
    --demo-a-dataset /path/to/dataset_a/lerobot \
    --demo-a-episode 18 \
    --demo-b-dataset /path/to/dataset_b/lerobot \
    --demo-b-episode 9

The two demos should come from the same layout_id for fixture alignment.
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")

import imageio
import numpy as np
from robosuite.controllers import load_composite_controller_config
from robosuite.environments.base import make

import robocasa  # noqa: F401
import robocasa.macros as macros
import robocasa.utils.camera_utils as CamUtils
import robocasa.utils.env_utils as EnvUtils
import robocasa.utils.lerobot_utils as LU
from robocasa.wrappers.enclosing_wall_render_wrapper import EnclosingWallRenderWrapper

# PandaOmron has 13 qpos entries: 3 base (fwd, side, yaw) + 1 torso + 7 arm + 2 gripper
# In the env, these are split across separate index arrays:
#   _ref_base_joint_pos_indexes (3), _ref_torso_joint_pos_indexes (1),
#   _ref_joint_pos_indexes (7 arm), _ref_gripper_joint_pos_indexes['right'] (2)
PANDAOMRON_QPOS_COUNT = 13


def parse_args():
    parser = argparse.ArgumentParser(description="Compose two single-robot demos into a 2-robot trajectory")
    parser.add_argument("--demo-a-dataset", type=str, default=None, help="Lerobot dataset path for robot 0")
    parser.add_argument("--demo-a-episode", type=int, default=None, help="Episode index for robot 0")
    parser.add_argument("--demo-b-dataset", type=str, default=None, help="Lerobot dataset path for robot 1")
    parser.add_argument("--demo-b-episode", type=int, default=None, help="Episode index for robot 1")
    parser.add_argument("--layout", type=int, default=None, help="Kitchen layout ID (auto-detected from demos if omitted)")
    parser.add_argument("--style", type=int, default=None, help="Kitchen style ID (auto-detected or defaults to 34)")
    parser.add_argument("--seed", type=int, default=None, help="Env seed")
    parser.add_argument("--output-dir", type=str, default="experiments/composed_outputs")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--video-skip", type=int, default=1, help="Save every Nth frame to video")
    parser.add_argument(
        "--cameras",
        type=str,
        nargs="*",
        default=["robot0_agentview_center", "robot1_agentview_center"],
        help="Camera names to render",
    )
    parser.add_argument(
        "--gl",
        type=str,
        default="osmesa",
        choices=["osmesa", "egl", "glfw"],
        help="MuJoCo OpenGL backend",
    )
    return parser.parse_args()


def get_dataset_base_path():
    """Get the base path for robocasa datasets."""
    if macros.DATASET_BASE_PATH is not None:
        return Path(macros.DATASET_BASE_PATH)
    return Path(robocasa.__path__[0]).parent.absolute() / "datasets"


def find_available_demos():
    """Scan available atomic task datasets and index demos by layout_id.

    Returns:
        dict: layout_id -> list of (task_name, ep_num, style_id, dataset_path)
    """
    base = get_dataset_base_path() / "v1.0" / "pretrain" / "atomic"
    if not base.exists():
        return {}

    layout_index = defaultdict(list)
    for task_dir in sorted(base.iterdir()):
        if not task_dir.is_dir():
            continue
        lerobot_dirs = list(task_dir.glob("*/lerobot"))
        if not lerobot_dirs:
            continue
        ds = lerobot_dirs[0]
        extras = ds / "extras"
        if not extras.exists():
            continue
        for ep_dir in sorted(extras.glob("episode_*")):
            meta_path = ep_dir / "ep_meta.json"
            if not meta_path.exists():
                continue
            with open(meta_path) as f:
                meta = json.load(f)
            layout_id = meta.get("layout_id")
            style_id = meta.get("style_id")
            ep_num = int(ep_dir.name.split("_")[1])
            layout_index[layout_id].append((task_dir.name, ep_num, style_id, str(ds)))

    return layout_index


def _layout_has_enclosing_walls(layout_id):
    """Check if a layout has enclosing walls that the render wrapper can hide."""
    return len(EnclosingWallRenderWrapper._get_enclosing_wall_names_from_layout(layout_id)) > 0


def auto_select_demo_pair(layout_index):
    """Find two demos from different tasks that share the same layout_id.

    Prefers layouts with enclosing walls (so the wrapper can make them
    transparent for room-view rendering), then by task diversity.

    Returns:
        tuple: (demo_a_info, demo_b_info, layout_id) where each info is
               (task_name, ep_num, style_id, dataset_path)
    """
    # Score each layout: (has_enclosing_walls, num_tasks)
    candidates = []
    for layout_id, entries in layout_index.items():
        task_names = set(e[0] for e in entries)
        if len(task_names) < 2:
            continue
        has_walls = _layout_has_enclosing_walls(layout_id)
        candidates.append((has_walls, len(task_names), layout_id))

    if not candidates:
        return None

    # Sort: prefer enclosing walls, then most task diversity
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best_layout = candidates[0][2]

    entries = layout_index[best_layout]
    # Group by task
    by_task = defaultdict(list)
    for entry in entries:
        by_task[entry[0]].append(entry)

    # Pick first two different tasks, first demo from each
    tasks = sorted(by_task.keys())
    demo_a = by_task[tasks[0]][0]
    demo_b = by_task[tasks[1]][0]
    return demo_a, demo_b, best_layout


def load_demo(dataset_dir, ep_num):
    """Load a single demo's states, actions, metadata, and model XML."""
    dataset_dir = Path(dataset_dir)
    states = LU.get_episode_states(dataset_dir, ep_num)
    actions = LU.get_episode_actions(dataset_dir, ep_num)
    ep_meta = LU.get_episode_meta(dataset_dir, ep_num)
    model_xml = LU.get_episode_model_xml(dataset_dir, ep_num)
    return {
        "states": states,
        "actions": actions,
        "ep_meta": ep_meta,
        "model_xml": model_xml,
    }


def extract_robot_qpos_from_state(state_flat):
    """Extract robot joint positions from a single-robot flattened MuJoCo state.

    Flattened state: [time(1), qpos(nq), qvel(nv)]
    Robot joints are first in qpos since the robot is the first body in the XML.
    """
    return state_flat[1 : 1 + PANDAOMRON_QPOS_COUNT].copy()


def build_demo_joint_map(model_xml, state_flat):
    """Extract a {joint_name: qpos_value(s)} mapping from a demo's XML and state.

    Parses the demo's model XML to find all joint names and their types,
    then reads their values from the flattened state vector. This allows
    transferring fixture/object states to a different env by joint name.

    Returns:
        dict: {joint_name: np.array of qpos values}
    """
    import xml.etree.ElementTree as ET

    root = ET.fromstring(model_xml)

    # Build ordered list of (joint_name, qpos_size) from XML
    # MuJoCo qpos sizes: hinge=1, slide=1, ball=4, free=7
    type_to_size = {"hinge": 1, "slide": 1, "ball": 4, "free": 7}
    joints_info = []
    for j in root.iter("joint"):
        name = j.get("name")
        if name is None:
            continue
        jtype = j.get("type", "hinge")
        size = type_to_size.get(jtype, 1)
        joints_info.append((name, size))

    # state_flat = [time(1), qpos(nq), qvel(nv)]
    qpos_start = 1
    joint_map = {}
    offset = qpos_start
    for name, size in joints_info:
        joint_map[name] = state_flat[offset : offset + size].copy()
        offset += size

    return joint_map


def map_demo_action_to_robot(demo_action, robot):
    """Map a demo action (HDF5 ordering) to the robot's expected action vector.

    Demo action HDF5 format (up to 12D):
      [ee_pos(3), ee_rot(3), gripper(1), base_motion(4), control_mode(1)]

    Robot action format (from _action_split_indexes):
      Varies by controller config, typically:
      right(6), torso(1), base(2-3), right_gripper(1)
    """
    robot_action = np.zeros(robot.action_dim)
    splits = robot._action_split_indexes

    # Arm: demo[0:6] -> 'right' part
    if "right" in splits:
        start, end = splits["right"]
        arm_dim = min(6, end - start)
        robot_action[start : start + arm_dim] = demo_action[:arm_dim]

    # Gripper: demo[6] -> 'right_gripper' part
    if "right_gripper" in splits and len(demo_action) > 6:
        start, end = splits["right_gripper"]
        robot_action[start] = demo_action[6]

    # Base: demo[7:11] -> 'base' part (if demo has base actions)
    if "base" in splits and len(demo_action) > 7:
        start, end = splits["base"]
        base_dim = min(len(demo_action) - 7, end - start)
        robot_action[start : start + base_dim] = demo_action[7 : 7 + base_dim]

    # Torso: leave at 0 (demos typically don't include torso control)

    return robot_action


def idle_action(robot):
    """Return a zero action vector (robot holds position)."""
    return np.zeros(robot.action_dim)


def main():
    args = parse_args()
    os.environ["MUJOCO_GL"] = args.gl

    # --- Resolve demo sources ---
    manual_mode = args.demo_a_dataset is not None and args.demo_b_dataset is not None

    if manual_mode:
        demo_a_dataset = args.demo_a_dataset
        demo_a_episode = args.demo_a_episode or 0
        demo_b_dataset = args.demo_b_dataset
        demo_b_episode = args.demo_b_episode or 0
        task_a = Path(demo_a_dataset).parent.parent.name
        task_b = Path(demo_b_dataset).parent.parent.name
    else:
        print("Scanning available datasets for a matching demo pair...")
        layout_index = find_available_demos()
        if not layout_index:
            print("ERROR: No atomic task datasets found. Download datasets first.")
            return

        result = auto_select_demo_pair(layout_index)
        if result is None:
            print("ERROR: No two demos from different tasks share the same layout_id.")
            print("Available layouts and tasks:")
            for lid, entries in sorted(layout_index.items()):
                tasks = set(e[0] for e in entries)
                print(f"  Layout {lid}: {sorted(tasks)}")
            return

        info_a, info_b, auto_layout = result
        task_a, demo_a_episode, style_a, demo_a_dataset = info_a
        task_b, demo_b_episode, style_b, demo_b_dataset = info_b

        if args.layout is None:
            args.layout = auto_layout
        if args.style is None:
            args.style = style_a  # use demo A's style

        print(f"Auto-selected demos from layout {args.layout}:")
        print(f"  Robot 0: {task_a} ep={demo_a_episode} (style={style_a})")
        print(f"  Robot 1: {task_b} ep={demo_b_episode} (style={style_b})")

    if args.layout is None:
        print("ERROR: --layout is required in manual mode (or let auto-discovery set it)")
        return
    if args.style is None:
        args.style = 34

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load demos ---
    print(f"\nLoading demo A: {demo_a_dataset} ep={demo_a_episode}")
    demo_a = load_demo(demo_a_dataset, demo_a_episode)
    print(f"  Actions shape: {demo_a['actions'].shape}")
    print(f"  Layout ID: {demo_a['ep_meta'].get('layout_id', 'unknown')}")

    print(f"Loading demo B: {demo_b_dataset} ep={demo_b_episode}")
    demo_b = load_demo(demo_b_dataset, demo_b_episode)
    print(f"  Actions shape: {demo_b['actions'].shape}")
    print(f"  Layout ID: {demo_b['ep_meta'].get('layout_id', 'unknown')}")

    # Validate layout compatibility
    layout_a = demo_a["ep_meta"].get("layout_id")
    layout_b = demo_b["ep_meta"].get("layout_id")
    if layout_a is not None and layout_b is not None and layout_a != layout_b:
        print(f"WARNING: Layout mismatch! Demo A={layout_a}, Demo B={layout_b}")
    if layout_a is not None and layout_a != args.layout:
        print(f"WARNING: Demo A layout ({layout_a}) != requested layout ({args.layout})")
    if layout_b is not None and layout_b != args.layout:
        print(f"WARNING: Demo B layout ({layout_b}) != requested layout ({args.layout})")

    # --- Create 2-robot env ---
    # We patch init_robot_base_pose during reset() so each robot's mount body
    # orientation matches its demo exactly. This is critical: base actions
    # (forward/side velocities) are in the mount body's local frame, and
    # OSC arm actions depend on the base frame. If the mount body orientation
    # doesn't match the demo, actions replay incorrectly.
    demo_b_obj_names = [cfg["name"] for cfg in demo_b["ep_meta"].get("object_cfgs", [])]

    print(f"\nCreating 2-robot Kitchen env (layout={args.layout}, style={args.style})...")

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
    env = EnclosingWallRenderWrapper(env, alpha=0.1, enabled=True)

    # Patch init_robot_base_pose so _load_model() uses exact demo anchors.
    # Each demo was recorded as robot 0 with a specific anchor orientation
    # determined by the task's reference fixture. We force each composed-env
    # robot to use the same anchor so the compiled mount body frame matches.
    demo_anchors = [
        (np.array(demo_a["ep_meta"]["init_robot_base_pos"]),
         np.array(demo_a["ep_meta"]["init_robot_base_ori"])),
        (np.array(demo_b["ep_meta"]["init_robot_base_pos"]),
         np.array(demo_b["ep_meta"]["init_robot_base_ori"])),
    ]
    _original_init_robot_base_pose = EnvUtils.init_robot_base_pose

    def _patched_init_robot_base_pose(env, robot_idx=0, offset=None):
        if robot_idx < len(demo_anchors):
            return demo_anchors[robot_idx]
        return _original_init_robot_base_pose(env, robot_idx=robot_idx, offset=offset)

    EnvUtils.init_robot_base_pose = _patched_init_robot_base_pose

    # Set demo A's ep_meta so reset() spawns its objects and configures fixtures
    if hasattr(env, "set_ep_meta"):
        env.set_ep_meta(demo_a["ep_meta"])
    env.reset()

    # Restore original function
    EnvUtils.init_robot_base_pose = _original_init_robot_base_pose

    # Print action space info for debugging
    for i, robot in enumerate(env.robots):
        print(f"  Robot {i}: action_dim={robot.action_dim}, splits={dict(robot._action_split_indexes)}")
    print(f"  Total env action_dim: {env.action_dim}")

    # --- Transfer fixture and object states from demos ---
    # Build joint name -> qpos value maps from each demo's model XML + state.
    # Then write matching joints into the 2-robot env by name. This sets
    # cabinet door angles, object positions, drawer states, etc.
    #
    # Key problem: both demos share the same layout so they have ALL fixture
    # joints, but only their task-relevant fixtures are in the right state
    # (e.g. cabinet open). Blindly merging would let one demo's "closed"
    # overwrite the other's "open". Solution: each demo's fixture_refs
    # identify its task-relevant fixtures — those joints get priority from
    # that demo. For non-referenced fixtures with conflicts, we take the
    # value with the larger absolute value (i.e. "more open").
    print(f"\nTransferring scene state from demos...")
    demo_a_joints = build_demo_joint_map(demo_a["model_xml"], demo_a["states"][0])
    demo_b_joints = build_demo_joint_map(demo_b["model_xml"], demo_b["states"][0])

    # Demo B's object joints (obj_joint0, etc.) won't exist in the env since
    # we only spawn demo A's objects. Skip them during merge.
    demo_b_filtered = {
        name: val for name, val in demo_b_joints.items()
        if not any(name.startswith(obj_name + "_") for obj_name in demo_b_obj_names)
    }

    # Build sets of joint name prefixes that each demo "owns" via fixture_refs.
    # fixture_refs maps role names (e.g. "cab") to fixture IDs like
    # "hingecabinet_3_right_group_1". Any joint starting with that ID belongs
    # to that demo's task-relevant fixtures.
    demo_a_fixture_prefixes = set(
        fid for fid in demo_a["ep_meta"].get("fixture_refs", {}).values()
    )
    demo_b_fixture_prefixes = set(
        fid for fid in demo_b["ep_meta"].get("fixture_refs", {}).values()
    )
    print(f"  Demo A fixture_refs: {demo_a_fixture_prefixes}")
    print(f"  Demo B fixture_refs: {demo_b_fixture_prefixes}")

    def _joint_owned_by(joint_name, prefixes):
        return any(joint_name.startswith(p) for p in prefixes)

    # Merge with fixture-ref-aware priority:
    # 1. If joint belongs to demo A's fixtures → use demo A's value
    # 2. If joint belongs to demo B's fixtures → use demo B's value
    # 3. If both or neither claim it → use the larger absolute value ("more open")
    merged_joints = {}
    all_joint_names = set(demo_a_joints.keys()) | set(demo_b_filtered.keys())
    for joint_name in all_joint_names:
        in_a = joint_name in demo_a_joints
        in_b = joint_name in demo_b_filtered
        owned_a = _joint_owned_by(joint_name, demo_a_fixture_prefixes)
        owned_b = _joint_owned_by(joint_name, demo_b_fixture_prefixes)

        if in_a and not in_b:
            merged_joints[joint_name] = demo_a_joints[joint_name]
        elif in_b and not in_a:
            merged_joints[joint_name] = demo_b_filtered[joint_name]
        elif owned_a and not owned_b:
            merged_joints[joint_name] = demo_a_joints[joint_name]
        elif owned_b and not owned_a:
            merged_joints[joint_name] = demo_b_filtered[joint_name]
        else:
            # Both have it, neither or both own it — pick "more open" (larger abs)
            val_a = demo_a_joints[joint_name]
            val_b = demo_b_filtered[joint_name]
            if np.max(np.abs(val_b)) > np.max(np.abs(val_a)):
                merged_joints[joint_name] = val_b
            else:
                merged_joints[joint_name] = val_a

    # Skip robot joints — we set those separately below
    robot_joint_prefixes = ("robot0_", "robot1_", "gripper0_", "gripper1_",
                            "mobilebase0_", "mobilebase1_")
    set_count = 0
    skip_count = 0
    for joint_name, qpos_vals in merged_joints.items():
        if any(joint_name.startswith(p) for p in robot_joint_prefixes):
            continue
        try:
            addr = env.sim.model.get_joint_qpos_addr(joint_name)
            if isinstance(addr, tuple):
                # Free/ball joints return (start, end)
                start, end = addr
                env.sim.data.qpos[start:end] = qpos_vals
            else:
                # Hinge/slide joints return a single index
                env.sim.data.qpos[addr : addr + len(qpos_vals)] = qpos_vals
            set_count += 1
        except Exception:
            skip_count += 1
    print(f"  Set {set_count} fixture/object joints, skipped {skip_count} (not found in env)")

    # --- Initialize robot positions from demo states ---
    # Since we patched the anchors to match each demo, the mount body frame
    # is identical to the demo's. set_robot_to_position handles x/y correctly,
    # and the raw yaw qpos from the demo works without adjustment.
    robot0_init_qpos = extract_robot_qpos_from_state(demo_a["states"][0])
    robot1_init_qpos = extract_robot_qpos_from_state(demo_b["states"][0])

    print(f"\nSetting robot positions from demos...")
    for i, (robot, init_qpos, demo) in enumerate(
        [
            (env.robots[0], robot0_init_qpos, demo_a),
            (env.robots[1], robot1_init_qpos, demo_b),
        ]
    ):
        # Set arm (7) + gripper (2) + torso (1) from the demo state
        env.sim.data.qpos[robot._ref_torso_joint_pos_indexes] = init_qpos[3:4]
        env.sim.data.qpos[robot._ref_joint_pos_indexes] = init_qpos[4:11]
        env.sim.data.qpos[robot._ref_gripper_joint_pos_indexes["right"]] = init_qpos[11:13]

        # Set base position using env_utils which handles per-robot mount offsets
        demo_base_world = np.array(demo["ep_meta"]["init_robot_base_pos"])
        EnvUtils.set_robot_to_position(env, demo_base_world, robot_idx=i)

        # Set yaw directly from demo state (no adjustment needed — anchors match)
        base_joints = EnvUtils._get_mobile_base_joints(env, robot_idx=i)
        if base_joints is not None:
            _, _, yaw_jnt_name = base_joints
            env.sim.data.qpos[env.sim.model.get_joint_qpos_addr(yaw_jnt_name)] = init_qpos[2]

        print(f"  Robot {i} world pos: {demo_base_world}, yaw: {init_qpos[2]:.3f}")

    env.sim.forward()

    # Re-sync each robot's controller with the new joint positions.
    # Without this, the OSC controller would try to "correct" back to the old pose.
    for robot in env.robots:
        robot.base_pos = env.sim.data.get_body_xpos(robot.robot_model.root_body)
        robot.base_ori = env.sim.data.get_body_xmat(robot.robot_model.root_body).reshape((3, 3))
        robot.composite_controller.update_state()
        robot.composite_controller.reset()

    # --- Prepare actions ---
    actions_a = demo_a["actions"]
    actions_b = demo_b["actions"]
    T = max(len(actions_a), len(actions_b))

    print(f"\nReplaying {T} steps (demo A: {len(actions_a)}, demo B: {len(actions_b)})...")

    # --- Set up room-wide camera (free camera showing whole kitchen) ---
    room_cam_config = CamUtils.LAYOUT_CAMS.get(
        env.layout_id, CamUtils.DEFAULT_LAYOUT_CAM
    )
    render_ctx = env.sim._render_context_offscreen
    if render_ctx is None:
        raise RuntimeError("Offscreen render context is not initialized.")

    # --- Discover available fixed cameras ---
    available_cameras = {env.sim.model.camera_id2name(i) for i in range(env.sim.model.ncam)}
    fixed_cameras = [c for c in args.cameras if c in available_cameras]
    if not fixed_cameras:
        fixed_cameras = sorted(c for c in available_cameras if "robot" in c)[:2]
    print(f"  Fixed cameras: {fixed_cameras}")
    print(f"  Room camera: layout {env.layout_id}")

    # --- Set up video writers ---
    writers = {}
    for cam in fixed_cameras:
        video_path = os.path.join(args.output_dir, f"{cam}.mp4")
        writers[cam] = imageio.get_writer(video_path, fps=args.fps)

    room_path = os.path.join(args.output_dir, "room_view.mp4")
    room_writer = imageio.get_writer(room_path, fps=args.fps)

    # --- Step loop ---
    try:
        for t in range(T):
            if t < len(actions_a):
                r0_action = map_demo_action_to_robot(actions_a[t], env.robots[0])
            else:
                r0_action = idle_action(env.robots[0])

            if t < len(actions_b):
                r1_action = map_demo_action_to_robot(actions_b[t], env.robots[1])
            else:
                r1_action = idle_action(env.robots[1])

            full_action = np.concatenate([r0_action, r1_action])
            env.step(full_action)

            if t % args.video_skip == 0 or t == T - 1:
                # Render room-wide free camera
                render_ctx.cam.lookat[:] = room_cam_config["lookat"]
                render_ctx.cam.distance = room_cam_config["distance"]
                render_ctx.cam.azimuth = room_cam_config["azimuth"]
                render_ctx.cam.elevation = room_cam_config["elevation"]
                render_ctx.render(
                    width=args.width * 2,
                    height=args.height,
                    camera_id=-1,
                )
                room_frame = render_ctx.read_pixels(args.width * 2, args.height)
                room_writer.append_data(room_frame[::-1])

                # Render fixed cameras
                for cam in fixed_cameras:
                    frame = env.sim.render(
                        height=args.height,
                        width=args.width,
                        camera_name=cam,
                    )[::-1]
                    writers[cam].append_data(frame)

            if (t + 1) % 100 == 0 or t == T - 1:
                print(f"  Step {t + 1}/{T}")

    finally:
        for w in writers.values():
            w.close()
        room_writer.close()
        env.close()

    # --- Save metadata ---
    meta = {
        "demo_a": {
            "task": task_a if not manual_mode else Path(demo_a_dataset).parent.parent.name,
            "dataset": str(demo_a_dataset),
            "episode": demo_a_episode,
            "num_actions": len(actions_a),
            "layout_id": demo_a["ep_meta"].get("layout_id"),
        },
        "demo_b": {
            "task": task_b if not manual_mode else Path(demo_b_dataset).parent.parent.name,
            "dataset": str(demo_b_dataset),
            "episode": demo_b_episode,
            "num_actions": len(actions_b),
            "layout_id": demo_b["ep_meta"].get("layout_id"),
        },
        "env": {
            "layout": args.layout,
            "style": args.style,
            "total_steps": T,
        },
        "cameras": fixed_cameras + ["room_view"],
    }
    meta_path = os.path.join(args.output_dir, "compose_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"\nDone. Output saved to {args.output_dir}/")
    print(f"  Videos: room_view, {', '.join(fixed_cameras)}")
    print(f"  Metadata: compose_meta.json")


if __name__ == "__main__":
    main()
