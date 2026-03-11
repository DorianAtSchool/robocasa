"""
Two-robot demo that replays recorded dataset trajectories in a single simulator.

Both robots share one MuJoCo scene. Robot joint states (qpos/qvel[0:13]) are
extracted from two single-robot dataset trajectories and injected into the
corresponding joint slots of each robot in the two-robot environment.
"""

import argparse
import json
import os
import time
from collections import OrderedDict
from pathlib import Path

import imageio
import numpy as np
import robosuite
from termcolor import colored

import robocasa
import robocasa.utils.lerobot_utils as LU
from robocasa.scripts.dataset_scripts.playback_utils import (
    resolve_instruction_from_ep_meta,
)
from robocasa.scripts.download_datasets import download_datasets
from robocasa.utils.dataset_registry_utils import get_ds_path
from robocasa.wrappers.gym_wrapper import RoboCasaMultiAgentGymEnv

# PandaOmron joint layout in qpos:
#   [0:4]  mobile base (forward, side, yaw, torso)
#   [4:11] arm joints
#   [11:13] gripper fingers
ROBOT_NQPOS = 13
BASE_NQPOS = 4          # base joints to SKIP (keep env spawn position)
ARM_GRIPPER_START = 4    # first arm joint index within the robot's qpos block
ARM_GRIPPER_NQPOS = 9    # 7 arm + 2 gripper

ALL_TASKS = OrderedDict(
    [
        # Atomic: pick and place
        ("PickPlaceCounterToCabinet", "pick and place from counter to cabinet"),
        ("PickPlaceCounterToSink", "pick and place from counter to sink"),
        ("PickPlaceMicrowaveToCounter", "pick and place from microwave to counter"),
        ("PickPlaceStoveToCounter", "pick and place from stove to counter"),
        # Atomic: doors / drawers
        ("OpenSingleDoor", "open cabinet or microwave door"),
        ("CloseDrawer", "close drawer"),
        # Atomic: appliances
        ("TurnOnMicrowave", "turn on microwave"),
        ("TurnOnSinkFaucet", "turn on sink faucet"),
        ("TurnOnStove", "turn on stove"),
        # Composite: cooking / prep
        ("ArrangeVegetables", "arrange vegetables on a cutting board"),
        ("MicrowaveThawing", "place frozen food in microwave for thawing"),
        ("RestockPantry", "restock cans in pantry"),
        ("PreSoakPan", "prepare pan for washing"),
        ("PrepareCoffee", "make coffee"),
        # Composite: navigation
        (
            "HotDogSetup",
            "gather hot dog ingredients and place on dining table [Navigation]",
        ),
        (
            "DeliverStraw",
            "place the straw in the glass cup on the dining counter [Navigation]",
        ),
        (
            "GatherTableware",
            "gather tableware from around the kitchen [Navigation]",
        ),
        (
            "NavigateKitchen",
            "navigate to a target fixture in the kitchen [Navigation]",
        ),
        # Composite: serving / multi-step
        ("CoffeeServeMug", "serve a mug of coffee"),
        ("ServeTea", "serve tea [Navigation]"),
        ("GatherVegetables", "gather vegetables from around the kitchen"),
        ("GatherCuttingTools", "gather cutting tools"),
    ]
)


def get_ds_path_any_split(task, source="human"):
    path = get_ds_path(task, source=source, split="pretrain")
    if path is not None:
        return path
    return get_ds_path(task, source=source, split="target")


def choose_option(options, option_name, default=None):
    if default is None:
        default = list(options.keys())[0]
    print("{}s:".format(option_name.capitalize()))
    for i, (k, v) in enumerate(options.items()):
        print("[{}] {}: {}".format(i, k, v))
    print()
    try:
        s = input(
            "Choose an option 0 to {}, or any other key for default ({}): ".format(
                len(options) - 1, default
            )
        )
        k = min(max(int(s), 0), len(options) - 1)
        choice = list(options.keys())[k]
    except Exception:
        choice = default
        print("Using {} by default.\n".format(choice))
    return choice


def ensure_dataset(task):
    dataset = get_ds_path_any_split(task, source="human")
    if dataset is None:
        raise ValueError(f"No registered dataset path for task={task} source=human")
    if not os.path.exists(dataset):
        print(colored("Dataset not found locally. Downloading...", "yellow"))
        download_datasets(tasks=[task], split=["pretrain", "target"], source=["human"])
    return dataset


def load_trajectory(dataset, ep_idx=0):
    """Load trajectory and extract per-timestep robot joint states."""
    dataset = Path(dataset)
    states = LU.get_episode_states(dataset, ep_idx)
    ep_meta = LU.get_episode_meta(dataset, ep_idx)

    # Flat state = [time(1), qpos(...), qvel(...)]
    # Robot joints are qpos[0:ROBOT_NQPOS] and qvel[0:ROBOT_NQVEL]
    # In the flat array: qpos starts at index 1
    robot_qpos_list = []
    robot_qvel_list = []
    for state in states:
        # Infer sizes: flat = 1 + nqpos + nqvel
        # For PandaOmron single-robot: nqvel = nqpos - 3 (free joints have 7 qpos but 6 qvel)
        # But robot joints are all hinge (1 qpos, 1 qvel each), so indices match
        qpos_start = 1
        robot_qpos_list.append(state[qpos_start : qpos_start + ROBOT_NQPOS])
        # qvel starts after all qpos
        # We need the total qpos size to find qvel offset
        # Use first state to compute: total = 1 + nqpos + nqvel, and nqpos is unknown
        # But we know robot qvel is at the same relative offset in the qvel block
    # Compute qvel offset from first state
    # nqpos can vary per scene, so we compute it from the flat state size
    # For hinge joints nqvel = nqpos - (num_free_joints * 1)
    # Simpler: just load one env to get the sizes... OR just extract qpos and skip qvel
    # Setting qvel to zero is fine for state playback (we're setting qpos each frame anyway)

    return np.array(robot_qpos_list), ep_meta


def build_env(args, task_name):
    """Build a two-robot env for the given task."""
    camera_names = [
        "robot0_agentview_center",
        "robot1_agentview_center",
    ]

    env_kwargs = dict(
        env_name=task_name,
        robots=["PandaOmron", "PandaOmron"],
        env_configuration=args.env_configuration,
        split=args.split,
        spread_two_robots=True,
        multi_robot_min_separation=args.min_robot_separation,
    )

    if args.onscreen:
        onscreen_camera = None if args.camera == "free" else args.camera
        env_kwargs.update(
            enable_render=False,
            has_renderer=True,
            has_offscreen_renderer=True,
            use_camera_obs=False,
            render_camera=onscreen_camera,
            renderer="mjviewer",
        )
    else:
        env_kwargs.update(
            enable_render=True,
            camera_names=camera_names,
            camera_widths=args.camera_width,
            camera_heights=args.camera_height,
        )

    return RoboCasaMultiAgentGymEnv(**env_kwargs)


def inject_robot_joints(env, robot_idx, qpos_vals):
    """Inject arm+gripper joints only, keeping the env's base position."""
    offset = robot_idx * ROBOT_NQPOS + ARM_GRIPPER_START
    env.env.sim.data.qpos[offset : offset + ARM_GRIPPER_NQPOS] = qpos_vals[ARM_GRIPPER_START:]
    env.env.sim.forward()


def run_playback(env, traj_r0, traj_r1, args, video_writer=None):
    """Replay two trajectories in the shared two-robot sim."""
    obs, info = env.reset()

    # Extend trajectories with repeated final frame
    extend = 50
    traj_r0 = np.concatenate([traj_r0, np.tile(traj_r0[-1:], (extend, 1))])
    traj_r1 = np.concatenate([traj_r1, np.tile(traj_r1[-1:], (extend, 1))])
    traj_len = max(len(traj_r0), len(traj_r1))

    print(colored(f"Playing back {traj_len} steps...", "yellow"))

    for t in range(traj_len):
        start = time.time()

        # Inject robot joint states from each trajectory
        idx_r0 = min(t, len(traj_r0) - 1)
        idx_r1 = min(t, len(traj_r1) - 1)
        inject_robot_joints(env, 0, traj_r0[idx_r0])
        inject_robot_joints(env, 1, traj_r1[idx_r1])

        if args.onscreen:
            env.env.render()
            elapsed = time.time() - start
            diff = 1 / 60 - elapsed
            if diff > 0:
                time.sleep(diff)

        if video_writer is not None and (t % args.video_skip == 0 or t == traj_len - 1):
            im_r0 = env.env.sim.render(
                height=args.camera_height, width=args.camera_width,
                camera_name="robot0_agentview_center",
            )[::-1]
            im_r1 = env.env.sim.render(
                height=args.camera_height, width=args.camera_width,
                camera_name="robot1_agentview_center",
            )[::-1]
            video_writer.append_data(np.concatenate([im_r0, im_r1], axis=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Two-robot demo: replays recorded dataset trajectories in a single "
            "simulator. Pick a task for each robot."
        ),
    )
    parser.add_argument("--task_r0", type=str, default=None, help="Task for robot 0")
    parser.add_argument("--task_r1", type=str, default=None, help="Task for robot 1")
    parser.add_argument("--split", type=str, default="pretrain")
    parser.add_argument(
        "--env_configuration", type=str, default="opposed",
        choices=["opposed", "parallel", "default"],
    )
    parser.add_argument("--min_robot_separation", type=float, default=1.25)
    parser.add_argument(
        "--onscreen", action="store_true",
        help="Launch an interactive MuJoCo viewer window.",
    )
    parser.add_argument(
        "--camera", type=str, default="kitchen_overview",
        help="On-screen camera when using --onscreen.",
    )
    parser.add_argument(
        "--video_path", type=str, default="/tmp/robocasa_demo_tasks_multi_agent",
        help="Path to video folder.",
    )
    parser.add_argument("--video_skip", type=int, default=5)
    parser.add_argument("--camera_height", type=int, default=512)
    parser.add_argument("--camera_width", type=int, default=512)
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()

    # Filter to tasks that have datasets
    tasks = OrderedDict(
        (k, v) for k, v in ALL_TASKS.items()
        if get_ds_path_any_split(k, source="human") is not None
    )
    if not tasks:
        raise RuntimeError("No tasks with registered human demo paths.")

    video_num = -1
    while True:
        print(colored("--- Select task for Robot 0 ---", "green"))
        task_r0 = args.task_r0 if args.task_r0 else choose_option(tasks, "task")

        print(colored("--- Select task for Robot 1 ---", "green"))
        task_r1 = args.task_r1 if args.task_r1 else choose_option(tasks, "task")

        video_num += 1
        print()
        print(colored(f"Robot 0: {task_r0}", "cyan"))
        print(colored(f"Robot 1: {task_r1}", "cyan"))
        print()

        # Load trajectories (robot joint states only)
        dataset_r0 = ensure_dataset(task_r0)
        dataset_r1 = ensure_dataset(task_r1)

        traj_r0, meta_r0 = load_trajectory(dataset_r0, ep_idx=0)
        traj_r1, meta_r1 = load_trajectory(dataset_r1, ep_idx=0)

        for label, meta in [("Robot 0", meta_r0), ("Robot 1", meta_r1)]:
            lang = resolve_instruction_from_ep_meta(meta)
            if lang:
                print(colored(f"  {label}: {lang}", "cyan"))

        # Create two-robot env using robot 0's task for the scene
        print(colored("Creating two-robot environment...", "yellow"))
        env = build_env(args, task_name=task_r0)

        video_writer = None
        video_file = None
        if not args.onscreen:
            video_dir = Path(args.video_path).expanduser().resolve()
            video_dir.mkdir(parents=True, exist_ok=True)
            video_file = video_dir / f"video_{video_num}.mp4"
            video_writer = imageio.get_writer(str(video_file), fps=args.fps)

        try:
            run_playback(env, traj_r0, traj_r1, args, video_writer=video_writer)
        finally:
            if video_writer is not None:
                video_writer.close()
                print(colored(f"Video saved: {video_file}", "green"))
            env.close()

        if args.task_r0 is not None and args.task_r1 is not None:
            break
        print()
