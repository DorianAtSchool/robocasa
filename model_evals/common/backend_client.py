"""Shared command-line runner for RoboCasa VLA backends."""

from __future__ import annotations

import argparse
import logging
import traceback

import numpy as np

from model_evals.common.robocasa_eval_client import PolicyAdapter, eval_env
from model_evals.common.trajectory_spawn import (
    DEFAULT_TRAJECTORY_INIT_ROOT,
    list_trajectory_task_jobs,
    load_trajectory_for_env,
)
from model_evals.common.websocket import WebsocketClient

logger = logging.getLogger(__name__)


def parse_common_args(description: str):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18055)
    parser.add_argument(
        "--task_set",
        type=str,
        nargs="+",
        default=["atomic_seen", "composite_seen", "composite_unseen"],
    )
    parser.add_argument("--eval_trajectory_tasks", action="store_true")
    parser.add_argument("--single_task", type=str, default=None)
    parser.add_argument("--task_text_override", type=str, default=None)
    parser.add_argument(
        "--cameras",
        type=str,
        default="default",
        help=(
            "Video cameras to record: default records robot0 main view, robot0 wrist, "
            "and one robot1 view; use all, render, or comma-separated obs/sim camera names."
        ),
    )
    parser.add_argument("--split", type=str, default="pretrain", choices=["pretrain", "target"])
    parser.add_argument("--num_trials", type=int, default=50)
    parser.add_argument("--replan_steps", type=int, default=5)
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--worker_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--action_chunk", type=int, default=None)
    parser.add_argument("--extra_robot", action="store_true")
    parser.add_argument("--trajectory_init_json", type=str, default=None)
    parser.add_argument("--trajectory_init_root", type=str, default=None)
    parser.add_argument("--trajectory_index", type=int, default=0)
    parser.add_argument("--trajectory_indices", type=int, nargs="+", default=None)
    parser.add_argument(
        "--trajectory_init_mode",
        type=str,
        required=True,
        choices=["none", "passive_other", "all_agents"],
    )
    parser.add_argument("--trajectory_other_agent", type=str, default="agent_1")
    parser.add_argument(
        "--missing_trajectory_policy",
        type=str,
        default="error",
        choices=["error", "fallback", "skip"],
    )
    return parser.parse_args()


def _resolve_trajectory_config(args):
    trajectory_indices = args.trajectory_indices or [args.trajectory_index]
    trajectory_init_root = args.trajectory_init_root

    if args.eval_trajectory_tasks:
        if args.single_task:
            raise ValueError("--eval_trajectory_tasks cannot be combined with --single_task.")
        if args.trajectory_init_json is not None:
            raise ValueError("--eval_trajectory_tasks uses --trajectory_init_root, not --trajectory_init_json.")
        if trajectory_init_root is None:
            trajectory_init_root = DEFAULT_TRAJECTORY_INIT_ROOT
    elif (
        args.trajectory_init_mode != "none"
        and args.trajectory_init_json is None
        and trajectory_init_root is None
    ):
        trajectory_init_root = DEFAULT_TRAJECTORY_INIT_ROOT

    if args.trajectory_init_mode == "none" and (
        args.trajectory_init_json is not None or trajectory_init_root is not None
    ):
        logger.warning(
            "Trajectory init source was provided but --trajectory_init_mode=none, so it will be ignored."
        )

    return trajectory_init_root, trajectory_indices


def _build_jobs(args, trajectory_init_root: str | None, trajectory_indices: list[int]):
    from robocasa.utils.dataset_registry import TASK_SET_REGISTRY

    if args.eval_trajectory_tasks:
        return list_trajectory_task_jobs(trajectory_init_root, trajectory_indices)

    if args.single_task:
        all_env_names = [args.single_task]
    else:
        all_env_names = []
        for task in args.task_set:
            all_env_names.extend(TASK_SET_REGISTRY[task])

    return [
        {
            "env_name": env_name,
            "trajectory_index": args.trajectory_index,
            "trajectory_path": None,
            "trajectory_init": None,
            "log_label": None,
        }
        for env_name in all_env_names
    ]


def _print_run_header(backend_name: str, args, trajectory_init_root, trajectory_indices, all_jobs, my_jobs):
    print("=" * 60)
    print(f"{backend_name} RoboCasa Client [worker {args.worker_id}/{args.num_workers}]")
    print("=" * 60)
    print(f"  Server:      {args.host}:{args.port}")
    print(f"  Task sets:   {args.task_set}")
    print(f"  Single task: {args.single_task}")
    print(f"  Text override: {args.task_text_override}")
    print(f"  Cameras:     {args.cameras}")
    print(f"  Split:       {args.split}")
    print(f"  Num trials:  {args.num_trials}")
    print(f"  Replan:      every {args.replan_steps} steps")
    if args.action_chunk is not None:
        print(f"  Action chunk (expected): {args.action_chunk}")
    print(f"  Log dir:     {args.log_dir}")
    print(f"  Extra robot: {args.extra_robot}")
    print(f"  Eval traj tasks: {args.eval_trajectory_tasks}")
    print(f"  Traj init:   {args.trajectory_init_mode}")
    print(f"  Traj json:   {args.trajectory_init_json}")
    print(f"  Traj root:   {trajectory_init_root}")
    print(f"  Traj indices:{trajectory_indices}")
    print(f"  Traj agent:  {args.trajectory_other_agent}")
    print(f"  Missing traj:{args.missing_trajectory_policy}")
    print(f"  Jobs total:  {len(all_jobs)}, this worker: {len(my_jobs)}")
    print(f"  My tasks:    {[job['env_name'] for job in my_jobs]}")
    print("=" * 60)


def run_backend_client(
    backend_name: str,
    description: str,
    adapter: PolicyAdapter,
    client_factory=None,
):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_common_args(description)
    np.random.seed(args.seed)

    trajectory_init_root, trajectory_indices = _resolve_trajectory_config(args)
    all_jobs = _build_jobs(args, trajectory_init_root, trajectory_indices)
    my_jobs = [job for i, job in enumerate(all_jobs) if i % args.num_workers == args.worker_id]
    _print_run_header(backend_name, args, trajectory_init_root, trajectory_indices, all_jobs, my_jobs)

    if client_factory is None:
        client_factory = WebsocketClient
    client = client_factory(args.host, args.port)
    for job in my_jobs:
        env_name = job["env_name"]
        try:
            trajectory_init = job.get("trajectory_init")
            trajectory_path = job.get("trajectory_path")
            trajectory_index = job.get("trajectory_index", args.trajectory_index)
            if args.trajectory_init_mode != "none" and trajectory_init is None:
                trajectory_init, trajectory_path = load_trajectory_for_env(
                    env_name=env_name,
                    trajectory_json=args.trajectory_init_json,
                    trajectory_root=trajectory_init_root,
                    trajectory_index=trajectory_index,
                )
                if trajectory_init is None:
                    if args.missing_trajectory_policy == "skip":
                        logger.warning("Skipping %s because no matching trajectory was found.", env_name)
                        continue
                    if args.missing_trajectory_policy == "error":
                        raise FileNotFoundError(
                            f"No trajectory init file found for {env_name} under "
                            f"{trajectory_init_root!r} at index {trajectory_index}."
                        )
                    logger.info("No trajectory init for %s; using fallback spawning.", env_name)
            if trajectory_init is not None:
                logger.info("Using trajectory init for %s: %s", env_name, trajectory_path)

            eval_env(
                env_name=env_name,
                split=args.split,
                log_dir=args.log_dir,
                num_trials=args.num_trials,
                replan_steps=args.replan_steps,
                client=client,
                seed=args.seed,
                adapter=adapter,
                expected_action_chunk=args.action_chunk,
                extra_robot=args.extra_robot,
                task_text_override=args.task_text_override,
                cameras=args.cameras,
                trajectory_init=trajectory_init,
                trajectory_init_mode=args.trajectory_init_mode,
                trajectory_other_agent=args.trajectory_other_agent,
                trajectory_path=trajectory_path,
                log_label=job.get("log_label"),
                missing_trajectory_policy=args.missing_trajectory_policy,
            )
        except Exception as exc:
            logger.error("Error evaluating %s: %s", env_name, exc)
            traceback.print_exc()

    client.close()
    print(f"Worker {args.worker_id} complete.")

