"""Shared RoboCasa rollout evaluator for VLA backends."""

from __future__ import annotations

import collections
from dataclasses import dataclass
from datetime import datetime
import json
import logging
import os
import pathlib
from typing import Callable

import imageio
import numpy as np
import tqdm

from model_evals.common.cameras import capture_frame, resolve_record_cameras
from model_evals.common.trajectory_spawn import (
    apply_fallback_initialization,
    apply_trajectory_initialization,
    refresh_observation,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PolicyAdapter:
    """Backend hooks used by the shared RoboCasa evaluator."""

    build_request: Callable[[dict, str], dict]
    extract_actions: Callable[[dict], np.ndarray]
    postprocess_action: Callable[[np.ndarray], np.ndarray]
    on_episode_start: Callable[[str, int], None] | None = None


def eval_env(
    env_name: str,
    split: str,
    log_dir: str,
    num_trials: int,
    replan_steps: int,
    client,
    seed: int,
    adapter: PolicyAdapter,
    expected_action_chunk: int | None = None,
    extra_robot: bool = False,
    task_text_override: str | None = None,
    cameras: str = "default",
    trajectory_init: dict | None = None,
    trajectory_init_mode: str = "none",
    trajectory_other_agent: str = "agent_1",
    trajectory_path: str | None = None,
    log_label: str | None = None,
    missing_trajectory_policy: str = "error",
):
    import gymnasium as gym
    from robocasa.utils.dataset_registry_utils import get_task_horizon
    from robocasa.utils.env_utils import convert_action

    task_horizon = get_task_horizon(env_name)
    horizon = int(task_horizon * 1.5)

    now = datetime.now().strftime("%Y-%m-%d-%H-%M")
    if log_label:
        log_path = f"{log_dir}/{env_name}/{log_label}/{now}"
    else:
        log_path = f"{log_dir}/{env_name}/{now}"

    for _root, _dirs, files in os.walk(os.path.dirname(log_path)):
        if "stats.json" in files:
            print(f"{env_name}/{split}, stats path exists, skipping.")
            return

    pathlib.Path(log_path).mkdir(parents=True, exist_ok=True)

    use_second_robot = (
        extra_robot
        or trajectory_init_mode in {"passive_other", "all_agents"}
        or missing_trajectory_policy == "fallback"
    )
    env_kwargs = {"split": split, "seed": seed}
    if use_second_robot:
        env_kwargs["robots"] = ["PandaOmron", "PandaOmron"]
    env = gym.make(f"robocasa/{env_name}", **env_kwargs)

    total_episodes, total_successes = 0, 0
    task_episodes, task_successes = 0, 0
    chunk_len_checked = False
    trajectory_placements_by_episode = []

    for episode_idx in tqdm.tqdm(range(num_trials), desc=env_name):
        obs, _info = env.reset()
        trajectory_placements = apply_trajectory_initialization(
            env, trajectory_init, trajectory_init_mode, trajectory_other_agent
        )
        if not trajectory_placements and missing_trajectory_policy == "fallback":
            trajectory_placements = apply_fallback_initialization(env_name, env, robot_idx=1)
        if trajectory_placements:
            obs = refresh_observation(env)
            trajectory_placements_by_episode.append(
                {"episode_idx": episode_idx, "placements": trajectory_placements}
            )
            logger.info("Robot initialization: %s", trajectory_placements)

        if adapter.on_episode_start is not None:
            adapter.on_episode_start(env_name, task_episodes)

        task_lang = task_text_override or obs["annotation.human.task_description"]
        action_plan = collections.deque()

        t = 0
        camera_entries = resolve_record_cameras(env, obs, cameras)
        replay_images_by_camera = {label: [] for _, _, label in camera_entries}

        logger.info("Starting episode %s, task: %s", task_episodes + 1, task_lang)
        logger.info("Recording cameras: %s", [label for _, _, label in camera_entries])

        done = False
        while t < horizon:
            if not action_plan:
                request = adapter.build_request(obs, task_lang)
                result = client.infer(request)
                action_chunk = adapter.extract_actions(result)
                if expected_action_chunk is not None and not chunk_len_checked:
                    got = len(action_chunk)
                    if got != expected_action_chunk:
                        raise RuntimeError(
                            f"Server returned {got} actions but --action_chunk expects "
                            f"{expected_action_chunk}. Restart the backend server with "
                            "matching --action_chunk or fix .server_info."
                        )
                    chunk_len_checked = True
                assert len(action_chunk) >= replan_steps, (
                    f"Want to replan every {replan_steps} steps, "
                    f"but got {len(action_chunk)} actions"
                )
                action_plan.extend(action_chunk[:replan_steps])

            action = action_plan.popleft()
            action = adapter.postprocess_action(action)
            action = convert_action(action)

            obs, _reward, done, _truncated, info = env.step(action)
            done = info["success"]

            if t % 2 == 0 or t == horizon - 1 or done:
                for camera_entry in camera_entries:
                    frame = capture_frame(env, obs, camera_entry)
                    if frame is not None:
                        replay_images_by_camera[camera_entry[2]].append(frame)

            if done:
                task_successes += 1
                total_successes += 1
                break
            t += 1

        task_episodes += 1
        total_episodes += 1

        suffix = "success" if done else "failure"
        for camera_label, replay_images in replay_images_by_camera.items():
            if not replay_images:
                continue
            if len(replay_images_by_camera) == 1:
                video_name = f"rollout_{episode_idx}_{suffix}.mp4"
            else:
                video_name = f"rollout_{episode_idx}_{camera_label}_{suffix}.mp4"
            imageio.mimwrite(
                pathlib.Path(log_path) / video_name,
                [np.asarray(x) for x in replay_images],
                fps=20,
            )

        logger.info("Success: %s", done)
        logger.info(
            "Episodes: %s, Successes: %s (%.1f%%)",
            total_episodes,
            total_successes,
            total_successes / total_episodes * 100,
        )

    logger.info("[%s] Success rate: %.3f", env_name, float(total_successes) / float(total_episodes))

    with open(os.path.join(log_path, "stats.json"), "w") as f:
        json.dump(
            {
                "num_episodes": total_episodes,
                "success_rate": float(total_successes) / float(total_episodes),
                "trajectory_init_mode": trajectory_init_mode,
                "trajectory_other_agent": trajectory_other_agent,
                "trajectory_path": trajectory_path,
                "log_label": log_label,
                "missing_trajectory_policy": missing_trajectory_policy,
                "trajectory_initializations": trajectory_placements_by_episode,
            },
            f,
            indent=4,
        )

    env.env.close()
    del env.env
    del env

