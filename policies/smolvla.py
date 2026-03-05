#!/usr/bin/env python
"""
Run SmolVLA rollouts on a RoboCasa gym environment.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

# Avoid shadowing Python's stdlib `random` by experiments/random.py.
_THIS_DIR = Path(__file__).resolve().parent
if sys.path and Path(sys.path[0]).resolve() == _THIS_DIR:
    sys.path.pop(0)

import gymnasium as gym
import imageio
import numpy as np
import torch


def _bootstrap_robosuite_path() -> None:
    """Ensure the in-repo robosuite package is importable before robocasa import."""
    repo_root = Path(__file__).resolve().parents[1]
    robosuite_src = repo_root / "robosuite"
    if robosuite_src.is_dir():
        sys.path.insert(0, str(robosuite_src))


def _require_transformers() -> None:
    """Fail fast with actionable guidance if transformers is unavailable."""
    if importlib.util.find_spec("transformers") is None:
        raise SystemExit(
            "Missing dependency: 'transformers'. Install it with either "
            "`pip install \"lerobot[smolvla]\"` or `pip install transformers`."
        )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"Expected a positive integer, got {value}.")
    return parsed


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _load_policy(policy_path: str, local_files_only: bool, device: torch.device):
    _require_transformers()
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy.from_pretrained(
        policy_path,
        local_files_only=local_files_only,
    )
    policy.to(device)
    policy.eval()

    action_feature = policy.config.action_feature
    if action_feature is None:
        raise ValueError("Policy config has no action feature; expected a 12D action output.")
    if action_feature.shape[0] != 12:
        raise ValueError(
            f"Incompatible policy action feature shape {action_feature.shape}. "
            "Expected 12 for RoboCasa PandaOmron actions."
        )

    return policy


def _to_policy_image_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    if image.ndim != 3:
        raise ValueError(f"Expected HWC image with 3 dims, got shape {image.shape}.")
    if image.dtype != np.uint8:
        raise ValueError(
            f"Expected uint8 image input, got dtype {image.dtype} for shape {image.shape}."
        )

    tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous().to(torch.float32) / 255.0
    return tensor.unsqueeze(0).to(device=device)


def _build_state(obs: dict, device: torch.device) -> torch.Tensor:
    required_state_keys = (
        "state.base_position",
        "state.base_rotation",
        "state.end_effector_position_relative",
        "state.end_effector_rotation_relative",
        "state.gripper_qpos",
    )
    missing = [key for key in required_state_keys if key not in obs]
    if missing:
        raise ValueError(
            "Missing required state keys in observation: "
            f"{missing}. Available keys: {sorted(obs.keys())}"
        )

    state_np = np.concatenate([np.asarray(obs[key], dtype=np.float32) for key in required_state_keys], axis=0)
    if state_np.shape != (16,):
        raise ValueError(
            f"Constructed observation.state has shape {state_np.shape}; expected (16,). "
            "Check observation key mapping."
        )
    return torch.from_numpy(state_np).unsqueeze(0).to(device=device)


def _map_policy_image_key_to_obs_key(policy_key: str, obs: dict) -> str | None:
    if policy_key in obs:
        return policy_key

    if policy_key.startswith("observation.images."):
        camera_name = policy_key.removeprefix("observation.images.")
        fallback_key = f"video.{camera_name}"
        if fallback_key in obs:
            return fallback_key

    return None


def _build_policy_batch(
    obs: dict,
    policy,
    task_override: str | None,
    device: torch.device,
) -> dict[str, torch.Tensor | list[str]]:
    batch: dict[str, torch.Tensor | list[str]] = {
        "observation.state": _build_state(obs, device),
    }

    expected_image_keys = list(policy.config.image_features.keys())
    available_image_keys = sorted([k for k in obs if k.startswith("video.") or k.startswith("observation.images.")])
    missing_image_keys = []
    for policy_image_key in expected_image_keys:
        obs_image_key = _map_policy_image_key_to_obs_key(policy_image_key, obs)
        if obs_image_key is None:
            missing_image_keys.append(policy_image_key)
            continue

        batch[policy_image_key] = _to_policy_image_tensor(obs[obs_image_key], device)

    if missing_image_keys:
        raise ValueError(
            "Observation is missing image keys required by policy.\n"
            f"Required policy keys: {expected_image_keys}\n"
            f"Missing policy keys: {missing_image_keys}\n"
            f"Available observation image keys: {available_image_keys}"
        )

    if task_override is not None:
        task_text = task_override
    else:
        task_text = obs.get("annotation.human.task_description", "")
    if not isinstance(task_text, str):
        task_text = str(task_text)
    batch["task"] = [task_text]

    return batch


def _policy_action_to_env_action(action: torch.Tensor) -> dict[str, np.ndarray]:
    if not isinstance(action, torch.Tensor):
        raise TypeError(f"Expected action to be torch.Tensor, got {type(action)}.")

    action = action.detach().to("cpu")
    if action.shape != (1, 12):
        raise ValueError(
            f"Expected SmolVLA select_action output shape (1, 12), got {tuple(action.shape)}."
        )

    action_np = action.numpy()[0].astype(np.float32, copy=False)
    return {
        "action.base_motion": action_np[0:4].copy(),
        "action.control_mode": action_np[4:5].copy(),
        "action.end_effector_position": action_np[5:8].copy(),
        "action.end_effector_rotation": action_np[8:11].copy(),
        "action.gripper_close": action_np[11:12].copy(),
    }


def run_smolvla_rollouts(args: argparse.Namespace) -> dict:
    _bootstrap_robosuite_path()
    import robocasa  # noqa: F401  # needed to register gym envs

    device = _resolve_device(args.device)
    policy = _load_policy(args.policy_path, args.local_files_only, device)

    env = gym.make(
        args.env_id,
        split=args.split,
        seed=args.seed,
    )

    video_writer = None
    video_path = None
    if args.video_path:
        video_path = Path(args.video_path).expanduser()
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_writer = imageio.get_writer(str(video_path), fps=20)

    info: dict = {}
    num_success_rollouts = 0
    try:
        for _ in range(args.num_rollouts):
            obs, info = env.reset()
            policy.reset()

            for _ in range(args.num_steps):
                policy_batch = _build_policy_batch(obs, policy, args.task_text, device)
                with torch.inference_mode():
                    action = policy.select_action(policy_batch)
                env_action = _policy_action_to_env_action(action)
                obs, reward, terminated, truncated, info = env.step(env_action)

                if video_writer is not None:
                    frame = env.unwrapped.sim.render(
                        height=512,
                        width=768,
                        camera_name=args.render_camera,
                    )[::-1]
                    video_writer.append_data(frame)

                if info.get("success", False):
                    num_success_rollouts += 1
                    break
    finally:
        if video_writer is not None:
            video_writer.close()
        env.close()

    info = dict(info)
    info["num_success_rollouts"] = num_success_rollouts

    if video_path is not None:
        print(f"Saved video of rollouts to {video_path}")

    return info


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SmolVLA policy rollouts in RoboCasa.")
    parser.add_argument(
        "--policy-path",
        type=str,
        required=True,
        help="SmolVLA checkpoint path (local directory) or Hugging Face repo id.",
    )
    parser.add_argument(
        "--env-id",
        type=str,
        default="robocasa/PickPlaceCounterToCabinet",
        help="Gym environment id.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="pretrain",
        help="RoboCasa split (e.g., pretrain, target, all).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Environment seed.",
    )
    parser.add_argument(
        "--num-rollouts",
        type=_positive_int,
        default=3,
        help="Number of rollouts.",
    )
    parser.add_argument(
        "--num-steps",
        type=_positive_int,
        default=100,
        help="Max number of steps per rollout.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device to run policy on: auto, cpu, cuda, cuda:0, etc.",
    )
    parser.add_argument(
        "--video-path",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "smolvla_test.mp4"),
        help="Output rollout video path. Set to empty string to disable.",
    )
    parser.add_argument(
        "--render-camera",
        type=str,
        default="robot0_agentview_center",
        help="Camera name used when rendering rollout video.",
    )
    parser.add_argument(
        "--task-text",
        type=str,
        default=None,
        help="Optional task text override. Defaults to environment-provided task description.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load policy/config using local cache/files only (no network download).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.video_path == "":
        args.video_path = None

    info = run_smolvla_rollouts(args)
    print(info)


if __name__ == "__main__":
    main()
