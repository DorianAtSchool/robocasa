# OLD


import argparse
from pathlib import Path

import imageio

from robocasa.wrappers.gym_wrapper import RoboCasaMultiAgentGymEnv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="AddIceCubes")
    parser.add_argument("--split", type=str, default="pretrain")
    parser.add_argument(
        "--env_configuration",
        type=str,
        default="opposed",
        choices=["opposed", "parallel", "default"],
    )
    parser.add_argument(
        "--min_robot_separation",
        type=float,
        default=1.25,
        help="Minimum XY separation (meters) between robot spawn anchors.",
    )
    parser.add_argument(
        "--clustered_spawn",
        action="store_true",
        help="Use legacy nearby spawn behavior for the second robot.",
    )
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument(
        "--onscreen",
        action="store_true",
        help="Launch an interactive Mujoco viewer window.",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="robot0_frontview",
        help="On-screen camera when using --onscreen.",
    )
    parser.add_argument(
        "--video_camera",
        type=str,
        default="robot0_frontview",
        help=(
            "Camera used for saved video "
            "(e.g. robot0_frontview, robot1_frontview, kitchen_overview)."
        ),
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="Optional mp4 path to save rollout video.",
    )
    parser.add_argument("--fps", type=int, default=20)
    return parser.parse_args()


def build_env(args):
    onscreen_camera = None if args.camera == "free" else args.camera
    env_kwargs = dict(
        env_name=args.env_name,
        robots=["PandaOmron", "PandaOmron"],
        env_configuration=args.env_configuration,
        split=args.split,
        spread_two_robots=not args.clustered_spawn,
        multi_robot_min_separation=args.min_robot_separation,
    )
    if args.onscreen:
        env_kwargs.update(
            enable_render=False,
            has_renderer=True,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            render_camera=onscreen_camera,
            renderer="mjviewer",
        )
    elif args.video_path is not None:
        # Requires offscreen rendering support (EGL / OSMesa depending on setup).
        env_kwargs.update(
            enable_render=True,
            camera_names=[args.video_camera],
        )
    else:
        # Fully headless fallback: no viewer and no offscreen renderer required.
        env_kwargs.update(
            enable_render=False,
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False,
        )

    return RoboCasaMultiAgentGymEnv(**env_kwargs)


def main():
    args = parse_args()
    env = build_env(args)
    video_writer = None
    if args.video_path is not None:
        video_path = Path(args.video_path).expanduser().resolve()
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_writer = imageio.get_writer(str(video_path), fps=args.fps)
        print(f"Recording video to: {video_path}")

    try:
        obs, info = env.reset()
        print("reset agents:", list(obs.keys()))
        total_rewards = {agent: 0.0 for agent in env.agents}

        for step in range(args.steps):
            actions = env.action_space.sample()
            obs, rewards, terminated, truncated, infos = env.step(actions)
            for agent, rew in rewards.items():
                total_rewards[agent] += float(rew)

            if args.onscreen:
                # Trigger render loop on underlying robosuite env.
                env.env.render()

            if video_writer is not None:
                frame = env.render()
                video_writer.append_data(frame)

            if terminated.get("__all__", False):
                print(f"Episode terminated at step {step + 1}")
                break

        print("total_rewards:", total_rewards)
        print("terminated:", terminated)
    finally:
        if video_writer is not None:
            video_writer.close()
        env.close()


if __name__ == "__main__":
    main()
