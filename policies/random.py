import gymnasium as gym
import robocasa
from pathlib import Path
from robocasa.utils.env_utils import run_random_rollouts

env = gym.make(
    "robocasa/PickPlaceCounterToCabinet",
    split="pretrain",
    seed=0,
)

# Save output under repo root (one level above this script's directory).
repo_root = Path(__file__).resolve().parents[1]
video_path = repo_root / "test.mp4"

# use env.unwrapped so run_random_rollouts can access env.sim
info = run_random_rollouts(
    env.unwrapped,
    num_rollouts=3,
    num_steps=100,
    video_path=str(video_path),
)
print(info)
