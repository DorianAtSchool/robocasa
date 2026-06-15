"""Launch GR00T N1.5 with the Isaac-GR00T RobotInferenceServer."""

from __future__ import annotations

import argparse

from gr00t.eval.robot import RobotInferenceServer
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy


def parse_args():
    parser = argparse.ArgumentParser(description="GR00T N1.5 RoboCasa inference server")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-config", default="panda_omron")
    parser.add_argument("--embodiment-tag", default="new_embodiment")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--denoising-steps", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    data_config = DATA_CONFIG_MAP[args.data_config]
    policy = Gr00tPolicy(
        model_path=args.model_path,
        modality_config=data_config.modality_config(),
        modality_transform=data_config.transform(),
        embodiment_tag=args.embodiment_tag,
        denoising_steps=args.denoising_steps,
    )
    server = RobotInferenceServer(policy, host=args.host, port=args.port)
    server.run()


if __name__ == "__main__":
    main()
