"""GigaWorld-Policy RoboCasa evaluation client."""

from __future__ import annotations

import numpy as np

from model_evals.common.backend_client import run_backend_client
from model_evals.common.robocasa_eval_client import PolicyAdapter


def build_gwp_request(obs: dict, prompt: str) -> dict:
    state = np.concatenate(
        [
            obs["state.base_position"],
            obs["state.base_rotation"],
            obs["state.end_effector_position_relative"],
            obs["state.end_effector_rotation_relative"],
            obs["state.gripper_qpos"],
        ],
        axis=0,
    )
    request = {
        "observation/image": np.ascontiguousarray(obs["video.robot0_agentview_left"]),
        "observation/wrist_image": np.ascontiguousarray(obs["video.robot0_eye_in_hand"]),
        "observation/state": state,
        "prompt": prompt,
    }
    if "video.robot0_agentview_right" in obs:
        request["observation/right_image"] = np.ascontiguousarray(
            obs["video.robot0_agentview_right"]
        )
    return request


def extract_gwp_actions(result: dict) -> np.ndarray:
    return result["actions"]


def reorder_lerobot_to_hdf5(action: np.ndarray) -> np.ndarray:
    # GWP returns lerobot ordering; RoboCasa convert_action expects HDF5 ordering.
    # lerobot: [base_motion(4), control_mode(1), ee_pos(3), ee_rot(3), gripper(1)]
    # HDF5:    [ee_pos(3), ee_rot(3), gripper(1), base_motion(4), control_mode(1)]
    return np.concatenate(
        [
            action[5:8],
            action[8:11],
            action[11:12],
            action[0:4],
            action[4:5],
        ]
    )


GWP_ADAPTER = PolicyAdapter(
    build_request=build_gwp_request,
    extract_actions=extract_gwp_actions,
    postprocess_action=reorder_lerobot_to_hdf5,
)


if __name__ == "__main__":
    run_backend_client(
        backend_name="GigaWorld-Policy",
        description="GigaWorld-Policy RoboCasa Evaluation Client",
        adapter=GWP_ADAPTER,
    )
