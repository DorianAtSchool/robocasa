"""GR00T N1.5 RoboCasa evaluation client."""

from __future__ import annotations

import numpy as np

from model_evals.common.backend_client import run_backend_client
from model_evals.common.robocasa_eval_client import PolicyAdapter
from model_evals.common.torch_zmq_client import TorchZmqPolicyClient


class Gr00tN15Adapter:
    """Build panda_omron observations and decode GR00T action dictionaries."""

    @staticmethod
    def _t_image(image: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(image[None, ...], dtype=np.uint8)

    @staticmethod
    def _t_state(state: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(state[None, ...], dtype=np.float32)

    def build_request(self, obs: dict, prompt: str) -> dict:
        return {
            "video.robot0_agentview_left": self._t_image(obs["video.robot0_agentview_left"]),
            "video.robot0_agentview_right": self._t_image(obs["video.robot0_agentview_right"]),
            "video.robot0_eye_in_hand": self._t_image(obs["video.robot0_eye_in_hand"]),
            "state.end_effector_position_relative": self._t_state(
                obs["state.end_effector_position_relative"]
            ),
            "state.end_effector_rotation_relative": self._t_state(
                obs["state.end_effector_rotation_relative"]
            ),
            "state.gripper_qpos": self._t_state(obs["state.gripper_qpos"]),
            "state.base_position": self._t_state(obs["state.base_position"]),
            "state.base_rotation": self._t_state(obs["state.base_rotation"]),
            "annotation.human.task_description": [prompt],
        }

    def extract_actions(self, result) -> np.ndarray:
        actions = result.get("actions", result) if isinstance(result, dict) else result
        if not isinstance(actions, dict):
            raise TypeError(f"GR00T server returned unsupported action payload: {type(actions)}")
        chunks = [
            actions["action.end_effector_position"],
            actions["action.end_effector_rotation"],
            actions["action.gripper_close"],
            actions["action.base_motion"],
            actions["action.control_mode"],
        ]
        action_chunk = np.concatenate([np.asarray(chunk) for chunk in chunks], axis=-1)
        if action_chunk.ndim == 3 and action_chunk.shape[0] == 1:
            action_chunk = action_chunk[0]
        if action_chunk.ndim != 2:
            raise ValueError(f"Expected GR00T action chunk shaped (T, D), got {action_chunk.shape}")
        return np.ascontiguousarray(action_chunk, dtype=np.float32)

    @staticmethod
    def postprocess_action(action: np.ndarray) -> np.ndarray:
        return action


_gr00t_adapter = Gr00tN15Adapter()
GR00T_N15_ADAPTER = PolicyAdapter(
    build_request=_gr00t_adapter.build_request,
    extract_actions=_gr00t_adapter.extract_actions,
    postprocess_action=_gr00t_adapter.postprocess_action,
)


if __name__ == "__main__":
    run_backend_client(
        backend_name="GR00T N1.5",
        description="GR00T N1.5 RoboCasa Evaluation Client",
        adapter=GR00T_N15_ADAPTER,
        client_factory=lambda host, port: TorchZmqPolicyClient(host, port),
    )
