"""RLDX-1 RoboCasa evaluation client."""

from __future__ import annotations

import collections
import uuid

import numpy as np

from model_evals.common.backend_client import run_backend_client
from model_evals.common.robocasa_eval_client import PolicyAdapter
from model_evals.common.zmq_client import ZmqPolicyClient


class RLDXAdapter:
    """Build requests for RLDXSimPolicyWrapper and decode flat action dicts."""

    video_horizon = 4

    def __init__(self):
        self.session_id = f"rldx-{uuid.uuid4().hex[:8]}"
        self.reset_memory = True
        self._video_history = self._new_video_history()

    def _new_video_history(self):
        return {
            "video.robot0_agentview_left": collections.deque(maxlen=self.video_horizon),
            "video.robot0_agentview_right": collections.deque(maxlen=self.video_horizon),
            "video.robot0_eye_in_hand": collections.deque(maxlen=self.video_horizon),
        }

    def on_episode_start(self, env_name: str, episode_idx: int):
        self.session_id = f"{env_name}-{episode_idx}-{uuid.uuid4().hex[:8]}"
        self.reset_memory = True
        self._video_history = self._new_video_history()

    def _bt_image(self, key: str, image: np.ndarray) -> np.ndarray:
        frame = np.ascontiguousarray(image, dtype=np.uint8)
        history = self._video_history[key]
        history.append(frame)
        frames = list(history)
        if len(frames) < self.video_horizon:
            frames = [frames[0]] * (self.video_horizon - len(frames)) + frames
        return np.ascontiguousarray(np.stack(frames, axis=0)[None, ...], dtype=np.uint8)

    @staticmethod
    def _bt_state(state: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(state[None, None, ...], dtype=np.float32)

    def build_request(self, obs: dict, prompt: str) -> dict:
        observation = {
            "video.robot0_agentview_left": self._bt_image(
                "video.robot0_agentview_left", obs["video.robot0_agentview_left"]
            ),
            "video.robot0_agentview_right": self._bt_image(
                "video.robot0_agentview_right", obs["video.robot0_agentview_right"]
            ),
            "video.robot0_eye_in_hand": self._bt_image(
                "video.robot0_eye_in_hand", obs["video.robot0_eye_in_hand"]
            ),
            "state.end_effector_position_relative": self._bt_state(
                obs["state.end_effector_position_relative"]
            ),
            "state.end_effector_rotation_relative": self._bt_state(
                obs["state.end_effector_rotation_relative"]
            ),
            "state.gripper_qpos": self._bt_state(obs["state.gripper_qpos"]),
            "state.base_position": self._bt_state(obs["state.base_position"]),
            "state.base_rotation": self._bt_state(obs["state.base_rotation"]),
            "annotation.human.task_description": [prompt],
        }
        options = {
            "session_ids": [self.session_id],
            "reset_memory": [self.reset_memory],
        }
        self.reset_memory = False
        return {"observation": observation, "options": options}

    def extract_actions(self, result) -> np.ndarray:
        actions = result[0] if isinstance(result, (list, tuple)) else result
        if not isinstance(actions, dict):
            raise TypeError(f"RLDX server returned unsupported action payload: {type(actions)}")
        chunks = [
            actions["action.end_effector_position"],
            actions["action.end_effector_rotation"],
            actions["action.gripper_close"],
            actions["action.base_motion"],
            actions["action.control_mode"],
        ]
        # RLDXSimPolicyWrapper returns arrays shaped (B, T, D). Shared RoboCasa
        # rollout expects an action chunk shaped (T, action_dim).
        return np.concatenate(chunks, axis=-1)[0]

    @staticmethod
    def postprocess_action(action: np.ndarray) -> np.ndarray:
        return action


_rldx_adapter = RLDXAdapter()
RLDX_ADAPTER = PolicyAdapter(
    build_request=_rldx_adapter.build_request,
    extract_actions=_rldx_adapter.extract_actions,
    postprocess_action=_rldx_adapter.postprocess_action,
    on_episode_start=_rldx_adapter.on_episode_start,
)


if __name__ == "__main__":
    run_backend_client(
        backend_name="RLDX-1",
        description="RLDX-1 RoboCasa Evaluation Client",
        adapter=RLDX_ADAPTER,
        client_factory=lambda host, port: ZmqPolicyClient(host, port),
    )
