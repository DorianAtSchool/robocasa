"""π0.5 / openpi RoboCasa evaluation client."""

from __future__ import annotations

import numpy as np
from PIL import Image

from model_evals.common.backend_client import run_backend_client
from model_evals.common.robocasa_eval_client import PolicyAdapter


def resize_with_pad(img: np.ndarray, target_h: int = 224, target_w: int = 224) -> np.ndarray:
    """Match openpi RoboCasa eval image preprocessing without requiring openpi-client."""
    pil_img = Image.fromarray(img)
    w, h = pil_img.size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    pil_img = pil_img.resize((new_w, new_h), Image.BILINEAR)
    result = Image.new("RGB", (target_w, target_h), (0, 0, 0))
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    result.paste(pil_img, (paste_x, paste_y))
    return np.asarray(result, dtype=np.uint8)


def build_pi05_request(obs: dict, prompt: str) -> dict:
    # This state order matches robocasa-benchmark/openpi examples/robocasa/main.py.
    state = np.concatenate(
        (
            obs["state.end_effector_position_relative"],
            obs["state.end_effector_rotation_relative"],
            obs["state.base_position"],
            obs["state.base_rotation"],
            obs["state.gripper_qpos"],
        ),
        axis=0,
    )
    request = {
        "observation/image": resize_with_pad(
            np.ascontiguousarray(obs["video.robot0_agentview_left"])
        ),
        "observation/wrist_image": resize_with_pad(
            np.ascontiguousarray(obs["video.robot0_eye_in_hand"])
        ),
        "observation/state": state,
        "prompt": prompt,
    }
    if "video.robot0_agentview_right" in obs:
        request["observation/right_image"] = resize_with_pad(
            np.ascontiguousarray(obs["video.robot0_agentview_right"])
        )
    return request


def extract_pi05_actions(result: dict) -> np.ndarray:
    return result["actions"]


def identity_action(action: np.ndarray) -> np.ndarray:
    # The openpi RoboCasa example passes server actions directly to convert_action.
    return action


PI05_ADAPTER = PolicyAdapter(
    build_request=build_pi05_request,
    extract_actions=extract_pi05_actions,
    postprocess_action=identity_action,
)


if __name__ == "__main__":
    run_backend_client(
        backend_name="pi05",
        description="π0.5 / openpi RoboCasa Evaluation Client",
        adapter=PI05_ADAPTER,
    )

