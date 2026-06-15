"""Camera selection and frame capture for RoboCasa VLA evaluation."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

CameraEntry = tuple[str, str, str]


def available_sim_cameras(env) -> set[str]:
    sim = getattr(getattr(env, "env", env), "sim", None)
    model = getattr(sim, "model", None)
    names = getattr(model, "camera_names", ())
    return {str(name) for name in names}


def _unique(entries: list[CameraEntry]) -> list[CameraEntry]:
    seen = set()
    unique_entries = []
    for entry in entries:
        if entry[2] in seen:
            continue
        seen.add(entry[2])
        unique_entries.append(entry)
    return unique_entries


def resolve_record_cameras(env, obs: dict, cameras: str) -> list[CameraEntry]:
    obs_entries = [
        ("obs", "video.robot0_agentview_left", "robot0_agentview_left"),
        ("obs", "video.robot0_agentview_right", "robot0_agentview_right"),
        ("obs", "video.robot0_eye_in_hand", "robot0_eye_in_hand"),
    ]
    render_candidates = [
        "robot0_agentview_center",
        "robot1_agentview_center",
        "robot1_agentview_left",
        "robot1_agentview_right",
        "robot1_eye_in_hand",
        "robot0_robotview",
        "robot1_robotview",
        "birdview",
        "topview",
        "top_view",
        "room_view",
        "frontview",
        "agentview",
    ]
    sim_cameras = available_sim_cameras(env)

    def first_available_sim(names):
        for name in names:
            if name in sim_cameras:
                return ("sim", name, name)
        return None

    def first_available_obs(entries):
        for entry in entries:
            if entry[1] in obs:
                return entry
        return None

    if cameras in {"", "default"}:
        entries = []
        robot0_default = first_available_sim(
            ["robot0_agentview_center", "robot0_robotview", "agentview"]
        ) or first_available_obs(obs_entries[:2])
        robot0_wrist = first_available_obs(
            [("obs", "video.robot0_eye_in_hand", "robot0_eye_in_hand")]
        ) or first_available_sim(["robot0_eye_in_hand"])
        robot1_default = first_available_sim(
            [
                "robot1_agentview_center",
                "robot1_robotview",
                "robot1_agentview_left",
                "robot1_agentview_right",
                "robot1_eye_in_hand",
            ]
        )

        for entry in (robot0_default, robot0_wrist, robot1_default):
            if entry is not None:
                entries.append(entry)
        return _unique(entries) or [("render", "default", "render")]

    if cameras == "render":
        return [("render", "default", "render")]

    if cameras == "all":
        entries = [entry for entry in obs_entries if entry[1] in obs]
        entries.extend(
            ("sim", camera_name, camera_name)
            for camera_name in render_candidates
            if camera_name in sim_cameras
        )
        return _unique(entries) or [("render", "default", "render")]

    entries = []
    aliases = {label: (kind, key, label) for kind, key, label in obs_entries}
    for raw_name in cameras.split(","):
        name = raw_name.strip()
        if not name:
            continue
        if name in aliases and aliases[name][1] in obs:
            entries.append(aliases[name])
        elif name in obs:
            entries.append(("obs", name, name.replace("video.", "")))
        elif name in sim_cameras:
            entries.append(("sim", name, name))
        elif name == "render":
            entries.append(("render", "default", "render"))
        else:
            logger.warning("Requested camera %s is not available; skipping.", name)
    return entries or [("render", "default", "render")]


def capture_frame(env, obs: dict, camera_entry: CameraEntry) -> np.ndarray | None:
    kind, key, _ = camera_entry
    try:
        if kind == "obs":
            return np.ascontiguousarray(obs[key])
        if kind == "sim":
            frame = env.env.sim.render(height=512, width=768, camera_name=key)[::-1]
            return np.ascontiguousarray(frame)
        frame = env.render()
        return np.ascontiguousarray(frame)
    except Exception as exc:
        logger.warning("Unable to capture camera %s: %s", key, exc)
        return None

