"""Generate one concrete grounding payload from saved task-level metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from data_generation.task_level.grounding_specs import (
    build_resolved_grounding_payload,
)
from robocasa.utils.sim_tool_executor import SimToolExecutor


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses CLI arguments for the grounding-generation script."""

    parser = argparse.ArgumentParser(
        description=(
            "Resolve one saved symbolic task-level trajectory into a concrete "
            "grounding payload for a chosen scene instance."
        )
    )
    parser.add_argument(
        "--trajectory-path",
        type=Path,
        required=True,
        help="Path to one saved task-level trajectory JSON.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional output path for the resolved grounding JSON.",
    )
    parser.add_argument(
        "--scene-description-path",
        type=Path,
        default=None,
        help="Optional precomputed scene-description JSON. Overrides layout/style/seed.",
    )
    parser.add_argument("--layout", type=int, default=None)
    parser.add_argument("--style", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--robots", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Runs the grounding-generation entrypoint and returns a process exit code."""

    args = parse_args(argv)
    trajectory = _load_json(args.trajectory_path)
    scene, scene_config = _load_or_build_scene(trajectory, args)
    payload = build_resolved_grounding_payload(
        trajectory,
        scene,
        scene_config=scene_config,
    )
    output_path = args.output_path or _default_output_path(args.trajectory_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote grounding payload to {output_path}")
    return 0


def _load_json(path: Path) -> dict[str, Any]:
    """Loads one JSON file and validates that it contains a dict payload."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def _load_or_build_scene(
    trajectory: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Loads a scene description or builds one from a task scene configuration."""

    if args.scene_description_path is not None:
        scene = _load_json(args.scene_description_path)
        return scene, {"scene_description_path": str(args.scene_description_path)}

    missing_scene_args = [
        arg_name
        for arg_name in ("layout", "style", "seed")
        if getattr(args, arg_name) is None
    ]
    if missing_scene_args:
        missing_args_text = ", ".join(
            f"--{arg_name}" for arg_name in missing_scene_args
        )
        raise ValueError(
            "Either --scene-description-path or all of "
            f"{missing_args_text} must be provided."
        )

    robots = args.robots or _infer_robot_count(trajectory)
    executor = SimToolExecutor(
        task_name=trajectory["composite_task"],
        robots=robots,
        layout=args.layout,
        style=args.style,
        seed=args.seed,
    )
    try:
        scene = executor.get_scene_description()
    finally:
        executor.close()

    return scene, {
        "layout": args.layout,
        "style": args.style,
        "seed": args.seed,
        "robots": robots,
    }


def _infer_robot_count(trajectory: dict[str, Any]) -> int:
    """Infers the robot count from saved trajectory metadata."""

    agents = trajectory.get("agents")
    if isinstance(agents, list) and agents:
        return len(agents)
    return 2


def _default_output_path(trajectory_path: Path) -> Path:
    """Builds the default grounding output path for one trajectory."""

    return trajectory_path.with_name(f"{trajectory_path.stem}_grounding.json")


if __name__ == "__main__":
    raise SystemExit(main())
