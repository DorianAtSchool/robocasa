"""
Generate compact task descriptions and prompt text for external LLM planning.

The generated context is intentionally task-focused and alias-based so it stays
cheap to send to an external LLM while still being mappable back to the live
scene symbols that the simulator expects.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
from typing import Any

from robocasa.utils.sim_tool_executor import SimToolExecutor
from robocasa.utils.sim_tool_specs import get_sim_tool_specs


_DEFAULT_RULES = [
    "Use only the exact ids listed in this context.",
    "Do not invent new objects, fixtures, parts, controls, or tools.",
    "Use fixture ids for fixture-valued arguments.",
    "Use object ids for object-valued arguments.",
    "Use only the listed parts and controls for each fixture id.",
    "Use communicate before shared-space convergence when coordination matters.",
    "Use wait when one robot should yield while another completes a blocking step.",
    "Return valid JSON only.",
]

_COMMON_SUPPORT_OBJECT_IDS = {"plate", "bowl", "tray", "mug", "cup", "saucer"}
_SURFACE_FIXTURE_TYPES = {
    "counter",
    "counter_non_dining",
    "counter_non_corner",
    "dining_counter",
    "island",
    "sink",
    "stove",
}
_RECEPTACLE_FIXTURE_TYPES = {
    "cabinet",
    "cabinet_with_door",
    "cabinet_single_door",
    "cabinet_double_door",
    "drawer",
    "top_drawer",
    "fridge",
    "microwave",
    "oven",
    "dish_rack",
    "dishwasher",
}
_DISPENSER_FIXTURE_TYPES = {"coffee_machine"}
_FIXTURE_TYPE_KEYWORDS = {
    "fridge": "fridge",
    "cabinet": "cabinet",
    "drawer": "drawer",
    "microwave": "microwave",
    "coffee_machine": "coffee",
    "stove": "stove",
    "sink": "sink",
    "dishwasher": "dishwasher",
    "toaster": "toaster",
    "oven": "oven",
    "dining_counter": "dining",
}
_PREFERRED_SUPPORT_FIXTURE_TYPES = {
    "dining_counter",
    "counter",
    "counter_non_dining",
    "counter_non_corner",
    "island",
}
_APPLIANCE_FIXTURE_TYPES = {
    "toaster",
    "toaster_oven",
    "microwave",
    "coffee_machine",
    "stove",
    "oven",
    "dishwasher",
}


def list_composite_task_names() -> list[str]:
    """Return all composite Kitchen task class names by scanning the source tree."""
    root = Path(__file__).resolve().parents[1] / "environments" / "kitchen" / "composite"
    task_names = []
    for path in root.rglob("*.py"):
        text = path.read_text()
        match = re.search(r"class\s+(\w+)\(Kitchen\):", text)
        if match:
            task_names.append(match.group(1))
    return sorted(set(task_names))


def _normalize_tool_spec(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": spec["name"],
        "parameters": [param["name"] for param in spec["parameters"]],
    }


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _short_parts(parts: list[str]) -> list[str]:
    short = []
    for token in ("hinged", "sliding", "lid", "head"):
        if token in parts:
            short.append(token)
    return short if short else parts[:2]


def _score_object(instruction_tokens: set[str], object_id: str, object_type: str) -> int:
    score = 0
    if instruction_tokens & _tokenize(object_id):
        score += 3
    if instruction_tokens & _tokenize(object_type):
        score += 2
    if object_id in _COMMON_SUPPORT_OBJECT_IDS:
        score += 2
    return score


def _select_relevant_objects(objects: dict[str, Any], instruction: str) -> list[dict[str, Any]]:
    instruction_tokens = _tokenize(instruction or "")
    scored = []
    for object_id, info in sorted(objects.items()):
        score = _score_object(instruction_tokens, object_id, info["object_type"])
        scored.append(
            (
                object_id,
                score,
                {
                    "object_id": object_id,
                    "object_type": info["object_type"],
                    "location": info["location"],
                },
            )
        )

    selected = [item for _, score, item in scored if score > 0]
    if not selected:
        selected = [item for _, _, item in scored[:6]]

    filtered = []
    for item in selected:
        if item["object_id"].endswith("_container"):
            has_named_support = any(
                other["object_id"] in _COMMON_SUPPORT_OBJECT_IDS
                and other["object_type"] == item["object_type"]
                for other in selected
            )
            if has_named_support:
                continue
        filtered.append(item)

    if not any(item["object_id"] in _COMMON_SUPPORT_OBJECT_IDS for item in filtered):
        for _, _, item in scored:
            if item["object_id"] in _COMMON_SUPPORT_OBJECT_IDS:
                filtered.append(item)
                break

    return filtered


def _normalize_object_location(
    executor: SimToolExecutor,
    scene_fixtures: dict[str, Any],
    obj: dict[str, Any],
    instruction: str,
) -> str:
    location = obj["location"]
    if location not in scene_fixtures:
        return executor._find_nearest_fixture_for_object(obj["object_id"])

    fixture_type = scene_fixtures[location]["fixture_type"]
    if obj["object_id"] in _COMMON_SUPPORT_OBJECT_IDS and fixture_type in {"stool", "window"}:
        preferred_types = {"dining_counter"} if "dining" in (instruction or "").lower() else _PREFERRED_SUPPORT_FIXTURE_TYPES
        return executor._find_nearest_fixture_for_object(
            obj["object_id"],
            preferred_fixture_types=preferred_types,
        )

    return location


def _select_relevant_fixture_ids(
    scene_fixtures: dict[str, Any],
    relevant_objects: list[dict[str, Any]],
    instruction: str,
) -> list[str]:
    selected_ids = set()
    instruction_lower = (instruction or "").lower()

    for obj in relevant_objects:
        if obj["location"] in scene_fixtures:
            selected_ids.add(obj["location"])

    for fixture_id, info in scene_fixtures.items():
        fixture_type = info["fixture_type"]
        keyword = _FIXTURE_TYPE_KEYWORDS.get(fixture_type)
        if keyword and keyword in instruction_lower:
            selected_ids.add(fixture_id)
            if fixture_type in _APPLIANCE_FIXTURE_TYPES:
                for nearby in info.get("nearby_fixtures", []):
                    nearby_info = scene_fixtures.get(nearby)
                    if nearby_info is None:
                        continue
                    if nearby_info["fixture_type"] in _SURFACE_FIXTURE_TYPES:
                        selected_ids.add(nearby)

    expanded_ids = set(selected_ids)
    for fixture_id in list(selected_ids):
        for nearby in scene_fixtures[fixture_id].get("nearby_fixtures", []):
            if nearby in selected_ids:
                expanded_ids.add(nearby)

    return sorted(expanded_ids)


def build_compact_task_context(
    task_name: str,
    robots: int = 2,
    layout: int | None = None,
    style: int | None = None,
    seed: int | None = 42,
    width: int = 320,
    height: int = 240,
    gl_backend: str = "osmesa",
) -> dict[str, Any]:
    """Build a compact, planner-facing context for one task instance."""
    executor = SimToolExecutor(
        task_name=task_name,
        robots=robots,
        layout=layout,
        style=style,
        seed=seed,
        render_width=width,
        render_height=height,
        gl_backend=gl_backend,
    )
    try:
        scene = executor.get_scene_description()
        instruction = scene.get("task") or ""
        scene_fixtures = scene.get("fixtures", {})
        scene_objects = scene.get("objects", {})

        relevant_objects = _select_relevant_objects(scene_objects, instruction)
        relevant_objects = [
            {
                **obj,
                "location": _normalize_object_location(
                    executor,
                    scene_fixtures,
                    obj,
                    instruction,
                ),
            }
            for obj in relevant_objects
        ]
        selected_fixture_ids = _select_relevant_fixture_ids(
            scene_fixtures,
            relevant_objects,
            instruction,
        )

        compact_fixtures = []
        for fixture_id in selected_fixture_ids:
            info = scene_fixtures[fixture_id]
            compact_fixtures.append(
                {
                    "fixture_id": fixture_id,
                    "fixture_type": info["fixture_type"],
                    "parts": _short_parts(executor.get_parts(fixture_id)),
                    "controls": executor.get_controls(fixture_id),
                    "nearby": [
                        nearby
                        for nearby in info.get("nearby_fixtures", [])
                        if nearby in selected_fixture_ids
                    ],
                }
            )

        compact_objects = []
        for obj in relevant_objects:
            compact_objects.append(
                {
                    "object_id": obj["object_id"],
                    "object_type": obj["object_type"],
                    "location": obj["location"],
                }
            )

        return {
            "task_name": task_name,
            "instruction": instruction,
            "robots": [f"agent_{idx}" for idx in range(robots)],
            "tools": [_normalize_tool_spec(spec) for spec in get_sim_tool_specs()],
            "objects": compact_objects,
            "fixtures": compact_fixtures,
            "rules": list(_DEFAULT_RULES),
            "scene_parameters": {
                "layout": int(executor.env.layout_id),
                "style": int(executor.env.style_id),
                "seed": seed,
            },
        }
    finally:
        executor.close()


def render_llm_prompt(context: dict[str, Any]) -> str:
    """Render a compact text prompt from the structured context."""
    example_fixture_id = (
        context["fixtures"][0]["fixture_id"] if context.get("fixtures") else "fixture_id_here"
    )
    lines = [
        "Plan a multi-robot symbolic kitchen trajectory.",
        "Return JSON only.",
        f"Task: {context['instruction']}",
        "",
        f"Robots: {', '.join(context['robots'])}",
        "",
        "Tools:",
    ]

    for tool in context["tools"]:
        params = ", ".join(tool["parameters"])
        lines.append(f"- {tool['name']}({params})" if params else f"- {tool['name']}()")

    lines.extend(["", "Objects:"])
    for obj in context["objects"]:
        lines.append(
            f"- {obj['object_id']} ({obj['object_type']}) @ {obj['location']}"
        )

    lines.extend(["", "Fixtures:"])
    for fixture in context["fixtures"]:
        extras = []
        if fixture["parts"]:
            extras.append(f"parts={fixture['parts']}")
        if fixture["controls"]:
            extras.append(f"controls={fixture['controls']}")
        if fixture["nearby"]:
            extras.append(f"nearby={fixture['nearby']}")
        suffix = f", {', '.join(extras)}" if extras else ""
        lines.append(
            f"- {fixture['fixture_id']} ({fixture['fixture_type']}){suffix}"
        )

    lines.extend(["", "Rules:"])
    for rule in context["rules"]:
        lines.append(f"- {rule}")

    lines.extend(
        [
            "",
            "Output format:",
            "[",
            '  {"tool": "communicate", "robot_idx": 0, "args": {"to": "agent_1", "message": "..."}},',
            f'  {{"tool": "navigate_to_fixture", "robot_idx": 0, "args": {{"fixture_id": "{example_fixture_id}"}}}}',
            "]",
        ]
    )
    return "\n".join(lines)


def _write_outputs(output_dir: Path, context: dict[str, Any], prompt_text: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "compact_context.json", "w") as f:
        json.dump(context, f, indent=2)
    with open(output_dir / "prompt.txt", "w") as f:
        f.write(prompt_text)


def _main():
    parser = argparse.ArgumentParser(
        description="Generate compact task descriptions and prompt text for external LLM planners."
    )
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--robots", type=int, default=2)
    parser.add_argument("--layout", type=int, default=None)
    parser.add_argument("--style", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--gl-backend", type=str, default="osmesa")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="Print all composite task names and exit.",
    )
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", args.gl_backend)

    if args.list_tasks:
        for task_name in list_composite_task_names():
            print(task_name)
        return

    if args.task is None:
        parser.error("--task is required unless --list-tasks is used")
    if args.output_dir is None:
        parser.error("--output-dir is required when generating descriptions")

    context = build_compact_task_context(
        task_name=args.task,
        robots=args.robots,
        layout=args.layout,
        style=args.style,
        seed=args.seed,
        width=args.width,
        height=args.height,
        gl_backend=args.gl_backend,
    )
    prompt_text = render_llm_prompt(context)
    output_dir = Path(args.output_dir)
    _write_outputs(output_dir, context, prompt_text)
    print(
        json.dumps(
            {
                "task_name": context["task_name"],
                "output_dir": str(output_dir),
                "files": [
                    str(output_dir / "compact_context.json"),
                    str(output_dir / "prompt.txt"),
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    _main()
