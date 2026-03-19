"""
Primitive simulator tool specifications for symbolic kitchen planning.

These specs are intentionally lightweight: they describe the simulator-facing
tool surface, not the higher-level semantic task decomposition that may sit on
top of it.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def _build_tool_spec(name: str, description: str, *arg_names: str) -> dict[str, Any]:
    """Build a minimal tool spec with required string parameters."""
    return {
        "name": name,
        "description": description,
        "parameters": [
            {
                "name": arg_name,
                "type": "string",
                "required": True,
            }
            for arg_name in arg_names
        ],
    }


SIM_TOOL_SPECS: list[dict[str, Any]] = [
    _build_tool_spec(
        "get_image",
        "Capture and save one or more images from named views.",
        "views",
        "image_paths",
    ),
    _build_tool_spec(
        "close_hinged_part",
        "Close a hinged door, lid, or articulated head on a fixture.",
        "target_id",
        "part_id",
    ),
    _build_tool_spec(
        "close_sliding_part",
        "Push in a sliding drawer or rack on a fixture.",
        "target_id",
        "part_id",
    ),
    _build_tool_spec(
        "communicate",
        "Send a coordination message to another robot agent.",
        "to",
        "message",
    ),
    _build_tool_spec(
        "navigate_to_fixture",
        "Move the robot base to the working pose of a fixture.",
        "fixture_id",
    ),
    _build_tool_spec(
        "open_hinged_part",
        "Open a hinged door, lid, or articulated head on a fixture.",
        "target_id",
        "part_id",
    ),
    _build_tool_spec(
        "open_sliding_part",
        "Pull out a sliding drawer or rack on a fixture.",
        "target_id",
        "part_id",
    ),
    _build_tool_spec(
        "pick_up_object",
        "Grasp and lift an object from a symbolic source region.",
        "object_id",
        "source_id",
    ),
    _build_tool_spec(
        "place_in_receptacle",
        "Place an object into an interior or container-like receptacle.",
        "object_id",
        "receptacle_id",
    ),
    _build_tool_spec(
        "place_next_to",
        "Place an object adjacent to another object on the same surface.",
        "object_id",
        "reference_object_id",
    ),
    _build_tool_spec(
        "place_on_object",
        "Place an object on top of another movable support object.",
        "object_id",
        "support_object_id",
        "anchor_fixture_id",
    ),
    _build_tool_spec(
        "place_on_surface",
        "Place an object onto an exposed support surface or attachment seat.",
        "object_id",
        "support_id",
    ),
    _build_tool_spec(
        "place_under",
        "Place an object directly beneath a reference fixture. For dispensers "
        "(e.g. coffee machine nozzle, sink faucet) the object is positioned at "
        "the dispenser output site; for other fixtures (e.g. wall cabinet) the "
        "object is placed on the nearest surface below.",
        "object_id",
        "reference_fixture_id",
    ),
    _build_tool_spec(
        "press_button",
        "Activate a discrete button-like control on a fixture.",
        "target_id",
        "control_id",
    ),
    _build_tool_spec(
        "press_lever",
        "Activate a discrete lever-like control on a fixture.",
        "target_id",
        "control_id",
    ),
    _build_tool_spec(
        "set_rotary_control",
        "Set a rotary or handle-like control to an explicit goal state.",
        "target_id",
        "control_id",
        "goal",
    ),
    _build_tool_spec(
        "give_space",
        "Move away from a fixture so another robot can access it.",
        "fixture_id",
    ),
    _build_tool_spec(
        "wait",
        "Pause and observe while another robot completes its step.",
    ),
]

SIM_TOOL_SPEC_BY_NAME = {spec["name"]: spec for spec in SIM_TOOL_SPECS}


def get_sim_tool_specs() -> list[dict[str, Any]]:
    """Return a defensive copy of the simulator tool specs."""
    return deepcopy(SIM_TOOL_SPECS)


def get_sim_tool_spec(name: str) -> dict[str, Any]:
    """Return a defensive copy of a single simulator tool spec."""
    return deepcopy(SIM_TOOL_SPEC_BY_NAME[name])


__all__ = [
    "SIM_TOOL_SPECS",
    "SIM_TOOL_SPEC_BY_NAME",
    "get_sim_tool_specs",
    "get_sim_tool_spec",
]
