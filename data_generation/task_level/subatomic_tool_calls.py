"""Define the shared subatomic tool-call catalog used by task-level prompts."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class SubatomicToolArg:
    """Represents one prompt-facing argument for a shared subatomic tool."""

    name: str


@dataclass(frozen=True)
class SubatomicToolSpec:
    """Stores prompt-facing metadata for a shared subatomic tool."""

    name: str
    description: str
    constructor_args: tuple[SubatomicToolArg, ...]

    def to_prompt_block(self) -> str:
        """Renders a compact prompt entry for a single tool."""

        if self.constructor_args:
            arg_text = ", ".join(arg.name for arg in self.constructor_args)
        else:
            arg_text = "no explicit constructor inputs"
        return f"- {self.name}: {self.description} Inputs: {arg_text}."

def _build_tool_spec(
    name: str,
    description: str,
    *arg_names: str,
) -> SubatomicToolSpec:
    """Builds a static subatomic tool specification."""

    constructor_args = tuple(SubatomicToolArg(name=arg_name) for arg_name in arg_names)
    return SubatomicToolSpec(
        name=name,
        description=description,
        constructor_args=constructor_args,
    )


# Keep the shared subatomic catalog static so prompt construction is predictable.
SUBATOMIC_TOOL_SPECS: tuple[SubatomicToolSpec, ...] = (
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
        "place_on_object",
        "Place an object on top of another movable support object.",
        "object_id",
        "support_object_id",
    ),
    _build_tool_spec(
        "place_on_surface",
        "Place an object onto an exposed support surface or attachment seat.",
        "object_id",
        "support_id",
    ),
    _build_tool_spec(
        "place_under_dispenser",
        "Place an object under a machine dispenser target.",
        "object_id",
        "dispenser_id",
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
)


@lru_cache(maxsize=1)
def discover_subatomic_tools() -> tuple[SubatomicToolSpec, ...]:
    """Returns the shared subatomic catalog used by task-level generation."""

    return tuple(sorted(SUBATOMIC_TOOL_SPECS, key=lambda tool: tool.name))


def render_subatomic_tool_catalog(
    tool_specs: tuple[SubatomicToolSpec, ...] | None = None,
) -> str:
    """Renders the subatomic tool catalog for inclusion in task prompts."""

    if tool_specs is None:
        tool_specs = discover_subatomic_tools()
    return "\n".join(tool.to_prompt_block() for tool in tool_specs)
