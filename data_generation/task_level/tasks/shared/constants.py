"""Define shared symbolic tool categories for task-level prompting and validation."""

from __future__ import annotations

# Keep the shared FSM vocabulary small so task validators stay intuitive.
NAVIGATION_TOOL_NAMES = frozenset({"navigate_to_fixture"})
ACQUIRE_TOOL_NAMES = frozenset({"pick_up_object"})
RELEASE_TOOL_NAMES = frozenset(
    {
        "place_in_receptacle",
        "place_next_to",
        "place_on_object",
        "place_on_surface",
        "place_under",
        "place_under_dispenser",
    }
)
OBSERVATION_TOOL_NAMES = frozenset(
    {
        "get_image",
        "get_env_image",
        "get_agent_image",
    }
)
WAIT_TOOL_NAMES = frozenset({"wait"})
GIVE_SPACE_TOOL_NAMES = frozenset({"give_space"})
OPEN_PART_TOOL_NAMES = frozenset({"open_hinged_part", "open_sliding_part"})
CLOSE_PART_TOOL_NAMES = frozenset({"close_hinged_part", "close_sliding_part"})
PLACE_LOCATION_ARG_NAMES = (
    "support_id",
    "receptacle_id",
    "support_object_id",
    "dispenser_id",
)
