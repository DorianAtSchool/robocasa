"""Adapter for external full-trajectory JSON into executor tool calls."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass
class ResolutionRecord:
    entity_type: str
    requested_id: str
    resolved_id: str
    method: str
    confidence: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_type": self.entity_type,
            "requested_id": self.requested_id,
            "resolved_id": self.resolved_id,
            "method": self.method,
            "confidence": self.confidence,
            "reason": self.reason,
        }


class TrajectoryAdapter:
    """Normalize and execute external trajectory JSON against SimToolExecutor."""

    _FIXTURE_TYPE_FAMILIES = {
        "cabinet": {
            "cabinet",
            "cabinet_single_door",
            "cabinet_double_door",
            "cabinet_with_door",
        },
        "counter": {
            "counter",
            "counter_non_dining",
            "counter_non_corner",
            "dining_counter",
            "island",
        },
        "drawer": {
            "drawer",
            "top_drawer",
        },
    }
    _FIXTURE_ARG_NAMES = {
        "anchor_fixture_id",
        "fixture_id",
        "reference_fixture_id",
        "source_id",
        "support_id",
        "target_id",
    }
    _OBJECT_ARG_NAMES = {
        "object_id",
        "reference_object_id",
        "support_object_id",
    }

    def __init__(
        self,
        executor,
        allow_approximate_ids: bool = True,
    ):
        self.executor = executor
        self.allow_approximate_ids = allow_approximate_ids
        self.scene = executor.get_scene_description()
        self._fixture_aliases: dict[str, str] = {}
        self._object_aliases: dict[str, str] = {}
        self._dispenser_aliases: dict[str, str] = {}
        self._resolution_log: list[ResolutionRecord] = []

    def adapt(
        self,
        trajectory: dict[str, Any],
        output_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        """Return a normalized executor-facing trajectory."""
        trajectory = deepcopy(trajectory)
        initial_state = trajectory.get("initial_state") or {}
        resolved_initial_state = self._adapt_initial_state(initial_state)
        tool_calls = []

        for step in trajectory.get("steps", []):
            tool_calls.append(
                self._adapt_step(
                    step,
                    resolved_initial_state=resolved_initial_state,
                    output_dir=output_dir,
                )
            )

        return {
            "trajectory_id": trajectory.get("trajectory_id"),
            "composite_task": trajectory.get("composite_task"),
            "agents": deepcopy(trajectory.get("agents", [])),
            "initial_state": resolved_initial_state,
            "tool_calls": tool_calls,
            "resolution_log": [record.to_dict() for record in self._resolution_log],
            "source_trajectory": trajectory,
        }

    def execute(
        self,
        trajectory: dict[str, Any],
        output_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        """Adapt, load initial state, then execute the normalized plan."""
        adapted = self.adapt(trajectory, output_dir=output_dir)
        load_summary = self.executor.load_initial_state(adapted["initial_state"])
        metadata = {
            "trajectory_id": adapted.get("trajectory_id"),
            "composite_task": adapted.get("composite_task"),
            "load_initial_state": load_summary,
            "resolution_log": adapted["resolution_log"],
            "steps": [],
        }

        for step in adapted["tool_calls"]:
            result = self.executor.execute(
                step["tool"],
                robot_idx=step["robot_idx"],
                **step.get("args", {}),
            )
            metadata["steps"].append(
                {
                    "step_index": step["metadata"].get("step_index"),
                    "tool": step["tool"],
                    "robot_idx": step["robot_idx"],
                    "args": deepcopy(step.get("args", {})),
                    "success": result.success,
                    "details": deepcopy(result.details),
                    "metadata": deepcopy(step.get("metadata", {})),
                }
            )

        if output_dir is not None:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            with open(output_path / "adapted_trajectory.json", "w") as f:
                json.dump(adapted, f, indent=2)
            with open(output_path / "trajectory_execution_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

        return metadata

    def _adapt_initial_state(self, initial_state: dict[str, Any]) -> dict[str, Any]:
        resolved_fixtures = {}
        resolved_objects = {}
        resolved_agents = {}
        resolved_machine_state = {}

        fixture_context = initial_state.get("fixtures", {})
        object_context = initial_state.get("objects", {})
        machine_state = initial_state.get("machine_state", {})

        for requested_fixture_id, fixture_state in fixture_context.items():
            resolved_fixture_id = self._resolve_fixture_id(
                requested_fixture_id,
                requested_fixture_state=fixture_state,
            )
            resolved_fixture_state = deepcopy(fixture_state)
            parts = {}
            for part_id, part_state in fixture_state.get("parts", {}).items():
                parts[self._normalize_part_id(part_id, part_state)] = deepcopy(part_state)
            if parts:
                resolved_fixture_state["parts"] = parts
            resolved_fixtures[resolved_fixture_id] = resolved_fixture_state

        for requested_fixture_id, machine_cfg in machine_state.items():
            resolved_fixture_id = self._resolve_fixture_id(
                requested_fixture_id,
                requested_fixture_state=fixture_context.get(requested_fixture_id),
            )
            resolved_machine_state[resolved_fixture_id] = deepcopy(machine_cfg)
            dispenser_id = machine_cfg.get("dispenser_id")
            if isinstance(dispenser_id, str):
                self._dispenser_aliases[dispenser_id] = resolved_fixture_id

        for requested_object_id, object_state in object_context.items():
            resolved_object_id = self._resolve_object_id(
                requested_object_id,
                requested_object_state=object_state,
            )
            resolved_state = deepcopy(object_state)
            location = object_state.get("location")
            if isinstance(location, str):
                resolved_state["location"] = self._resolve_fixture_id(
                    location,
                    requested_fixture_state=fixture_context.get(location),
                )
            resolved_objects[resolved_object_id] = resolved_state

        for agent_id, agent_state in initial_state.get("agents", {}).items():
            resolved_state = deepcopy(agent_state)
            location = agent_state.get("location")
            if isinstance(location, str):
                resolved_state["location"] = self._resolve_fixture_id(
                    location,
                    requested_fixture_state=fixture_context.get(location),
                )
            held_object = agent_state.get("held_object")
            if isinstance(held_object, str):
                resolved_state["held_object"] = self._resolve_object_id(
                    held_object,
                    requested_object_state=object_context.get(held_object),
                )
            resolved_agents[agent_id] = resolved_state

        return {
            "agents": resolved_agents,
            "objects": resolved_objects,
            "fixtures": resolved_fixtures,
            "machine_state": resolved_machine_state,
        }

    def _adapt_step(
        self,
        step: dict[str, Any],
        resolved_initial_state: dict[str, Any],
        output_dir: str | Path | None,
    ) -> dict[str, Any]:
        tool_name = str(step.get("tool", "")).strip()
        args = deepcopy(step.get("args", {}))
        agent_id = step.get("agent", "agent_0")
        robot_idx = self.executor._parse_agent_idx(agent_id)
        step_index = step.get("step")

        if tool_name == "place_under_dispenser":
            tool_name = "place_under"
            dispenser_id = args.pop("dispenser_id", None)
            if isinstance(dispenser_id, str):
                args["reference_fixture_id"] = self._resolve_dispenser_id(dispenser_id)

        if tool_name in {"get_env_image", "get_agent_image", "get_image"}:
            if "views" not in args and "view" in args:
                args["views"] = [args.pop("view")]
            elif isinstance(args.get("views"), str):
                args["views"] = [args["views"]]

            if tool_name == "get_agent_image":
                args.setdefault("agent_id", agent_id)

            image_paths = args.get("image_paths")
            if not (isinstance(image_paths, list) and image_paths):
                image_paths = step.get("image_paths")
            if isinstance(image_paths, list) and image_paths:
                args["image_paths"] = [
                    str(self._resolve_output_path(image_path, output_dir))
                    for image_path in image_paths
                ]
            else:
                image_path = args.pop("image_path", None)
                if not isinstance(image_path, str):
                    image_path = step.get("image_path")
                if isinstance(image_path, str):
                    args["image_paths"] = [
                        str(self._resolve_output_path(image_path, output_dir))
                    ]
            tool_name = "get_image"

        args = self._resolve_step_args(
            tool_name,
            args,
            resolved_initial_state=resolved_initial_state,
        )

        return {
            "tool": tool_name,
            "robot_idx": robot_idx,
            "args": args,
            "metadata": {
                "step_index": step_index,
                "source_agent": agent_id,
                "reasoning": step.get("reasoning"),
                "image_path": step.get("image_path"),
                "image_paths": deepcopy(step.get("image_paths")),
            },
        }

    def _resolve_step_args(
        self,
        tool_name: str,
        args: dict[str, Any],
        resolved_initial_state: dict[str, Any],
    ) -> dict[str, Any]:
        resolved_args = deepcopy(args)
        for arg_name, value in list(resolved_args.items()):
            if not isinstance(value, str):
                continue
            if arg_name in self._FIXTURE_ARG_NAMES:
                resolved_args[arg_name] = self._resolve_fixture_id(
                    value,
                    requested_fixture_state=(
                        resolved_initial_state.get("fixtures", {}).get(value)
                        or None
                    ),
                )
            elif arg_name in self._OBJECT_ARG_NAMES:
                resolved_args[arg_name] = self._resolve_object_id(
                    value,
                    requested_object_state=(
                        resolved_initial_state.get("objects", {}).get(value)
                        or None
                    ),
                )
            elif arg_name == "receptacle_id":
                if value in self.scene.get("fixtures", {}) or value in self._fixture_aliases:
                    resolved_args[arg_name] = self._resolve_fixture_id(value)
                else:
                    resolved_args[arg_name] = self._resolve_object_id(value)
            elif arg_name == "part_id":
                fixture_key = resolved_args.get("target_id")
                resolved_args[arg_name] = self._normalize_part_id(
                    value,
                    (
                        resolved_initial_state.get("fixtures", {})
                        .get(fixture_key, {})
                        .get("parts", {})
                        .get(value, {})
                    ),
                )
        return resolved_args

    def _resolve_output_path(
        self,
        image_path: str,
        output_dir: str | Path | None,
    ) -> Path:
        path = Path(image_path)
        if path.is_absolute() or output_dir is None:
            return path
        return Path(output_dir) / path

    def _resolve_dispenser_id(self, dispenser_id: str) -> str:
        if dispenser_id in self._dispenser_aliases:
            return self._dispenser_aliases[dispenser_id]
        if dispenser_id in self.scene.get("fixtures", {}):
            return dispenser_id
        raise ValueError(f"Unknown dispenser_id: {dispenser_id!r}")

    def _resolve_fixture_id(
        self,
        requested_id: str,
        requested_fixture_state: dict[str, Any] | None = None,
    ) -> str:
        if requested_id in self._fixture_aliases:
            return self._fixture_aliases[requested_id]
        if requested_id in self.scene.get("fixtures", {}):
            self._fixture_aliases[requested_id] = requested_id
            return requested_id

        fixture_type = None
        if requested_fixture_state is not None:
            fixture_type = requested_fixture_state.get("fixture_type")
        candidate_ids = self._fixture_candidates_for_type(fixture_type)
        resolved_id, method, confidence, reason = self._choose_candidate(
            requested_id=requested_id,
            candidate_ids=candidate_ids,
            entity_type="fixture",
            type_hint=fixture_type,
        )
        self._fixture_aliases[requested_id] = resolved_id
        self._resolution_log.append(
            ResolutionRecord(
                entity_type="fixture",
                requested_id=requested_id,
                resolved_id=resolved_id,
                method=method,
                confidence=confidence,
                reason=reason,
            )
        )
        return resolved_id

    def _resolve_object_id(
        self,
        requested_id: str,
        requested_object_state: dict[str, Any] | None = None,
    ) -> str:
        if requested_id in self._object_aliases:
            return self._object_aliases[requested_id]
        if requested_id in self.scene.get("objects", {}):
            self._object_aliases[requested_id] = requested_id
            return requested_id

        object_type = None
        if requested_object_state is not None:
            object_type = requested_object_state.get("object_type")
        candidate_ids = self._object_candidates_for_type(object_type)
        resolved_id, method, confidence, reason = self._choose_candidate(
            requested_id=requested_id,
            candidate_ids=candidate_ids,
            entity_type="object",
            type_hint=object_type,
        )
        self._object_aliases[requested_id] = resolved_id
        self._resolution_log.append(
            ResolutionRecord(
                entity_type="object",
                requested_id=requested_id,
                resolved_id=resolved_id,
                method=method,
                confidence=confidence,
                reason=reason,
            )
        )
        return resolved_id

    def _fixture_candidates_for_type(self, fixture_type: str | None) -> list[str]:
        fixtures = self.scene.get("fixtures", {})
        if fixture_type is None:
            return sorted(fixtures.keys())
        fixture_type = str(fixture_type).lower()
        return sorted(
            fixture_id
            for fixture_id, fixture_info in fixtures.items()
            if self._fixture_type_matches(
                fixture_type,
                str(fixture_info.get("fixture_type", "")).lower(),
            )
        )

    def _object_candidates_for_type(self, object_type: str | None) -> list[str]:
        objects = self.scene.get("objects", {})
        if object_type is None:
            return sorted(objects.keys())
        object_type = str(object_type).lower()
        matches = []
        for object_id, object_info in objects.items():
            actual_type = str(object_info.get("object_type", "")).lower()
            if actual_type == object_type or object_type in actual_type:
                matches.append(object_id)
        return sorted(matches)

    def _choose_candidate(
        self,
        requested_id: str,
        candidate_ids: list[str],
        entity_type: str,
        type_hint: str | None,
    ) -> tuple[str, str, float, str]:
        if not candidate_ids:
            raise ValueError(
                f"Unable to resolve {entity_type}_id {requested_id!r} with type hint {type_hint!r}"
            )

        if len(candidate_ids) == 1:
            return (
                candidate_ids[0],
                "type_match",
                0.9,
                f"Only candidate matching type hint {type_hint!r}",
            )

        exact_like = [
            candidate_id for candidate_id in candidate_ids
            if self._base_token(candidate_id) == self._base_token(requested_id)
        ]
        if len(exact_like) == 1:
            return (
                exact_like[0],
                "token_match",
                0.8,
                f"Matched normalized token for {requested_id!r}",
            )

        ordinal = self._extract_ordinal(requested_id)
        if ordinal is not None:
            ordinal_idx = ordinal - 1
            ordered = exact_like or candidate_ids
            if 0 <= ordinal_idx < len(ordered):
                return (
                    ordered[ordinal_idx],
                    "ordinal_guess",
                    0.6,
                    f"Selected candidate #{ordinal} among {len(ordered)} candidates",
                )

        if not self.allow_approximate_ids:
            raise ValueError(
                f"Ambiguous {entity_type}_id {requested_id!r}; candidates={candidate_ids}"
            )

        return (
            candidate_ids[0],
            "fallback_first_candidate",
            0.3,
            f"Fell back to first sorted candidate among {len(candidate_ids)} matches",
        )

    def _normalize_part_id(
        self,
        part_id: str,
        part_state: dict[str, Any] | None = None,
    ) -> str:
        part_id = str(part_id)
        part_type = ""
        if part_state is not None:
            part_type = str(part_state.get("part_type", "")).lower()
        if "hinged" in part_type:
            return "hinged"
        if "sliding" in part_type:
            return "sliding"
        lowered = part_id.lower()
        if lowered in {"door", "lid"}:
            return "hinged"
        return part_id

    def _base_token(self, token: str) -> str:
        token = str(token).lower().strip()
        parts = [part for part in token.split("_") if part and not part.isdigit()]
        return "_".join(parts)

    def _extract_ordinal(self, token: str) -> int | None:
        pieces = str(token).split("_")
        if not pieces:
            return None
        tail = pieces[-1]
        if tail.isdigit():
            return int(tail)
        return None

    def _fixture_type_matches(
        self,
        requested_type: str,
        actual_type: str,
    ) -> bool:
        if requested_type == actual_type:
            return True

        requested_family = self._FIXTURE_TYPE_FAMILIES.get(requested_type)
        if requested_family is not None and actual_type in requested_family:
            return True

        actual_family = self._FIXTURE_TYPE_FAMILIES.get(actual_type)
        if actual_family is not None and requested_type in actual_family:
            return True

        return requested_type in actual_type or actual_type in requested_type


def adapt_trajectory(
    executor,
    trajectory: dict[str, Any],
    output_dir: str | Path | None = None,
    allow_approximate_ids: bool = True,
) -> dict[str, Any]:
    """Convenience wrapper around :class:`TrajectoryAdapter`."""
    return TrajectoryAdapter(
        executor=executor,
        allow_approximate_ids=allow_approximate_ids,
    ).adapt(trajectory, output_dir=output_dir)


def execute_trajectory(
    executor,
    trajectory: dict[str, Any],
    output_dir: str | Path | None = None,
    allow_approximate_ids: bool = True,
) -> dict[str, Any]:
    """Adapt and execute one external trajectory."""
    return TrajectoryAdapter(
        executor=executor,
        allow_approximate_ids=allow_approximate_ids,
    ).execute(trajectory, output_dir=output_dir)


__all__ = [
    "ResolutionRecord",
    "TrajectoryAdapter",
    "adapt_trajectory",
    "execute_trajectory",
]
