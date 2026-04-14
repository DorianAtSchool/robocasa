"""Phase 2: Static TaskSpec validation.

Catches structural problems in generated specs before spending API credits
on Phase 3 trajectory generation. Each spec is checked for:

  1. Schema — `TaskSpec.from_dict(payload)` succeeds.
  2. Supported kinds — every goal/precondition/effect uses a kind the
     runtime FSM understands.
  3. Referential integrity — every id referenced by goals, preconditions,
     effects, allowed_tool_specs, or example_trajectory exists in
     `initial_state`.
  4. Trajectory dry-run — replay `example_trajectory` through
     `SpecDrivenTaskValidator.validate()`. Any TrajectoryValidationError
     means the spec disagrees with its own example trajectory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from data_generation.task_level.tasks.specs import TaskSpec
from data_generation.task_level.tasks.specs.runtime import (
    SpecDrivenTaskValidator,
    build_task_definition_from_spec,
)
from data_generation.task_level.tasks.shared.errors import TrajectoryValidationError

from .sim_normalization import collect_simulation_alignment_errors


SUPPORTED_GOAL_KINDS = frozenset(
    {
        "object_at_location",
        "object_count_at_location",
        "object_at_location_one_of",
        "machine_flag_true",
        "machine_flag_equals",
        "fixture_part_state",
        "fixture_control_state",
    }
)
SUPPORTED_PRECONDITION_KINDS = frozenset(
    {
        "object_must_remain_at_location",
        "fixture_part_state_required_for_pickup",
        "fixture_part_state_required_for_action",
        "object_location_required_for_action",
    }
)
SUPPORTED_EFFECT_KINDS = frozenset({"set_machine_flag_on_action"})
_CANONICAL_PART_STATES = frozenset({"open", "closed"})
_LEGACY_PART_STATE_ALIASES = {"pulled_out": "open", "pushed_in": "closed"}


def _parse_visited_machine_path(
    machine_path: Any,
) -> tuple[str, str] | None:
    """Return (fixture_id, machine_key) for `<control>_<state>_visited` paths."""

    if not isinstance(machine_path, list) or len(machine_path) != 2:
        return None
    fixture_id, machine_key = machine_path
    if not isinstance(fixture_id, str) or not isinstance(machine_key, str):
        return None
    if not machine_key.endswith("_visited") or "_" not in machine_key[:-8]:
        return None
    return fixture_id, machine_key


@dataclass
class SpecValidationResult:
    """Per-spec validation outcome with structured error details."""

    task_name: str
    spec_path: str
    passed: bool
    schema_error: str | None = None
    unsupported_kinds: list[str] = field(default_factory=list)
    goal_consistency_errors: list[str] = field(default_factory=list)
    grounding_errors: list[str] = field(default_factory=list)
    simulation_errors: list[str] = field(default_factory=list)
    referential_errors: list[str] = field(default_factory=list)
    dry_run_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_name": self.task_name,
            "spec_path": self.spec_path,
            "passed": self.passed,
            "schema_error": self.schema_error,
            "unsupported_kinds": list(self.unsupported_kinds),
            "goal_consistency_errors": list(self.goal_consistency_errors),
            "grounding_errors": list(self.grounding_errors),
            "simulation_errors": list(self.simulation_errors),
            "referential_errors": list(self.referential_errors),
            "dry_run_error": self.dry_run_error,
        }


def _collect_unsupported_kinds(payload: dict[str, Any]) -> list[str]:
    """Return error strings for any goal/precondition/effect using an unknown kind."""

    errors: list[str] = []
    for goal in payload.get("goal_conditions", []) or []:
        kind = goal.get("kind") if isinstance(goal, dict) else None
        if kind not in SUPPORTED_GOAL_KINDS:
            errors.append(f"goal_conditions: unsupported kind {kind!r}")
    for cond in payload.get("task_preconditions", []) or []:
        kind = cond.get("kind") if isinstance(cond, dict) else None
        if kind not in SUPPORTED_PRECONDITION_KINDS:
            errors.append(f"task_preconditions: unsupported kind {kind!r}")
    for effect in payload.get("task_effects", []) or []:
        kind = effect.get("kind") if isinstance(effect, dict) else None
        if kind not in SUPPORTED_EFFECT_KINDS:
            errors.append(f"task_effects: unsupported kind {kind!r}")
    return errors


def _has_exclusive_goal_assignment(
    goal_conditions: list[dict[str, Any]],
) -> bool:
    """Check whether exclusive one-of goals admit a one-to-one location assignment."""

    location_pools: list[tuple[str, tuple[str, ...]]] = []
    for condition in goal_conditions:
        if condition.get("kind") != "object_at_location_one_of":
            continue
        if not condition.get("exclusive", False):
            continue
        object_id = condition.get("object_id")
        locations = condition.get("locations")
        if not isinstance(object_id, str) or not isinstance(locations, list):
            continue
        normalized_locations = tuple(
            location
            for location in locations
            if isinstance(location, str)
        )
        if not normalized_locations:
            continue
        location_pools.append((object_id, normalized_locations))

    if not location_pools:
        return True

    pools_by_object = {
        object_id: tuple(dict.fromkeys(locations))
        for object_id, locations in location_pools
    }
    matched_object_by_location: dict[str, str] = {}

    def _try_match(object_id: str, seen_locations: set[str]) -> bool:
        for location in pools_by_object[object_id]:
            if location in seen_locations:
                continue
            seen_locations.add(location)
            current_owner = matched_object_by_location.get(location)
            if current_owner is None or _try_match(current_owner, seen_locations):
                matched_object_by_location[location] = object_id
                return True
        return False

    for object_id, _locations in sorted(location_pools, key=lambda item: len(item[1])):
        if not _try_match(object_id, set()):
            return False
    return True


def _collect_goal_consistency_errors(payload: dict[str, Any]) -> list[str]:
    """Return semantic goal errors that are impossible regardless of trajectory."""

    errors: list[str] = []
    goal_conditions = payload.get("goal_conditions", []) or []
    normalized_goal_conditions = [
        dict(condition) for condition in goal_conditions if isinstance(condition, dict)
    ]
    fixture_control_goal_states: dict[tuple[str, str], set[str]] = {}
    visited_machine_keys_by_control: dict[tuple[str, str], set[str]] = {}
    for index, goal in enumerate(normalized_goal_conditions):
        if goal.get("kind") == "object_count_at_location":
            object_ids = goal.get("object_ids")
            if not isinstance(object_ids, list) or not object_ids:
                errors.append(
                    f"goal_conditions[{index}]: object_count_at_location.object_ids must be a non-empty list"
                )
                continue
            normalized_object_ids = [
                object_id for object_id in object_ids if isinstance(object_id, str)
            ]
            count = goal.get("count")
            if not isinstance(count, int) or count < 0:
                errors.append(
                    f"goal_conditions[{index}]: object_count_at_location.count must be a non-negative integer"
                )
            elif count > len(normalized_object_ids):
                errors.append(
                    f"goal_conditions[{index}]: object_count_at_location.count={count} exceeds number of object_ids={len(normalized_object_ids)}"
                )
        if goal.get("kind") == "fixture_part_state":
            state = goal.get("state")
            if state not in _CANONICAL_PART_STATES:
                canonical_state = _LEGACY_PART_STATE_ALIASES.get(state)
                if canonical_state is not None:
                    errors.append(
                        f"goal_conditions[{index}]: fixture_part_state.state {state!r} must use canonical state {canonical_state!r}"
                    )
                else:
                    errors.append(
                        f"goal_conditions[{index}]: fixture_part_state.state {state!r} must be one of {sorted(_CANONICAL_PART_STATES)}"
                    )
        if goal.get("kind") == "fixture_control_state":
            fixture_id = goal.get("fixture_id")
            control_id = goal.get("control_id")
            state = goal.get("state")
            if (
                isinstance(fixture_id, str)
                and isinstance(control_id, str)
                and isinstance(state, str)
            ):
                fixture_control_goal_states.setdefault(
                    (fixture_id, control_id), set()
                ).add(state)
        if goal.get("kind") == "machine_flag_true":
            parsed_machine_path = _parse_visited_machine_path(goal.get("machine_path"))
            if parsed_machine_path is not None:
                fixture_id, machine_key = parsed_machine_path
                control_id, _visited_state = machine_key[:-8].rsplit("_", 1)
                visited_machine_keys_by_control.setdefault(
                    (fixture_id, control_id), set()
                ).add(machine_key)

    for fixture_id, fixture_state in (
        ((payload.get("initial_state") or {}).get("fixtures") or {}).items()
    ):
        if not isinstance(fixture_state, dict):
            continue
        fixture_parts = fixture_state.get("parts") or {}
        if not isinstance(fixture_parts, dict):
            continue
        for part_id, part_state in fixture_parts.items():
            if not isinstance(part_state, dict):
                continue
            state = part_state.get("state")
            if state in _CANONICAL_PART_STATES:
                continue
            canonical_state = _LEGACY_PART_STATE_ALIASES.get(state)
            if canonical_state is not None:
                errors.append(
                    f"initial_state.fixtures[{fixture_id!r}].parts[{part_id!r}].state {state!r} must use canonical state {canonical_state!r}"
                )
            else:
                errors.append(
                    f"initial_state.fixtures[{fixture_id!r}].parts[{part_id!r}].state {state!r} must be one of {sorted(_CANONICAL_PART_STATES)}"
                )

    for index, condition in enumerate(payload.get("task_preconditions", []) or []):
        if not isinstance(condition, dict):
            continue
        if condition.get("kind") not in {
            "fixture_part_state_required_for_pickup",
            "fixture_part_state_required_for_action",
        }:
            continue
        required_state = condition.get("required_state")
        if required_state in _CANONICAL_PART_STATES:
            continue
        canonical_state = _LEGACY_PART_STATE_ALIASES.get(required_state)
        if canonical_state is not None:
            errors.append(
                f"task_preconditions[{index}]: required_state {required_state!r} must use canonical state {canonical_state!r}"
            )
        else:
            errors.append(
                f"task_preconditions[{index}]: required_state {required_state!r} must be one of {sorted(_CANONICAL_PART_STATES)}"
            )

    for (fixture_id, control_id), states in fixture_control_goal_states.items():
        if len(states) > 1:
            errors.append(
                "goal_conditions: fixture_control_state requires "
                f"{fixture_id}.{control_id} to be in multiple states {sorted(states)}"
            )
        if (fixture_id, control_id) in visited_machine_keys_by_control:
            errors.append(
                "goal_conditions: mixed visited-state machine flags and "
                f"fixture_control_state for {fixture_id}.{control_id}; model all "
                "visited states consistently."
            )

    activation_flags_by_fixture: dict[str, list[list[str]]] = {}
    for effect in payload.get("task_effects", []) or []:
        if not isinstance(effect, dict):
            continue
        machine_path = effect.get("machine_path")
        if not isinstance(machine_path, list) or len(machine_path) != 2:
            continue
        if effect.get("value") is not True:
            continue
        fixture_id, flag_name = machine_path
        if not isinstance(fixture_id, str) or not isinstance(flag_name, str):
            continue
        if flag_name != "water_on":
            continue
        activation_flags_by_fixture.setdefault(fixture_id, []).append(machine_path)

    for index, effect in enumerate(payload.get("task_effects", []) or []):
        if not isinstance(effect, dict):
            continue
        parsed_machine_path = _parse_visited_machine_path(effect.get("machine_path"))
        if parsed_machine_path is None:
            continue
        fixture_id, machine_key = parsed_machine_path
        required_machine_values = effect.get("required_machine_values") or []
        for activation_path in activation_flags_by_fixture.get(fixture_id, []):
            if any(
                isinstance(requirement, dict)
                and requirement.get("machine_path") == activation_path
                and requirement.get("value") is True
                for requirement in required_machine_values
            ):
                continue
            errors.append(
                f"task_effects[{index}]: visited-state effect {machine_key!r} on "
                f"{fixture_id!r} must require machine_path {activation_path!r} == true"
            )
            break

    if _has_exclusive_goal_assignment(normalized_goal_conditions):
        return errors

    exclusive_objects = [
        str(condition.get("object_id"))
        for condition in normalized_goal_conditions
        if condition.get("kind") == "object_at_location_one_of"
        and condition.get("exclusive", False)
    ]
    errors.append(
        "goal_conditions: exclusive object_at_location_one_of goals do not admit "
        f"a one-to-one assignment for objects {sorted(exclusive_objects)}"
    )
    return errors


def _collect_grounding_errors(payload: dict[str, Any]) -> list[str]:
    """Return grounding errors that are statically inconsistent or circular."""

    errors: list[str] = []
    initial_objects = ((payload.get("initial_state") or {}).get("objects") or {})
    initial_fixtures = ((payload.get("initial_state") or {}).get("fixtures") or {})
    grounding_symbols = ((payload.get("grounding") or {}).get("symbols") or {})
    for symbol_name, symbol_spec in grounding_symbols.items():
        if not isinstance(symbol_spec, dict):
            continue
        if symbol_spec.get("entity_type") != "fixture":
            continue
        resolver = symbol_spec.get("resolver")
        if resolver not in {"source_fixture_for_object", "support_fixture_for_object"}:
            continue
        object_symbol = symbol_spec.get("object_symbol")
        if not isinstance(object_symbol, str):
            continue
        object_state = initial_objects.get(object_symbol)
        if not isinstance(object_state, dict):
            errors.append(
                f"grounding.symbols[{symbol_name!r}]: object_symbol {object_symbol!r} is not present in initial_state.objects"
            )
            continue
        object_location = object_state.get("location")
        fixture_support_sites = (
            initial_fixtures.get(symbol_name, {}).get("support_sites", {})
            if isinstance(initial_fixtures.get(symbol_name), dict)
            else {}
        )
        location_matches_fixture = object_location == symbol_name or (
            isinstance(fixture_support_sites, dict)
            and object_location in fixture_support_sites
        ) or (
            isinstance(fixture_support_sites, list)
            and object_location in fixture_support_sites
        )
        if resolver == "source_fixture_for_object" and not location_matches_fixture:
            errors.append(
                f"grounding.symbols[{symbol_name!r}]: source_fixture_for_object uses object_symbol {object_symbol!r}, but that object starts at {object_location!r}, not {symbol_name!r}"
            )
        if resolver == "support_fixture_for_object" and location_matches_fixture:
            errors.append(
                f"grounding.symbols[{symbol_name!r}]: support_fixture_for_object uses object_symbol {object_symbol!r} that already starts at {symbol_name!r}; ground the fixture more directly instead of through a resident object"
            )
    return errors


def _collect_known_ids(payload: dict[str, Any]) -> tuple[set[str], set[str], set[str]]:
    """Pull object/fixture/agent ids out of `initial_state`.

    Returns (object_ids, fixture_ids, agent_ids). The id sets are used by
    referential checks below; ids that come from `machine_state` are out
    of scope here because they are referenced via `machine_path` lists.
    """

    initial_state = payload.get("initial_state", {}) or {}
    object_ids = set((initial_state.get("objects") or {}).keys())
    fixture_ids = set((initial_state.get("fixtures") or {}).keys())
    agent_ids = set((initial_state.get("agents") or {}).keys())
    return object_ids, fixture_ids, agent_ids


def _check_referential_integrity(payload: dict[str, Any]) -> list[str]:
    """Check that ids referenced by goals/preconditions/effects/trajectory exist.

    Only strict checks are performed here. Locations, source_ids, part_ids,
    control_ids, and most trajectory `args` can legitimately reference
    runtime-resolved sub-entities (e.g. `coffee_machine_dispenser` derived
    from `machine_state.coffee_machine.dispenser_id`, or nested
    `fixtures.coffee_machine.controls.start_button`). Those are validated
    during the dry-run instead.

    We only flag:
      - `object_id` values in goals/preconditions that are not in
        `initial_state.objects` (these are always top-level ids).
      - `agent` values in trajectory steps that are not in the declared
        `initial_state.agents` map.
    """

    errors: list[str] = []
    object_ids, fixture_ids, agent_ids = _collect_known_ids(payload)

    for index, goal in enumerate(payload.get("goal_conditions", []) or []):
        if not isinstance(goal, dict):
            continue
        obj_id = goal.get("object_id")
        if obj_id is not None and obj_id not in object_ids:
            errors.append(
                f"goal_conditions[{index}]: object_id {obj_id!r} not in initial_state.objects"
            )
        if goal.get("kind") == "object_count_at_location":
            for object_id in goal.get("object_ids") or []:
                if object_id not in object_ids:
                    errors.append(
                        f"goal_conditions[{index}]: object_ids entry {object_id!r} not in initial_state.objects"
                    )
        if goal.get("kind") == "fixture_part_state":
            fixture_id = goal.get("fixture_id")
            if fixture_id not in fixture_ids:
                errors.append(
                    f"goal_conditions[{index}]: fixture_id {fixture_id!r} not in initial_state.fixtures"
                )
                continue
            fixture_parts = (
                (
                    (payload.get("initial_state") or {})
                    .get("fixtures", {})
                    .get(fixture_id, {})
                    .get("parts", {})
                )
                if isinstance(fixture_id, str)
                else {}
            )
            part_id = goal.get("part_id")
            if part_id not in fixture_parts:
                errors.append(
                    f"goal_conditions[{index}]: part_id {part_id!r} not in initial_state.fixtures[{fixture_id!r}].parts"
                )
        if goal.get("kind") == "fixture_control_state":
            fixture_id = goal.get("fixture_id")
            if fixture_id not in fixture_ids:
                errors.append(
                    f"goal_conditions[{index}]: fixture_id {fixture_id!r} not in initial_state.fixtures"
                )
                continue
            fixture_controls = (
                (
                    (payload.get("initial_state") or {})
                    .get("fixtures", {})
                    .get(fixture_id, {})
                    .get("controls", {})
                )
                if isinstance(fixture_id, str)
                else {}
            )
            control_id = goal.get("control_id")
            if control_id not in fixture_controls:
                errors.append(
                    f"goal_conditions[{index}]: control_id {control_id!r} not in initial_state.fixtures[{fixture_id!r}].controls"
                )

    for index, cond in enumerate(payload.get("task_preconditions", []) or []):
        if not isinstance(cond, dict):
            continue
        obj_id = cond.get("object_id")
        if obj_id is not None and obj_id not in object_ids:
            errors.append(
                f"task_preconditions[{index}]: object_id {obj_id!r} not in initial_state.objects"
            )
        if cond.get("kind") in {
            "fixture_part_state_required_for_pickup",
            "fixture_part_state_required_for_action",
        }:
            fixture_id = cond.get("fixture_id")
            if fixture_id not in fixture_ids:
                errors.append(
                    f"task_preconditions[{index}]: fixture_id {fixture_id!r} not in initial_state.fixtures"
                )
                continue
            fixture_parts = (
                (
                    (payload.get("initial_state") or {})
                    .get("fixtures", {})
                    .get(fixture_id, {})
                    .get("parts", {})
                )
                if isinstance(fixture_id, str)
                else {}
            )
            part_id = cond.get("part_id")
            if part_id not in fixture_parts:
                errors.append(
                    f"task_preconditions[{index}]: part_id {part_id!r} not in initial_state.fixtures[{fixture_id!r}].parts"
                )

    trajectory = payload.get("example_trajectory") or {}
    steps = trajectory.get("steps") or []
    for index, effect in enumerate(payload.get("task_effects", []) or []):
        if not isinstance(effect, dict):
            continue
        for requirement in effect.get("required_object_locations") or []:
            if not isinstance(requirement, dict):
                continue
            object_id = requirement.get("object_id")
            if object_id not in object_ids:
                errors.append(
                    f"task_effects[{index}]: required_object_locations object_id {object_id!r} not in initial_state.objects"
                )
        for requirement in effect.get("required_fixture_controls") or []:
            if not isinstance(requirement, dict):
                continue
            fixture_id = requirement.get("fixture_id")
            if fixture_id not in fixture_ids:
                errors.append(
                    f"task_effects[{index}]: required_fixture_controls fixture_id {fixture_id!r} not in initial_state.fixtures"
                )
                continue
            fixture_controls = (
                (
                    (payload.get("initial_state") or {})
                    .get("fixtures", {})
                    .get(fixture_id, {})
                    .get("controls", {})
                )
                if isinstance(fixture_id, str)
                else {}
            )
            control_id = requirement.get("control_id")
            if control_id not in fixture_controls:
                errors.append(
                    f"task_effects[{index}]: required_fixture_controls control_id {control_id!r} not in initial_state.fixtures[{fixture_id!r}].controls"
                )
    for step in steps:
        if not isinstance(step, dict):
            continue
        agent = step.get("agent")
        if agent is not None and agent not in agent_ids:
            errors.append(
                f"example_trajectory step {step.get('step')}: agent {agent!r} not in initial_state.agents"
            )
    return errors


def _dry_run_example_trajectory(task_spec: TaskSpec) -> str | None:
    """Replay `example_trajectory` through SpecDrivenTaskValidator.

    Mirrors the production path in `build_task_definition_from_spec` so
    runtime-injected tools (notably `open_hinged_part` for fixtures with
    hinged_part entries) are available to the validator. Constructing
    `SpecDrivenTaskValidator(task_spec)` directly would miss those tools
    and falsely reject valid trajectories.

    Returns None on success or a short error string on failure.
    """

    try:
        task_definition = build_task_definition_from_spec(task_spec)
    except Exception as exc:  # noqa: BLE001 — surface build errors as dry-run failures
        return f"build_task_definition_from_spec failed: {type(exc).__name__}: {exc}"

    validator = task_definition.validator_factory(None)
    try:
        validator.validate(dict(task_spec.example_trajectory))
    except TrajectoryValidationError as exc:
        details = getattr(exc, "details", None)
        details_text = (
            f" details={json.dumps(details, default=str)}"
            if isinstance(details, dict)
            else ""
        )
        return f"{type(exc).__name__}: {exc}{details_text}"
    except Exception as exc:  # noqa: BLE001 — surface unexpected validator errors
        return f"{type(exc).__name__}: {exc}"
    return None


def validate_one_spec(
    spec_path: Path,
    *,
    task_name: str | None = None,
) -> SpecValidationResult:
    """Run all four static checks on one TaskSpec JSON file."""

    try:
        with spec_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        return SpecValidationResult(
            task_name=task_name or spec_path.stem,
            spec_path=str(spec_path),
            passed=False,
            schema_error=f"failed to read spec file: {exc}",
        )

    resolved_task_name = task_name or str(payload.get("composite_task") or spec_path.stem)
    result = SpecValidationResult(
        task_name=resolved_task_name,
        spec_path=str(spec_path),
        passed=False,
    )

    # 1. Schema check.
    try:
        task_spec = TaskSpec.from_dict(payload)
    except (ValueError, KeyError, TypeError) as exc:
        result.schema_error = f"{type(exc).__name__}: {exc}"
        return result

    # 2. Supported kinds.
    result.unsupported_kinds = _collect_unsupported_kinds(payload)

    # 2.5. Goal consistency.
    result.goal_consistency_errors = _collect_goal_consistency_errors(payload)

    # 2.75. Grounding semantics.
    result.grounding_errors = _collect_grounding_errors(payload)

    # 2.9. Simulator alignment.
    try:
        result.simulation_errors = collect_simulation_alignment_errors(
            payload,
            task_name=resolved_task_name,
        )
    except Exception as exc:  # noqa: BLE001 — report simulator introspection failures as validation errors
        result.simulation_errors = [
            f"simulation alignment check failed: {type(exc).__name__}: {exc}"
        ]

    # 3. Referential integrity.
    result.referential_errors = _check_referential_integrity(payload)

    # Skip the dry-run when earlier checks already failed — the validator
    # will likely raise a less informative error.
    if (
        result.unsupported_kinds
        or result.goal_consistency_errors
        or result.grounding_errors
        or result.simulation_errors
        or result.referential_errors
    ):
        return result

    # 4. Example-trajectory dry-run.
    result.dry_run_error = _dry_run_example_trajectory(task_spec)
    result.passed = result.dry_run_error is None
    return result


def _summarize_validation(results: list[SpecValidationResult]) -> dict[str, Any]:
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    return {"total": len(results), "passed": passed, "failed": failed}


def run_phase2(
    *,
    output_dir: Path,
    spec_paths: list[Path] | None = None,
) -> list[SpecValidationResult]:
    """Validate every spec in `phase1/specs/` (or an explicit list)."""

    if spec_paths is None:
        specs_dir = output_dir / "phase1" / "specs"
        if not specs_dir.exists():
            raise FileNotFoundError(
                f"Phase 1 specs directory not found: {specs_dir}. "
                f"Run --phase 1 first."
            )
        spec_paths = sorted(specs_dir.glob("*.json"))

    results = [validate_one_spec(path) for path in spec_paths]

    phase_dir = output_dir / "phase2"
    phase_dir.mkdir(parents=True, exist_ok=True)
    with (phase_dir / "validation_results.json").open("w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in results], f, indent=2)
    with (phase_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(_summarize_validation(results), f, indent=2)

    return results
