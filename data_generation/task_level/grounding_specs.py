"""Build scene-agnostic grounding specs and resolve them for one scene."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

GROUNDING_MAP_VERSION = 2

_LEGACY_SYMBOL_ALIASES_BY_TASK = {
    "HotDogSetup": {
        "hotdog_bun_1": "bun",
        "sausage_1": "sausage",
        "condiment_1": "condiment",
        "plate_1": "serving_plate",
        "counter_1": "bun_source_fixture",
        "fridge_1": "sausage_source_fixture",
        "cabinet_1": "condiment_source_fixture",
        "dining_table_1": "serving_surface",
    },
    "PrepareCoffee": {
        "mug_1": "mug",
        "cabinet_1": "mug_source_fixture",
        "counter_1": "staging_surface",
        "coffee_machine_1": "coffee_machine",
    },
    "PrepareSandwichStation": {
        "ingredient_bowl_1": "ingredient_bowl",
        "baguette_1": "baguette",
        "tomato_slice_1": "tomato_slice",
        "pickle_slice_1": "pickle_slice",
        "turkey_slice_1": "turkey_slice",
        "fridge_1": "ingredient_source_fixture",
        "counter_1": "staging_surface",
        "toaster_oven_1": "toaster_oven",
    },
}

_CABINET_FIXTURE_TYPES = (
    "cabinet",
    "cabinet_single_door",
    "cabinet_double_door",
    "cabinet_with_door",
)
_COUNTER_FIXTURE_TYPES = (
    "counter",
    "counter_non_corner",
    "counter_non_dining",
)
_DINING_SURFACE_FIXTURE_TYPES = (
    "dining_counter",
    "island",
    "counter_non_dining",
)
_PLACEABLE_SURFACE_FIXTURE_TYPES = (
    "counter",
    "counter_non_corner",
    "counter_non_dining",
    "dining_counter",
    "island",
)


def build_grounding_map_for_task(
    composite_task: str,
    initial_state: dict[str, Any],
) -> dict[str, Any]:
    """Builds one scene-agnostic grounding map for a supported task."""

    initial_state = _normalize_initial_state_symbols(composite_task, initial_state)

    if composite_task == "HotDogSetup":
        return _build_hot_dog_setup_grounding_map(initial_state)
    if composite_task == "PrepareCoffee":
        return _build_prepare_coffee_grounding_map(initial_state)
    if composite_task == "PrepareSandwichStation":
        return _build_prepare_sandwich_station_grounding_map(initial_state)
    raise ValueError(f"Unsupported grounding-map task {composite_task!r}.")


def build_resolved_grounding_payload(
    trajectory: dict[str, Any],
    scene: dict[str, Any],
    *,
    scene_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Builds one persisted grounding payload for a trajectory and scene."""

    grounding_map = trajectory.get("grounding_map")
    if not isinstance(grounding_map, dict):
        grounding_map = build_grounding_map_for_task(
            trajectory["composite_task"],
            trajectory["initial_state"],
        )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "trajectory_id": trajectory.get("trajectory_id"),
        "composite_task": trajectory.get("composite_task"),
        "scene_config": deepcopy(scene_config) if scene_config is not None else None,
        "grounding_map": deepcopy(grounding_map),
        "scene_summary": _build_scene_summary(scene),
        "resolved_grounding": resolve_grounding_map(grounding_map, scene),
    }


def resolve_grounding_map(
    grounding_map: dict[str, Any],
    scene: dict[str, Any],
) -> dict[str, Any]:
    """Resolves a scene-agnostic grounding map into concrete scene IDs."""

    symbols = grounding_map.get("symbols", {})
    if not isinstance(symbols, dict):
        raise ValueError("grounding_map.symbols must be a dict.")

    resolved_symbols: dict[str, dict[str, Any]] = {}
    unresolved_symbols: list[dict[str, Any]] = []

    object_symbols = {
        symbol: spec
        for symbol, spec in symbols.items()
        if isinstance(spec, dict) and spec.get("entity_type") == "object"
    }
    fixture_symbols = {
        symbol: spec
        for symbol, spec in symbols.items()
        if isinstance(spec, dict) and spec.get("entity_type") == "fixture"
    }

    for symbol, spec in object_symbols.items():
        resolution = _resolve_symbol(symbol, spec, scene, resolved_symbols)
        if resolution.get("resolved_id") is None:
            unresolved_symbols.append(resolution)
            continue
        resolved_symbols[symbol] = resolution

    pending_fixture_symbols = dict(fixture_symbols)
    while pending_fixture_symbols:
        resolved_in_pass = False
        for symbol in list(pending_fixture_symbols):
            resolution = _resolve_symbol(
                symbol,
                pending_fixture_symbols[symbol],
                scene,
                resolved_symbols,
            )
            if resolution.get("resolved_id") is None:
                dependency_symbol = resolution.get("dependency_symbol")
                if (
                    isinstance(dependency_symbol, str)
                    and dependency_symbol not in resolved_symbols
                    and dependency_symbol in pending_fixture_symbols
                ):
                    continue
                unresolved_symbols.append(resolution)
                pending_fixture_symbols.pop(symbol)
                continue
            resolved_symbols[symbol] = resolution
            pending_fixture_symbols.pop(symbol)
            resolved_in_pass = True

        if resolved_in_pass:
            continue

        for symbol, spec in pending_fixture_symbols.items():
            unresolved_symbols.append(
                _resolve_symbol(symbol, spec, scene, resolved_symbols)
            )
        break

    return {
        "version": grounding_map.get("version", GROUNDING_MAP_VERSION),
        "map_kind": grounding_map.get("map_kind", "scene_agnostic"),
        "composite_task": grounding_map.get("composite_task"),
        "is_complete": len(unresolved_symbols) == 0,
        "resolved_symbols": resolved_symbols,
        "unresolved_symbols": unresolved_symbols,
    }


def _build_scene_summary(scene: dict[str, Any]) -> dict[str, Any]:
    """Builds a compact serializable scene summary for grounding outputs."""

    fixtures = scene.get("fixtures", {})
    objects = scene.get("objects", {})
    return {
        "fixtures": {
            fixture_id: {
                "fixture_type": fixture_info.get("fixture_type"),
                "can_place_objects": bool(fixture_info.get("can_place_objects", False)),
            }
            for fixture_id, fixture_info in fixtures.items()
            if isinstance(fixture_info, dict)
        },
        "objects": {
            object_id: {
                "object_type": object_info.get("object_type"),
                "location": object_info.get("location"),
            }
            for object_id, object_info in objects.items()
            if isinstance(object_info, dict)
        },
    }


def _build_hot_dog_setup_grounding_map(
    initial_state: dict[str, Any],
) -> dict[str, Any]:
    """Builds the reusable HotDogSetup grounding map."""

    return _grounding_map(
        "HotDogSetup",
        symbols={
            "bun": _object_entry(
                initial_state,
                "bun",
                preferred_fixture_types=_COUNTER_FIXTURE_TYPES,
                role="bun",
            ),
            "sausage": _object_entry(
                initial_state,
                "sausage",
                preferred_fixture_types=("fridge",),
                role="sausage",
            ),
            "condiment": _object_entry(
                initial_state,
                "condiment",
                preferred_fixture_types=_CABINET_FIXTURE_TYPES,
                role="condiment",
            ),
            "serving_plate": _object_entry(
                initial_state,
                "serving_plate",
                preferred_fixture_types=_DINING_SURFACE_FIXTURE_TYPES,
                role="serving_plate",
            ),
            "bun_source_fixture": _fixture_entry(
                initial_state,
                "bun_source_fixture",
                resolver="source_fixture_for_object",
                object_symbol="bun",
                preferred_fixture_types=_COUNTER_FIXTURE_TYPES,
                role="bun_source_fixture",
            ),
            "sausage_source_fixture": _fixture_entry(
                initial_state,
                "sausage_source_fixture",
                resolver="source_fixture_for_object",
                object_symbol="sausage",
                preferred_fixture_types=("fridge",),
                role="sausage_source_fixture",
            ),
            "condiment_source_fixture": _fixture_entry(
                initial_state,
                "condiment_source_fixture",
                resolver="source_fixture_for_object",
                object_symbol="condiment",
                preferred_fixture_types=_CABINET_FIXTURE_TYPES,
                role="condiment_source_fixture",
            ),
            "serving_surface": _fixture_entry(
                initial_state,
                "serving_surface",
                resolver="support_fixture_for_object",
                object_symbol="serving_plate",
                preferred_fixture_types=_DINING_SURFACE_FIXTURE_TYPES,
                role="serving_surface",
            ),
        },
    )


def _build_prepare_coffee_grounding_map(
    initial_state: dict[str, Any],
) -> dict[str, Any]:
    """Builds the reusable PrepareCoffee grounding map."""

    return _grounding_map(
        "PrepareCoffee",
        symbols={
            "mug": _object_entry(
                initial_state,
                "mug",
                preferred_fixture_types=_CABINET_FIXTURE_TYPES
                + _PLACEABLE_SURFACE_FIXTURE_TYPES,
                role="mug",
            ),
            "mug_source_fixture": _fixture_entry(
                initial_state,
                "mug_source_fixture",
                resolver="source_fixture_for_object",
                object_symbol="mug",
                preferred_fixture_types=_CABINET_FIXTURE_TYPES,
                role="mug_source_fixture",
            ),
            "coffee_machine": _fixture_entry(
                initial_state,
                "coffee_machine",
                resolver="unique_fixture_type",
                fixture_type="coffee_machine",
                role="coffee_machine",
            ),
            "staging_surface": _fixture_entry(
                initial_state,
                "staging_surface",
                resolver="nearest_placeable_surface_to_fixture",
                anchor_fixture_symbol="coffee_machine",
                preferred_fixture_types=_PLACEABLE_SURFACE_FIXTURE_TYPES,
                role="staging_surface",
            ),
        },
    )


def _build_prepare_sandwich_station_grounding_map(
    initial_state: dict[str, Any],
) -> dict[str, Any]:
    """Builds the reusable PrepareSandwichStation grounding map."""

    return _grounding_map(
        "PrepareSandwichStation",
        symbols={
            "ingredient_bowl": _object_entry(
                initial_state,
                "ingredient_bowl",
                preferred_fixture_types=("fridge",) + _PLACEABLE_SURFACE_FIXTURE_TYPES,
                role="ingredient_bowl",
            ),
            "baguette": _object_entry(
                initial_state,
                "baguette",
                preferred_fixture_types=("fridge",) + _PLACEABLE_SURFACE_FIXTURE_TYPES,
                role="baguette",
            ),
            "ingredient_source_fixture": _fixture_entry(
                initial_state,
                "ingredient_source_fixture",
                resolver="source_fixture_for_object",
                object_symbol="ingredient_bowl",
                preferred_fixture_types=("fridge",),
                role="ingredient_source_fixture",
            ),
            "toaster_oven": _fixture_entry(
                initial_state,
                "toaster_oven",
                resolver="unique_fixture_type",
                fixture_type="toaster_oven",
                role="toaster_oven",
            ),
            "staging_surface": _fixture_entry(
                initial_state,
                "staging_surface",
                resolver="nearest_placeable_surface_to_fixture",
                anchor_fixture_symbol="toaster_oven",
                preferred_fixture_types=_PLACEABLE_SURFACE_FIXTURE_TYPES,
                role="staging_surface",
            ),
        },
    )


def _normalize_initial_state_symbols(
    composite_task: str,
    initial_state: dict[str, Any],
) -> dict[str, Any]:
    """Normalizes legacy task symbols so grounding maps use canonical role names."""

    legacy_aliases = _LEGACY_SYMBOL_ALIASES_BY_TASK.get(composite_task, {})
    if not legacy_aliases:
        return deepcopy(initial_state)
    return _rename_symbol_aliases(initial_state, legacy_aliases)


def _rename_symbol_aliases(
    value: Any,
    aliases: dict[str, str],
) -> Any:
    """Recursively rewrites legacy symbolic IDs inside one structured payload."""

    if isinstance(value, dict):
        return {
            aliases.get(key, key): _rename_symbol_aliases(item, aliases)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_rename_symbol_aliases(item, aliases) for item in value]
    if isinstance(value, str):
        return aliases.get(value, value)
    return deepcopy(value)


def _grounding_map(
    composite_task: str,
    *,
    symbols: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Builds the top-level scene-agnostic grounding-map payload."""

    return {
        "version": GROUNDING_MAP_VERSION,
        "map_kind": "scene_agnostic",
        "composite_task": composite_task,
        "symbols": symbols,
    }


def _object_entry(
    initial_state: dict[str, Any],
    symbol: str,
    *,
    preferred_fixture_types: tuple[str, ...] = (),
    role: str | None = None,
) -> dict[str, Any]:
    """Builds one reusable object-grounding entry."""

    object_state = initial_state["objects"][symbol]
    return {
        "entity_type": "object",
        "resolver": "object_by_type",
        "role": role or symbol,
        "object_type": object_state["object_type"],
        "symbolic_location": object_state.get("location"),
        "preferred_fixture_types": list(preferred_fixture_types),
    }


def _fixture_entry(
    initial_state: dict[str, Any],
    symbol: str,
    *,
    resolver: str,
    role: str | None = None,
    **resolver_args: Any,
) -> dict[str, Any]:
    """Builds one reusable fixture-grounding entry."""

    fixture_state = initial_state["fixtures"][symbol]
    entry = {
        "entity_type": "fixture",
        "resolver": resolver,
        "role": role or symbol,
        "fixture_type": fixture_state["fixture_type"],
    }
    entry.update(deepcopy(resolver_args))
    return entry


def _resolve_symbol(
    symbol: str,
    spec: dict[str, Any],
    scene: dict[str, Any],
    resolved_symbols: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Resolves one symbol into a concrete scene identifier."""

    resolver = spec.get("resolver")
    if resolver == "object_by_type":
        return _resolve_object_by_type(symbol, spec, scene)
    if resolver == "source_fixture_for_object":
        return _resolve_fixture_for_object(
            symbol,
            spec,
            scene,
            resolved_symbols,
            reason_prefix="Using the source fixture for object symbol",
        )
    if resolver == "support_fixture_for_object":
        return _resolve_fixture_for_object(
            symbol,
            spec,
            scene,
            resolved_symbols,
            reason_prefix="Using the supporting fixture for object symbol",
        )
    if resolver == "unique_fixture_type":
        return _resolve_unique_fixture_type(symbol, spec, scene)
    if resolver == "nearest_placeable_surface_to_fixture":
        return _resolve_nearest_placeable_surface(symbol, spec, scene, resolved_symbols)
    return _unresolved_symbol(
        symbol,
        spec,
        reason=f"Unsupported resolver {resolver!r}.",
    )


def _resolve_object_by_type(
    symbol: str,
    spec: dict[str, Any],
    scene: dict[str, Any],
) -> dict[str, Any]:
    """Resolves one symbolic object to a concrete scene object."""

    object_type = spec.get("object_type")
    if not isinstance(object_type, str) or not object_type:
        return _unresolved_symbol(symbol, spec, reason="Missing object_type.")

    candidates = _object_candidates_for_type(
        scene,
        object_type,
        preferred_fixture_types=spec.get("preferred_fixture_types"),
    )
    if len(candidates) == 1:
        reason = f"Matched unique object with type {object_type!r}."
        if spec.get("preferred_fixture_types"):
            reason = (
                f"Matched unique object with type {object_type!r} after preferred "
                "fixture-type filtering."
            )
        confidence = 0.95 if spec.get("preferred_fixture_types") else 1.0
        return _resolved_symbol(
            symbol,
            spec,
            candidates[0],
            confidence=confidence,
            candidates=candidates,
            reason=reason,
        )
    if not candidates:
        return _unresolved_symbol(
            symbol,
            spec,
            reason=f"No scene object matched type {object_type!r}.",
            candidates=[],
        )
    return _unresolved_symbol(
        symbol,
        spec,
        reason=f"Multiple scene objects matched type {object_type!r}.",
        candidates=candidates,
    )


def _resolve_fixture_for_object(
    symbol: str,
    spec: dict[str, Any],
    scene: dict[str, Any],
    resolved_symbols: dict[str, dict[str, Any]],
    *,
    reason_prefix: str,
) -> dict[str, Any]:
    """Resolves one fixture symbol from a previously resolved object symbol."""

    object_symbol = spec.get("object_symbol")
    if not isinstance(object_symbol, str) or not object_symbol:
        return _unresolved_symbol(symbol, spec, reason="Missing object_symbol.")
    object_resolution = resolved_symbols.get(object_symbol)
    if object_resolution is None:
        return _unresolved_symbol(
            symbol,
            spec,
            reason=f"Object symbol {object_symbol!r} is not resolved yet.",
            dependency_symbol=object_symbol,
        )

    object_id = object_resolution["resolved_id"]
    object_info = scene.get("objects", {}).get(object_id)
    if not isinstance(object_info, dict):
        return _unresolved_symbol(
            symbol,
            spec,
            reason=f"Resolved object {object_id!r} is missing from the scene.",
        )

    fixture_id = object_info.get("location")
    if not isinstance(fixture_id, str):
        return _unresolved_symbol(
            symbol,
            spec,
            reason=f"Resolved object {object_id!r} has no fixture location.",
        )

    fixtures = scene.get("fixtures", {})
    fixture_info = fixtures.get(fixture_id)
    if not isinstance(fixture_info, dict):
        return _unresolved_symbol(
            symbol,
            spec,
            reason=f"Resolved object location {fixture_id!r} is not a known fixture.",
        )

    preferred_fixture_types = tuple(spec.get("preferred_fixture_types") or ())
    fixture_type = fixture_info.get("fixture_type")
    if preferred_fixture_types and fixture_type not in preferred_fixture_types:
        return _unresolved_symbol(
            symbol,
            spec,
            reason=(
                f"Fixture {fixture_id!r} has type {fixture_type!r}, which is outside "
                f"preferred types {sorted(preferred_fixture_types)!r}."
            ),
        )

    return _resolved_symbol(
        symbol,
        spec,
        fixture_id,
        confidence=1.0,
        candidates=[fixture_id],
        reason=f"{reason_prefix} {object_symbol!r}.",
    )


def _resolve_unique_fixture_type(
    symbol: str,
    spec: dict[str, Any],
    scene: dict[str, Any],
) -> dict[str, Any]:
    """Resolves one fixture symbol by unique fixture type."""

    fixture_type = spec.get("fixture_type")
    if not isinstance(fixture_type, str) or not fixture_type:
        return _unresolved_symbol(symbol, spec, reason="Missing fixture_type.")

    candidates = _fixture_candidates_for_types(scene, (fixture_type,))
    if len(candidates) == 1:
        return _resolved_symbol(
            symbol,
            spec,
            candidates[0],
            confidence=1.0,
            candidates=candidates,
            reason=f"Matched unique fixture with type {fixture_type!r}.",
        )
    if not candidates:
        return _unresolved_symbol(
            symbol,
            spec,
            reason=f"No scene fixture matched type {fixture_type!r}.",
            candidates=[],
        )
    return _unresolved_symbol(
        symbol,
        spec,
        reason=f"Multiple scene fixtures matched type {fixture_type!r}.",
        candidates=candidates,
    )


def _resolve_nearest_placeable_surface(
    symbol: str,
    spec: dict[str, Any],
    scene: dict[str, Any],
    resolved_symbols: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Resolves one fixture symbol to the nearest placeable surface."""

    anchor_fixture_symbol = spec.get("anchor_fixture_symbol")
    if not isinstance(anchor_fixture_symbol, str) or not anchor_fixture_symbol:
        return _unresolved_symbol(
            symbol,
            spec,
            reason="Missing anchor_fixture_symbol.",
        )

    anchor_resolution = resolved_symbols.get(anchor_fixture_symbol)
    if anchor_resolution is None:
        return _unresolved_symbol(
            symbol,
            spec,
            reason=f"Anchor fixture symbol {anchor_fixture_symbol!r} is not resolved yet.",
            dependency_symbol=anchor_fixture_symbol,
        )

    fixtures = scene.get("fixtures", {})
    anchor_fixture_id = anchor_resolution["resolved_id"]
    anchor_fixture = fixtures.get(anchor_fixture_id)
    if not isinstance(anchor_fixture, dict):
        return _unresolved_symbol(
            symbol,
            spec,
            reason=f"Resolved anchor fixture {anchor_fixture_id!r} is missing.",
        )

    preferred_fixture_types = tuple(spec.get("preferred_fixture_types") or ())
    candidates = _fixture_candidates_for_types(
        scene,
        preferred_fixture_types,
        require_placeable=True,
    )
    if not candidates:
        return _unresolved_symbol(
            symbol,
            spec,
            reason="No placeable candidate fixtures matched the preferred types.",
            candidates=[],
        )

    if bool(anchor_fixture.get("can_place_objects", False)) and (
        not preferred_fixture_types
        or anchor_fixture.get("fixture_type") in preferred_fixture_types
    ):
        return _resolved_symbol(
            symbol,
            spec,
            anchor_fixture_id,
            confidence=1.0,
            candidates=[anchor_fixture_id],
            reason=f"Anchor fixture {anchor_fixture_symbol!r} is already placeable.",
        )

    anchor_xy = _xy_position(anchor_fixture.get("position"))
    if anchor_xy is None:
        return _unresolved_symbol(
            symbol,
            spec,
            reason=f"Anchor fixture {anchor_fixture_id!r} has no valid position.",
            candidates=candidates,
        )

    resolved_id = min(
        candidates,
        key=lambda fixture_id: _distance_to_fixture(scene, fixture_id, anchor_xy),
    )
    return _resolved_symbol(
        symbol,
        spec,
        resolved_id,
        confidence=0.9,
        candidates=candidates,
        reason=f"Selected nearest placeable surface to {anchor_fixture_symbol!r}.",
    )


def _object_candidates_for_type(
    scene: dict[str, Any],
    object_type: str,
    *,
    preferred_fixture_types: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    """Returns scene object candidates that match one symbolic object entry."""

    preferred_fixture_types = tuple(preferred_fixture_types or ())
    fixtures = scene.get("fixtures", {})
    candidates: list[str] = []
    for object_id, object_info in scene.get("objects", {}).items():
        if not isinstance(object_info, dict):
            continue
        if object_info.get("object_type") != object_type:
            continue
        if preferred_fixture_types:
            location_id = object_info.get("location")
            fixture_type = fixtures.get(location_id, {}).get("fixture_type")
            if fixture_type not in preferred_fixture_types:
                continue
        candidates.append(object_id)
    return sorted(candidates)


def _fixture_candidates_for_types(
    scene: dict[str, Any],
    fixture_types: tuple[str, ...],
    *,
    require_placeable: bool = False,
) -> list[str]:
    """Returns scene fixture candidates that match one symbolic fixture entry."""

    candidates: list[str] = []
    for fixture_id, fixture_info in scene.get("fixtures", {}).items():
        if not isinstance(fixture_info, dict):
            continue
        if fixture_types and fixture_info.get("fixture_type") not in fixture_types:
            continue
        if require_placeable and not bool(fixture_info.get("can_place_objects", False)):
            continue
        candidates.append(fixture_id)
    return sorted(candidates)


def _distance_to_fixture(
    scene: dict[str, Any],
    fixture_id: str,
    anchor_xy: tuple[float, float],
) -> float:
    """Returns the 2-D distance between one fixture and an anchor point."""

    fixture_info = scene.get("fixtures", {}).get(fixture_id, {})
    fixture_xy = _xy_position(fixture_info.get("position"))
    if fixture_xy is None:
        return float("inf")
    dx = fixture_xy[0] - anchor_xy[0]
    dy = fixture_xy[1] - anchor_xy[1]
    return (dx * dx + dy * dy) ** 0.5


def _xy_position(position: Any) -> tuple[float, float] | None:
    """Returns the 2-D XY position tuple for one fixture position payload."""

    if not isinstance(position, (list, tuple)) or len(position) < 2:
        return None
    try:
        return (float(position[0]), float(position[1]))
    except (TypeError, ValueError):
        return None


def _resolved_symbol(
    symbol: str,
    spec: dict[str, Any],
    resolved_id: str,
    *,
    confidence: float,
    candidates: list[str],
    reason: str,
) -> dict[str, Any]:
    """Builds one successful symbol-resolution payload."""

    return {
        "symbol": symbol,
        "entity_type": spec.get("entity_type"),
        "resolver": spec.get("resolver"),
        "role": spec.get("role"),
        "resolved_id": resolved_id,
        "confidence": confidence,
        "candidates": candidates,
        "reason": reason,
    }


def _unresolved_symbol(
    symbol: str,
    spec: dict[str, Any],
    *,
    reason: str,
    candidates: list[str] | None = None,
    dependency_symbol: str | None = None,
) -> dict[str, Any]:
    """Builds one unresolved symbol-resolution payload."""

    payload = {
        "symbol": symbol,
        "entity_type": spec.get("entity_type"),
        "resolver": spec.get("resolver"),
        "role": spec.get("role"),
        "resolved_id": None,
        "reason": reason,
    }
    if candidates is not None:
        payload["candidates"] = candidates
    if dependency_symbol is not None:
        payload["dependency_symbol"] = dependency_symbol
    return payload
