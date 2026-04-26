"""Normalize TaskSpec simulator-facing IDs against live RoboCasa metadata."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from importlib import import_module
import re
from typing import Any

from robocasa.models.scenes.scene_registry import unpack_layout_ids, unpack_style_ids
from robocasa.utils.env_utils import KITCHEN_SCENES_5X1, KITCHEN_SCENES_5X5
from robocasa.utils.sim_tool_executor import SimToolExecutor

_PREFERRED_LAYOUT_STYLE_IDS = tuple(
    dict.fromkeys(
        (
            *KITCHEN_SCENES_5X5,
            *KITCHEN_SCENES_5X1,
            (42, 34),
            (56, 34),
        )
    )
)
_SIM_NORMALIZATION_SEEDS = (42, 99, 7)
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_TOKEN_ALIASES = {
    "back": ("rear",),
    "faucet": ("handle",),
    "timer": ("time",),
    "temperature": ("temp",),
    "function": ("mode",),
    "doneness": ("toast",),
    "browning": ("toast",),
}
_TOKEN_STOPWORDS_BY_KIND = {
    "control": frozenset({"control", "joint", "knob", "button", "lever"}),
    "part": frozenset({"part", "joint"}),
    "support_site": frozenset(
        {"support", "site", "surface", "region", "placement", "place", "burner"}
    ),
}


@dataclass(frozen=True)
class FixtureSimulationMetadata:
    """Simulator-derived metadata for one symbolic fixture in a TaskSpec."""

    symbol_id: str
    concrete_id: str
    fixture_type: str
    part_ids: tuple[str, ...]
    control_ids: tuple[str, ...]
    support_site_ids: tuple[str, ...]


@dataclass(frozen=True)
class SimulationReferenceMetadata:
    """Simulator-derived lookup tables reused by validation and normalization."""

    fixture_metadata_by_symbol: dict[str, FixtureSimulationMetadata]
    location_aliases: dict[str, str]


@lru_cache(maxsize=None)
def _load_task_class(
    task_name: str,
    source_python_module: str | None,
) -> type[Any] | None:
    if not source_python_module:
        return None
    try:
        module = import_module(source_python_module)
    except Exception:
        return None
    task_class = getattr(module, task_name, None)
    if isinstance(task_class, type):
        return task_class
    return None


def _ordered_scene_candidates(
    *,
    task_name: str,
    source_python_module: str | None,
) -> tuple[tuple[int, int, int], ...]:
    task_class = _load_task_class(task_name, source_python_module)
    exclude_layouts = set(getattr(task_class, "EXCLUDE_LAYOUTS", ()) or ())
    exclude_styles = set(getattr(task_class, "EXCLUDE_STYLES", ()) or ())

    preferred_pairs = [
        (layout, style)
        for layout, style in _PREFERRED_LAYOUT_STYLE_IDS
        if layout not in exclude_layouts and style not in exclude_styles
    ]
    if preferred_pairs:
        pairs = preferred_pairs
    else:
        all_pairs = [
            (layout, style)
            for layout in unpack_layout_ids(None)
            for style in unpack_style_ids(None)
            if layout not in exclude_layouts and style not in exclude_styles
        ]
        pairs = all_pairs[:32]

    return tuple(
        (layout, style, seed)
        for layout, style in pairs
        for seed in _SIM_NORMALIZATION_SEEDS
    )


def _extract_trailing_int(token: str) -> int | None:
    match = re.search(r"(\d+)$", str(token))
    if match is None:
        return None
    return int(match.group(1))


def _slot_pair_index_from_support_site(site_id: str | None) -> int | None:
    if not isinstance(site_id, str):
        return None
    lowered = site_id.lower()
    if lowered.startswith("sidel_") or lowered in {"slotl", "slotr"}:
        return 0
    if lowered.startswith("sider_"):
        return 1
    return None


def _iter_support_site_distances(
    executor: SimToolExecutor,
    *,
    fixture_id: str,
    object_id: str,
) -> list[tuple[float, str]]:
    support_site_ids = executor.get_support_sites(fixture_id)
    if not support_site_ids:
        return []
    try:
        fixture = executor._require_fixture(fixture_id)
        object_pos, _object_quat = executor._get_object_pose(object_id)
        reset_regions = fixture.get_reset_regions(env=executor.env)
    except Exception:
        return []
    if not isinstance(reset_regions, dict):
        return []

    distances: list[tuple[float, str]] = []
    for support_site_id in support_site_ids:
        try:
            raw_site_id = executor._resolve_fixture_site_id(fixture_id, support_site_id)
        except Exception:
            raw_site_id = support_site_id
        region = reset_regions.get(raw_site_id)
        if not isinstance(region, dict):
            continue
        offset = region.get("offset", (0.0, 0.0, 0.0))
        try:
            world_pos = executor.runner._fixture_local_to_world(fixture, offset)
        except Exception:
            continue
        distance = float(
            ((object_pos[0] - world_pos[0]) ** 2 + (object_pos[1] - world_pos[1]) ** 2)
            ** 0.5
        )
        distances.append((distance, support_site_id))
    distances.sort()
    return distances


def _infer_object_support_site(
    executor: SimToolExecutor,
    *,
    fixture_id: str,
    object_id: str,
) -> str | None:
    distances = _iter_support_site_distances(
        executor,
        fixture_id=fixture_id,
        object_id=object_id,
    )
    if not distances:
        return None
    return distances[0][1]


def _infer_object_support_assignment(
    executor: SimToolExecutor,
    *,
    object_id: str,
    candidate_fixture_ids: tuple[str, ...],
) -> tuple[str, str] | None:
    best_assignment: tuple[float, str, str] | None = None
    for fixture_id in candidate_fixture_ids:
        for distance, support_site_id in _iter_support_site_distances(
            executor,
            fixture_id=fixture_id,
            object_id=object_id,
        ):
            candidate = (distance, fixture_id, support_site_id)
            if best_assignment is None or candidate < best_assignment:
                best_assignment = candidate
    if best_assignment is None:
        return None
    _distance, fixture_id, support_site_id = best_assignment
    return fixture_id, support_site_id


@lru_cache(maxsize=None)
def _load_task_simulation_snapshot(
    task_name: str,
    robots: int,
    source_python_module: str | None = None,
) -> dict[str, Any]:
    """Return cached simulator metadata for one task class."""
    failures: list[str] = []
    for layout, style, seed in _ordered_scene_candidates(
        task_name=task_name,
        source_python_module=source_python_module,
    ):
        executor = None
        try:
            executor = SimToolExecutor(
                task_name=task_name,
                robots=robots,
                layout=layout,
                style=style,
                seed=seed,
            )
            scene = deepcopy(executor.get_scene_description())
            fixture_details: dict[str, dict[str, Any]] = {}
            for fixture_id, fixture_info in (scene.get("fixtures") or {}).items():
                if not isinstance(fixture_id, str) or not isinstance(fixture_info, dict):
                    continue
                fixture_details[fixture_id] = {
                    "fixture_type": fixture_info.get("fixture_type"),
                    "parts": tuple(executor.get_parts(fixture_id)),
                    "controls": tuple(executor.get_controls(fixture_id)),
                    "support_sites": tuple(executor.get_support_sites(fixture_id)),
                    "parent_fixture": fixture_info.get("parent_fixture"),
                }
            task_refs = deepcopy(((executor.env.get_ep_meta() or {}).get("refs")) or {})
            object_support_sites: dict[str, str] = {}
            object_support_fixtures: dict[str, str] = {}
            candidate_fixture_ids = tuple(
                fixture_id
                for fixture_id, details in fixture_details.items()
                if details.get("support_sites")
            )
            for object_id in scene.get("objects") or {}:
                if not isinstance(object_id, str):
                    continue
                assignment = _infer_object_support_assignment(
                    executor,
                    object_id=object_id,
                    candidate_fixture_ids=candidate_fixture_ids,
                )
                if assignment is None:
                    continue
                fixture_id, support_site_id = assignment
                if isinstance(fixture_id, str) and isinstance(support_site_id, str):
                    object_support_fixtures[object_id] = fixture_id
                    object_support_sites[object_id] = support_site_id
            return {
                "scene": scene,
                "fixture_details": fixture_details,
                "task_refs": task_refs,
                "object_support_fixtures": object_support_fixtures,
                "object_support_sites": object_support_sites,
                "scene_config": {"layout": layout, "style": style, "seed": seed},
            }
        except Exception as exc:
            failures.append(
                f"L{layout}/S{style}/sd{seed}: {type(exc).__name__}: {exc}"
            )
        finally:
            if executor is not None:
                executor.close()
    failure_summary = "; ".join(failures[:4])
    if len(failures) > 4:
        failure_summary += f"; ... (+{len(failures) - 4} more)"
    raise ValueError(failure_summary or "failed to build simulation snapshot")


def _infer_robot_count(payload: dict[str, Any]) -> int:
    initial_state = payload.get("initial_state") or {}
    agents = initial_state.get("agents")
    if isinstance(agents, dict) and agents:
        return len(agents)
    agent_ids = payload.get("agent_ids")
    if isinstance(agent_ids, list) and agent_ids:
        return len([agent_id for agent_id in agent_ids if isinstance(agent_id, str)])
    return 2


def _normalize_tokens(token: str, *, kind: str) -> tuple[str, ...]:
    lowered = str(token).lower()
    stopwords = _TOKEN_STOPWORDS_BY_KIND.get(kind, frozenset())
    normalized_tokens: list[str] = []
    for raw_token in _TOKEN_RE.findall(lowered):
        for alias in _TOKEN_ALIASES.get(raw_token, (raw_token,)):
            if alias in stopwords or not alias:
                continue
            if alias not in normalized_tokens:
                normalized_tokens.append(alias)
    if normalized_tokens:
        return tuple(normalized_tokens)
    return (lowered.strip(),)


def _match_sim_token(
    requested_id: str,
    candidates: tuple[str, ...] | list[str],
    *,
    kind: str,
    location_aliases: dict[str, str] | None = None,
) -> str | None:
    candidate_list = [candidate for candidate in candidates if isinstance(candidate, str)]
    if not candidate_list:
        return None
    if requested_id in candidate_list:
        return requested_id

    lowered_requested_id = str(requested_id).lower()
    if kind == "support_site" and location_aliases and requested_id in location_aliases:
        aliased_location = location_aliases[requested_id]
        if aliased_location in candidate_list:
            return aliased_location
    if kind == "part":
        if (
            "sliding" in candidate_list
            and any(token in lowered_requested_id for token in ("drawer", "slide", "rack"))
        ):
            return "sliding"
        if (
            "hinged" in candidate_list
            and any(token in lowered_requested_id for token in ("door", "lid", "head"))
        ):
            return "hinged"
    if kind == "control":
        if (
            "switch" in candidate_list
            and any(token in lowered_requested_id for token in ("power", "start", "switch"))
            and "button" in lowered_requested_id
        ):
            return "switch"
        if isinstance(location_aliases, dict) and "_" in lowered_requested_id:
            role_prefix, control_family = lowered_requested_id.rsplit("_", 1)
            support_site_id = location_aliases.get(role_prefix)
            if not isinstance(support_site_id, str):
                support_site_id = location_aliases.get(f"{role_prefix}_slot")
            if not isinstance(support_site_id, str):
                support_site_id = location_aliases.get(f"{role_prefix}_burner")
            if isinstance(support_site_id, str):
                fixture_relative_control_id = f"{control_family}_{support_site_id}"
                if fixture_relative_control_id in candidate_list:
                    return fixture_relative_control_id
                slot_pair_index = _slot_pair_index_from_support_site(support_site_id)
                if slot_pair_index is not None:
                    paired_control_id = f"{control_family}_{slot_pair_index}"
                    if paired_control_id in candidate_list:
                        return paired_control_id

    requested_signature = _normalize_tokens(requested_id, kind=kind)
    exact_signature_matches = [
        candidate
        for candidate in candidate_list
        if _normalize_tokens(candidate, kind=kind) == requested_signature
    ]
    if len(exact_signature_matches) == 1:
        return exact_signature_matches[0]

    requested_text = "_".join(requested_signature)
    scored: list[tuple[float, str]] = []
    for candidate in candidate_list:
        candidate_signature = _normalize_tokens(candidate, kind=kind)
        candidate_text = "_".join(candidate_signature)
        overlap = len(set(requested_signature) & set(candidate_signature))
        union = len(set(requested_signature) | set(candidate_signature))
        jaccard = overlap / union if union else 0.0
        ratio = SequenceMatcher(None, requested_text, candidate_text).ratio()
        substring_bonus = 0.05 if requested_text and (
            requested_text in candidate_text or candidate_text in requested_text
        ) else 0.0
        score = max(jaccard, ratio) + substring_bonus
        scored.append((score, candidate))
    scored.sort(reverse=True)
    if not scored:
        return None
    best_score, best_candidate = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else 0.0
    if best_score >= 0.67 and (best_score - second_score) >= 0.08:
        return best_candidate
    ordinal = _extract_trailing_int(requested_id)
    if ordinal is not None and 1 <= ordinal <= len(candidate_list):
        return candidate_list[ordinal - 1]
    return None


def _resolve_symbolic_object_aliases(
    *,
    payload: dict[str, Any],
    scene: dict[str, Any],
) -> dict[str, str]:
    initial_state = payload.get("initial_state") or {}
    objects_by_symbol = initial_state.get("objects") or {}
    scene_objects = scene.get("objects") or {}
    if not isinstance(objects_by_symbol, dict) or not isinstance(scene_objects, dict):
        return {}

    env_object_ids = set(scene_objects)
    type_to_env_key: dict[str, str] = {}
    for env_key, obj_info in scene_objects.items():
        if not isinstance(env_key, str) or not isinstance(obj_info, dict):
            continue
        obj_type = str(obj_info.get("object_type", ""))
        type_to_env_key[obj_type] = env_key
        type_to_env_key[env_key] = env_key

    aliases: dict[str, str] = {}
    for symbol, obj_state in objects_by_symbol.items():
        if not isinstance(symbol, str) or not isinstance(obj_state, dict):
            continue
        object_type = str(obj_state.get("object_type", ""))
        resolved: str | None = None
        if symbol in env_object_ids:
            resolved = symbol
        elif object_type in env_object_ids:
            resolved = object_type
        elif object_type in type_to_env_key:
            resolved = type_to_env_key[object_type]
        else:
            for env_key in env_object_ids:
                if object_type and (object_type in env_key or env_key in object_type):
                    resolved = env_key
                    break
        if isinstance(resolved, str):
            aliases[symbol] = resolved
    return aliases


def _resolve_symbolic_fixture_metadata(
    *,
    task_name: str,
    payload: dict[str, Any],
    snapshot: dict[str, Any] | None = None,
    object_aliases: dict[str, str] | None = None,
    source_python_module: str | None = None,
) -> dict[str, FixtureSimulationMetadata]:
    robots = _infer_robot_count(payload)
    if snapshot is None:
        snapshot = _load_task_simulation_snapshot(
            task_name,
            robots,
            source_python_module,
        )
    scene = snapshot["scene"]
    fixture_details = snapshot["fixture_details"]
    initial_state = payload.get("initial_state") or {}
    fixtures_by_symbol = initial_state.get("fixtures") or {}
    objects_by_symbol = initial_state.get("objects") or {}
    grounding_symbols = ((payload.get("grounding") or {}).get("symbols") or {})
    if not isinstance(fixtures_by_symbol, dict):
        return {}

    if object_aliases is None:
        object_aliases = _resolve_symbolic_object_aliases(payload=payload, scene=scene)
    object_placements = scene.get("object_placements") or {}
    fixture_refs = scene.get("fixture_refs") or {}
    scene_fixtures = scene.get("fixtures") or {}

    object_locations_by_fixture: dict[str, list[str]] = {}
    for object_symbol, object_state in objects_by_symbol.items():
        if not isinstance(object_symbol, str) or not isinstance(object_state, dict):
            continue
        location = object_state.get("location")
        if isinstance(location, str):
            object_locations_by_fixture.setdefault(location, []).append(object_symbol)

    resolved_concrete_ids: dict[str, str] = {}

    def _resolve_one(symbol_id: str) -> str | None:
        if symbol_id in resolved_concrete_ids:
            return resolved_concrete_ids[symbol_id]
        if symbol_id in scene_fixtures:
            resolved_concrete_ids[symbol_id] = symbol_id
            return symbol_id

        fixture_state = fixtures_by_symbol.get(symbol_id) or {}
        fixture_type = (
            str(fixture_state.get("fixture_type", ""))
            if isinstance(fixture_state, dict)
            else ""
        )

        for object_symbol in object_locations_by_fixture.get(symbol_id, []):
            resolved_object = object_aliases.get(object_symbol)
            concrete_fixture = (
                object_placements.get(resolved_object)
                if isinstance(resolved_object, str)
                else None
            )
            if isinstance(concrete_fixture, str):
                resolved_concrete_ids[symbol_id] = concrete_fixture
                return concrete_fixture

        symbol_grounding = grounding_symbols.get(symbol_id) or {}
        if isinstance(symbol_grounding, dict):
            resolver = symbol_grounding.get("resolver")
            grounding_type = str(symbol_grounding.get("fixture_type", ""))
            candidate_type = grounding_type or fixture_type
            if resolver == "unique_fixture_type" and candidate_type:
                candidates = [
                    fixture_id
                    for fixture_id, details in fixture_details.items()
                    if details.get("fixture_type") == candidate_type
                ]
                if len(candidates) == 1:
                    resolved_concrete_ids[symbol_id] = candidates[0]
                    return candidates[0]

        if fixture_type:
            candidates = [
                fixture_id
                for fixture_id, details in fixture_details.items()
                if details.get("fixture_type") == fixture_type
            ]
            if len(candidates) == 1:
                resolved_concrete_ids[symbol_id] = candidates[0]
                return candidates[0]

        if symbol_id in fixture_refs and isinstance(fixture_refs[symbol_id], str):
            concrete_fixture = fixture_refs[symbol_id]
            resolved_concrete_ids[symbol_id] = concrete_fixture
            return concrete_fixture

        for role, concrete_fixture in fixture_refs.items():
            if not isinstance(role, str) or not isinstance(concrete_fixture, str):
                continue
            if role == fixture_type or (fixture_type and (fixture_type in role or role in fixture_type)):
                resolved_concrete_ids[symbol_id] = concrete_fixture
                return concrete_fixture

        if isinstance(symbol_grounding, dict):
            anchor_symbol = symbol_grounding.get("anchor_fixture_symbol")
            if isinstance(anchor_symbol, str):
                anchor_concrete_id = _resolve_one(anchor_symbol)
                if isinstance(anchor_concrete_id, str):
                    parent_fixture = (
                        fixture_details.get(anchor_concrete_id, {}).get("parent_fixture")
                    )
                    if isinstance(parent_fixture, str):
                        resolved_concrete_ids[symbol_id] = parent_fixture
                        return parent_fixture
        return None

    resolved: dict[str, FixtureSimulationMetadata] = {}
    for symbol_id, fixture_state in fixtures_by_symbol.items():
        if not isinstance(symbol_id, str) or not isinstance(fixture_state, dict):
            continue
        concrete_id = _resolve_one(symbol_id)
        if not isinstance(concrete_id, str):
            continue
        details = fixture_details.get(concrete_id) or {}
        resolved[symbol_id] = FixtureSimulationMetadata(
            symbol_id=symbol_id,
            concrete_id=concrete_id,
            fixture_type=str(details.get("fixture_type", fixture_state.get("fixture_type", ""))),
            part_ids=tuple(details.get("parts") or ()),
            control_ids=tuple(details.get("controls") or ()),
            support_site_ids=tuple(details.get("support_sites") or ()),
        )
    return resolved


def _support_site_parent_by_id(initial_state: dict[str, Any] | None) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not isinstance(initial_state, dict):
        return mapping
    for fixture_id, fixture_state in (initial_state.get("fixtures") or {}).items():
        if not isinstance(fixture_id, str) or not isinstance(fixture_state, dict):
            continue
        support_sites = fixture_state.get("support_sites") or {}
        if isinstance(support_sites, dict):
            for support_site_id in support_sites:
                if isinstance(support_site_id, str):
                    mapping[support_site_id] = fixture_id
    return mapping


def _build_location_aliases(
    *,
    payload: dict[str, Any],
    snapshot: dict[str, Any],
    object_aliases: dict[str, str],
) -> dict[str, str]:
    """Map semantic location-role ids in the spec to simulator support-site ids."""

    aliases: dict[str, str] = {}
    initial_state = payload.get("initial_state") or {}
    objects_by_symbol = initial_state.get("objects") or {}
    if not isinstance(objects_by_symbol, dict):
        return aliases

    object_support_sites = snapshot.get("object_support_sites") or {}
    task_refs = snapshot.get("task_refs") or {}
    fixtures_by_symbol = initial_state.get("fixtures") or {}

    for object_symbol, object_state in objects_by_symbol.items():
        if not isinstance(object_symbol, str) or not isinstance(object_state, dict):
            continue
        symbolic_location = object_state.get("location")
        if not isinstance(symbolic_location, str):
            continue
        if symbolic_location in fixtures_by_symbol or symbolic_location in objects_by_symbol:
            continue
        env_object_id = object_aliases.get(object_symbol)
        if not isinstance(env_object_id, str):
            continue
        actual_support_site = object_support_sites.get(env_object_id)
        if isinstance(actual_support_site, str):
            aliases[symbolic_location] = actual_support_site

    for fixture_symbol, fixture_state in fixtures_by_symbol.items():
        if not isinstance(fixture_symbol, str) or not isinstance(fixture_state, dict):
            continue
        support_sites = fixture_state.get("support_sites") or {}
        if not isinstance(support_sites, dict):
            continue

        actual_sites_for_fixture = tuple(
            dict.fromkeys(
                actual_support_site
                for object_symbol, object_state in objects_by_symbol.items()
                if isinstance(object_symbol, str)
                and isinstance(object_state, dict)
                and object_state.get("location") == fixture_symbol
                and isinstance(object_aliases.get(object_symbol), str)
                and isinstance(
                    object_support_sites.get(object_aliases[object_symbol]),
                    str,
                )
                for actual_support_site in [
                    object_support_sites[object_aliases[object_symbol]]
                ]
            )
        )
        unresolved_symbolic_sites = [
            support_site_id
            for support_site_id in support_sites
            if isinstance(support_site_id, str)
            and support_site_id not in aliases
        ]
        if len(unresolved_symbolic_sites) == 1 and len(actual_sites_for_fixture) == 1:
            aliases[unresolved_symbolic_sites[0]] = actual_sites_for_fixture[0]

    knob_ref = task_refs.get("knob")
    if isinstance(knob_ref, str):
        for fixture_state in fixtures_by_symbol.values():
            if not isinstance(fixture_state, dict):
                continue
            support_sites = fixture_state.get("support_sites") or {}
            if not isinstance(support_sites, dict):
                continue
            for support_site_id in support_sites:
                if (
                    isinstance(support_site_id, str)
                    and support_site_id not in aliases
                    and "burner" in support_site_id.lower()
                ):
                    aliases[support_site_id] = knob_ref

    return aliases


def _resolve_unique_support_site_match(
    *,
    requested_id: str,
    fixture_metadata_by_symbol: dict[str, FixtureSimulationMetadata],
    location_aliases: dict[str, str] | None,
) -> tuple[str, str] | None:
    """Resolve a symbolic support-site alias to one concrete fixture/site pair."""

    if not isinstance(requested_id, str) or not requested_id:
        return None

    requested_lower = requested_id.lower()
    direct_matches: list[tuple[str, str]] = []
    for fixture_id, metadata in fixture_metadata_by_symbol.items():
        if not metadata.support_site_ids:
            continue
        resolved_id = _match_sim_token(
            requested_id,
            metadata.support_site_ids,
            kind="support_site",
            location_aliases=location_aliases,
        )
        if isinstance(resolved_id, str):
            direct_matches.append((fixture_id, resolved_id))

    deduped_direct_matches = tuple(dict.fromkeys(direct_matches))
    if len(deduped_direct_matches) == 1:
        return deduped_direct_matches[0]

    heuristic_matches: list[tuple[str, str]] = []
    for fixture_id, metadata in fixture_metadata_by_symbol.items():
        support_site_ids = tuple(
            support_site_id
            for support_site_id in metadata.support_site_ids
            if isinstance(support_site_id, str)
        )
        if not support_site_ids:
            continue

        if metadata.fixture_type == "sink" and "basin" in requested_lower:
            basin_sites = [
                support_site_id
                for support_site_id in support_site_ids
                if "basin" in support_site_id.lower()
            ]
            if basin_sites:
                heuristic_matches.append((fixture_id, sorted(basin_sites)[0]))
                continue

        if metadata.fixture_type in {"toaster_oven", "oven", "dishwasher"} and any(
            token in requested_lower for token in ("rack", "tray")
        ):
            rack_sites = [
                support_site_id
                for support_site_id in support_site_ids
                if any(token in support_site_id.lower() for token in ("rack", "tray"))
            ]
            if rack_sites:
                heuristic_matches.append((fixture_id, sorted(rack_sites)[0]))

    deduped_heuristic_matches = tuple(dict.fromkeys(heuristic_matches))
    if len(deduped_heuristic_matches) == 1:
        return deduped_heuristic_matches[0]
    return None


def _iter_symbolic_location_tokens(payload: dict[str, Any]) -> tuple[str, ...]:
    """Collect potentially unresolved symbolic location/support-site ids from a spec."""

    initial_state = payload.get("initial_state") or {}
    fixtures_by_id = initial_state.get("fixtures") or {}
    objects_by_id = initial_state.get("objects") or {}
    known_symbol_ids = {
        symbol_id
        for symbol_id in [*fixtures_by_id.keys(), *objects_by_id.keys()]
        if isinstance(symbol_id, str)
    }

    tokens: list[str] = []

    def _append_token(value: Any) -> None:
        if not isinstance(value, str):
            return
        if value in known_symbol_ids or value.startswith("held_by_"):
            return
        if value not in tokens:
            tokens.append(value)

    for object_state in objects_by_id.values():
        if isinstance(object_state, dict):
            _append_token(object_state.get("location"))

    for fixture_state in fixtures_by_id.values():
        if not isinstance(fixture_state, dict):
            continue
        support_sites = fixture_state.get("support_sites") or {}
        if isinstance(support_sites, dict):
            for support_site_id in support_sites:
                _append_token(support_site_id)

    for tool_spec in (payload.get("allowed_tool_specs") or {}).values():
        if not isinstance(tool_spec, dict):
            continue
        for location_key in (
            "allowed_support_ids",
            "allowed_source_ids",
            "allowed_receptacle_ids",
            "allowed_target_site_ids",
        ):
            for value in tool_spec.get(location_key) or []:
                _append_token(value)

    for condition in payload.get("goal_conditions") or []:
        if not isinstance(condition, dict):
            continue
        _append_token(condition.get("location"))
        for value in condition.get("locations") or []:
            _append_token(value)

    for condition in payload.get("task_preconditions") or []:
        if not isinstance(condition, dict):
            continue
        _append_token(condition.get("location"))
        _append_token(condition.get("required_location"))
        _append_token(condition.get("arg_value"))

    for effect in payload.get("task_effects") or []:
        if not isinstance(effect, dict):
            continue
        effect_args = effect.get("args") or {}
        if isinstance(effect_args, dict):
            for arg_name in (
                "source_id",
                "support_id",
                "receptacle_id",
                "target_id",
                "target_site_id",
                "reference_id",
            ):
                _append_token(effect_args.get(arg_name))
        for requirement in effect.get("required_object_locations") or []:
            if isinstance(requirement, dict):
                _append_token(requirement.get("location"))

    trajectory = payload.get("example_trajectory") or {}
    if isinstance(trajectory, dict):
        for step in trajectory.get("steps") or []:
            if not isinstance(step, dict):
                continue
            args = step.get("args") or {}
            if not isinstance(args, dict):
                continue
            for arg_name in (
                "source_id",
                "support_id",
                "receptacle_id",
                "target_id",
                "target_site_id",
                "reference_id",
            ):
                _append_token(args.get(arg_name))

    return tuple(tokens)


def _build_sim_support_site_aliases(
    *,
    payload: dict[str, Any],
    fixture_metadata_by_symbol: dict[str, FixtureSimulationMetadata],
    location_aliases: dict[str, str] | None,
) -> dict[str, str]:
    """Infer symbolic support-site aliases directly from simulator metadata."""

    aliases: dict[str, str] = {}
    merged_aliases = dict(location_aliases or {})
    for requested_id in _iter_symbolic_location_tokens(payload):
        match = _resolve_unique_support_site_match(
            requested_id=requested_id,
            fixture_metadata_by_symbol=fixture_metadata_by_symbol,
            location_aliases=merged_aliases,
        )
        if match is None:
            continue
        _fixture_id, support_site_id = match
        aliases[requested_id] = support_site_id
        merged_aliases[requested_id] = support_site_id
    return aliases


def _infer_support_site_parent_fixture(
    *,
    site_id: str,
    object_id: str | None,
    initial_state: dict[str, Any] | None,
    grounding: dict[str, Any] | None,
) -> str | None:
    if not isinstance(initial_state, dict):
        return None

    parent_by_support_site = _support_site_parent_by_id(initial_state)
    if site_id in parent_by_support_site:
        return parent_by_support_site[site_id]

    fixtures_by_id = initial_state.get("fixtures") or {}
    if not isinstance(fixtures_by_id, dict):
        return None

    if isinstance(object_id, str) and isinstance(grounding, dict):
        symbol_spec = (grounding.get("symbols") or {}).get(object_id, {})
        preferred_fixture_types = [
            fixture_type
            for fixture_type in symbol_spec.get("preferred_fixture_types") or []
            if isinstance(fixture_type, str)
        ]
        candidate_fixture_ids = [
            fixture_id
            for fixture_id, fixture_state in fixtures_by_id.items()
            if isinstance(fixture_id, str)
            and isinstance(fixture_state, dict)
            and fixture_state.get("fixture_type") in preferred_fixture_types
        ]
        deduped_candidate_fixture_ids = tuple(dict.fromkeys(candidate_fixture_ids))
        if len(deduped_candidate_fixture_ids) == 1:
            return deduped_candidate_fixture_ids[0]

    name_matched_fixture_ids = [
        fixture_id
        for fixture_id, fixture_state in fixtures_by_id.items()
        if isinstance(fixture_id, str)
        and isinstance(fixture_state, dict)
        and (
            fixture_id in site_id
            or (
                isinstance(fixture_state.get("fixture_type"), str)
                and fixture_state["fixture_type"] in site_id
            )
        )
    ]
    deduped_name_matched_fixture_ids = tuple(dict.fromkeys(name_matched_fixture_ids))
    if len(deduped_name_matched_fixture_ids) == 1:
        return deduped_name_matched_fixture_ids[0]
    return None


def _rename_fixture_section_keys(
    fixture_state: dict[str, Any],
    *,
    section_name: str,
    candidates: tuple[str, ...],
    kind: str,
    location_aliases: dict[str, str] | None,
    errors: list[str],
    error_prefix: str,
) -> dict[str, str]:
    had_section = section_name in fixture_state
    section = fixture_state.get(section_name)
    if not isinstance(section, dict):
        return {}
    renamed_section: dict[str, Any] = {}
    aliases: dict[str, str] = {}
    for requested_id, value in section.items():
        if not isinstance(requested_id, str):
            continue
        resolved_id = _match_sim_token(
            requested_id,
            candidates,
            kind=kind,
            location_aliases=location_aliases,
        )
        if resolved_id is None:
            errors.append(
                f"{error_prefix}.{section_name}[{requested_id!r}] does not match simulator {section_name.rstrip('s')} ids {sorted(candidates)}"
            )
            resolved_id = requested_id
        aliases[requested_id] = resolved_id
        if resolved_id in renamed_section and isinstance(renamed_section[resolved_id], dict) and isinstance(value, dict):
            merged_value = dict(renamed_section[resolved_id])
            merged_value.update(value)
            renamed_section[resolved_id] = merged_value
        else:
            renamed_section[resolved_id] = deepcopy(value)
    if renamed_section or had_section:
        fixture_state[section_name] = renamed_section
    return aliases


def _normalize_token_for_fixture(
    requested_id: Any,
    *,
    fixture_id: str | None,
    fixture_metadata_by_symbol: dict[str, FixtureSimulationMetadata],
    kind: str,
    location_aliases: dict[str, str] | None,
    errors: list[str],
    context: str,
) -> Any:
    if not isinstance(requested_id, str) or not isinstance(fixture_id, str):
        return requested_id
    metadata = fixture_metadata_by_symbol.get(fixture_id)
    if metadata is None:
        return requested_id
    candidates = {
        "control": metadata.control_ids,
        "part": metadata.part_ids,
        "support_site": metadata.support_site_ids,
    }.get(kind, ())
    if not candidates:
        return requested_id
    resolved_id = _match_sim_token(
        requested_id,
        candidates,
        kind=kind,
        location_aliases=location_aliases,
    )
    if resolved_id is None:
        errors.append(
            f"{context}: {kind}_id {requested_id!r} does not match simulator ids {sorted(candidates)} for fixture {fixture_id!r}"
        )
        return requested_id
    return resolved_id


def _normalize_location_value(
    requested_location: Any,
    *,
    object_id: str | None,
    initial_state: dict[str, Any] | None,
    grounding: dict[str, Any] | None,
    fixture_metadata_by_symbol: dict[str, FixtureSimulationMetadata],
    location_aliases: dict[str, str] | None,
    errors: list[str],
    context: str,
) -> Any:
    if not isinstance(requested_location, str):
        return requested_location
    if requested_location.startswith("held_by_"):
        return requested_location
    fixtures_by_id = (initial_state or {}).get("fixtures") or {}
    objects_by_id = (initial_state or {}).get("objects") or {}
    if requested_location in fixtures_by_id or requested_location in objects_by_id:
        return requested_location
    if (
        isinstance(location_aliases, dict)
        and requested_location in location_aliases
        and isinstance(location_aliases[requested_location], str)
    ):
        return location_aliases[requested_location]
    support_site_match = _resolve_unique_support_site_match(
        requested_id=requested_location,
        fixture_metadata_by_symbol=fixture_metadata_by_symbol,
        location_aliases=location_aliases,
    )
    if support_site_match is not None:
        _fixture_id, support_site_id = support_site_match
        return support_site_id
    parent_fixture_id = _infer_support_site_parent_fixture(
        site_id=requested_location,
        object_id=object_id,
        initial_state=initial_state,
        grounding=grounding,
    )
    if not isinstance(parent_fixture_id, str):
        return requested_location
    return _normalize_token_for_fixture(
        requested_location,
        fixture_id=parent_fixture_id,
        fixture_metadata_by_symbol=fixture_metadata_by_symbol,
        kind="support_site",
        location_aliases=location_aliases,
        errors=errors,
        context=context,
    )


def _normalize_allowed_tool_specs(
    *,
    payload: dict[str, Any],
    fixture_metadata_by_symbol: dict[str, FixtureSimulationMetadata],
    location_aliases: dict[str, str] | None,
    errors: list[str],
) -> None:
    allowed_tool_specs = payload.get("allowed_tool_specs") or {}
    if not isinstance(allowed_tool_specs, dict):
        return

    def _normalize_allowed_fixture_token_ids(
        requested_ids: list[Any],
        *,
        target_ids: list[str],
        kind: str,
        context: str,
    ) -> list[Any]:
        normalized_ids: list[Any] = []
        for requested_id in requested_ids:
            if not isinstance(requested_id, str) or not target_ids:
                normalized_ids.append(requested_id)
                continue

            matched_ids: list[str] = []
            candidate_targets: list[str] = []
            candidate_id_sets: dict[str, list[str]] = {}
            for target_id in target_ids:
                metadata = fixture_metadata_by_symbol.get(target_id)
                if metadata is None:
                    continue
                candidates = {
                    "control": metadata.control_ids,
                    "part": metadata.part_ids,
                }.get(kind, ())
                if not candidates:
                    continue
                candidate_targets.append(target_id)
                candidate_id_sets[target_id] = sorted(candidates)
                resolved_id = _match_sim_token(
                    requested_id,
                    candidates,
                    kind=kind,
                    location_aliases=location_aliases,
                )
                if isinstance(resolved_id, str):
                    matched_ids.append(resolved_id)

            if matched_ids:
                for matched_id in dict.fromkeys(matched_ids):
                    if matched_id not in normalized_ids:
                        normalized_ids.append(matched_id)
                continue

            if len(target_ids) == 1:
                normalized_ids.append(
                    _normalize_token_for_fixture(
                        requested_id,
                        fixture_id=target_ids[0],
                        fixture_metadata_by_symbol=fixture_metadata_by_symbol,
                        kind=kind,
                        location_aliases=location_aliases,
                        errors=errors,
                        context=context,
                    )
                )
                continue

            if candidate_targets:
                target_details = ", ".join(
                    f"{target_id}={candidate_id_sets.get(target_id, [])}"
                    for target_id in candidate_targets
                )
                errors.append(
                    f"{context}: {kind}_id {requested_id!r} does not match simulator ids across allowed targets ({target_details})"
                )
            normalized_ids.append(requested_id)
        return normalized_ids

    for tool_name, tool_spec in allowed_tool_specs.items():
        if not isinstance(tool_spec, dict):
            continue

        allowed_target_ids = [
            target_id
            for target_id in tool_spec.get("allowed_target_ids") or []
            if isinstance(target_id, str)
        ]
        if isinstance(tool_spec.get("allowed_control_ids"), list):
            tool_spec["allowed_control_ids"] = _normalize_allowed_fixture_token_ids(
                list(tool_spec["allowed_control_ids"]),
                target_ids=allowed_target_ids,
                kind="control",
                context=f"allowed_tool_specs[{tool_name!r}]",
            )
        if isinstance(tool_spec.get("allowed_part_ids"), list):
            tool_spec["allowed_part_ids"] = _normalize_allowed_fixture_token_ids(
                list(tool_spec["allowed_part_ids"]),
                target_ids=allowed_target_ids,
                kind="part",
                context=f"allowed_tool_specs[{tool_name!r}]",
            )
        support_fixture_ids = [
            fixture_id
            for fixture_id, metadata in fixture_metadata_by_symbol.items()
            if metadata.support_site_ids
        ]
        default_support_fixture_id = (
            support_fixture_ids[0] if len(support_fixture_ids) == 1 else None
        )
        for location_key in (
            "allowed_support_ids",
            "allowed_source_ids",
            "allowed_receptacle_ids",
            "allowed_target_site_ids",
        ):
            if isinstance(tool_spec.get(location_key), list):
                normalized_location_ids: list[Any] = []
                for location_id in tool_spec[location_key]:
                    normalized_location_id = _normalize_location_value(
                        location_id,
                        object_id=None,
                        initial_state=payload.get("initial_state"),
                        grounding=payload.get("grounding"),
                        fixture_metadata_by_symbol=fixture_metadata_by_symbol,
                        location_aliases=location_aliases,
                        errors=errors,
                        context=f"allowed_tool_specs[{tool_name!r}].{location_key}",
                    )
                    if (
                        location_key == "allowed_support_ids"
                        and isinstance(location_id, str)
                        and isinstance(normalized_location_id, str)
                        and location_id == normalized_location_id
                        and isinstance(default_support_fixture_id, str)
                        and location_id not in ((payload.get("initial_state") or {}).get("fixtures") or {})
                    ):
                        normalized_location_id = _normalize_token_for_fixture(
                            location_id,
                            fixture_id=default_support_fixture_id,
                            fixture_metadata_by_symbol=fixture_metadata_by_symbol,
                            kind="support_site",
                            location_aliases=location_aliases,
                            errors=errors,
                            context=f"allowed_tool_specs[{tool_name!r}].{location_key}",
                        )
                    normalized_location_ids.append(normalized_location_id)
                tool_spec[location_key] = normalized_location_ids


def _normalize_task_effects_and_trajectory(
    *,
    payload: dict[str, Any],
    fixture_metadata_by_symbol: dict[str, FixtureSimulationMetadata],
    location_aliases: dict[str, str] | None,
    errors: list[str],
) -> None:
    initial_state = payload.get("initial_state")
    grounding = payload.get("grounding")

    def _normalize_step_like(step_like: dict[str, Any], context: str) -> None:
        args = step_like.get("args")
        if not isinstance(args, dict):
            return
        target_id = args.get("target_id")
        object_id = args.get("object_id")
        if "control_id" in args:
            args["control_id"] = _normalize_token_for_fixture(
                args.get("control_id"),
                fixture_id=target_id if isinstance(target_id, str) else None,
                fixture_metadata_by_symbol=fixture_metadata_by_symbol,
                kind="control",
                location_aliases=location_aliases,
                errors=errors,
                context=context,
            )
        if "part_id" in args:
            args["part_id"] = _normalize_token_for_fixture(
                args.get("part_id"),
                fixture_id=target_id if isinstance(target_id, str) else None,
                fixture_metadata_by_symbol=fixture_metadata_by_symbol,
                kind="part",
                location_aliases=location_aliases,
                errors=errors,
                context=context,
            )
        for arg_name in (
            "source_id",
            "support_id",
            "receptacle_id",
            "target_id",
            "target_site_id",
            "reference_id",
        ):
            if arg_name in args:
                args[arg_name] = _normalize_location_value(
                    args.get(arg_name),
                    object_id=object_id if isinstance(object_id, str) else None,
                    initial_state=initial_state,
                    grounding=grounding,
                    fixture_metadata_by_symbol=fixture_metadata_by_symbol,
                    location_aliases=location_aliases,
                    errors=errors,
                    context=f"{context}.{arg_name}",
                )

    for index, effect in enumerate(payload.get("task_effects") or []):
        if not isinstance(effect, dict):
            continue
        _normalize_step_like(effect, f"task_effects[{index}]")
        for requirement in effect.get("required_fixture_controls") or []:
            if not isinstance(requirement, dict):
                continue
            requirement["control_id"] = _normalize_token_for_fixture(
                requirement.get("control_id"),
                fixture_id=requirement.get("fixture_id")
                if isinstance(requirement.get("fixture_id"), str)
                else None,
                fixture_metadata_by_symbol=fixture_metadata_by_symbol,
                kind="control",
                location_aliases=location_aliases,
                errors=errors,
                context=f"task_effects[{index}].required_fixture_controls",
            )
        for requirement in effect.get("required_object_locations") or []:
            if not isinstance(requirement, dict):
                continue
            requirement["location"] = _normalize_location_value(
                requirement.get("location"),
                object_id=requirement.get("object_id")
                if isinstance(requirement.get("object_id"), str)
                else None,
                initial_state=initial_state,
                grounding=grounding,
                fixture_metadata_by_symbol=fixture_metadata_by_symbol,
                location_aliases=location_aliases,
                errors=errors,
                context=f"task_effects[{index}].required_object_locations",
            )

    trajectory = payload.get("example_trajectory") or {}
    if isinstance(trajectory, dict):
        for index, step in enumerate(trajectory.get("steps") or []):
            if not isinstance(step, dict):
                continue
            _normalize_step_like(step, f"example_trajectory.steps[{index}]")


def _collect_referenced_support_sites_by_fixture(
    *,
    payload: dict[str, Any],
    fixture_metadata_by_symbol: dict[str, FixtureSimulationMetadata],
    location_aliases: dict[str, str] | None,
) -> dict[str, set[str]]:
    """Collect concrete support-site ids referenced anywhere in the spec."""

    referenced_support_sites_by_fixture: dict[str, set[str]] = {}

    def _record(value: Any) -> None:
        if not isinstance(value, str):
            return
        match = _resolve_unique_support_site_match(
            requested_id=value,
            fixture_metadata_by_symbol=fixture_metadata_by_symbol,
            location_aliases=location_aliases,
        )
        if match is None:
            return
        fixture_id, support_site_id = match
        referenced_support_sites_by_fixture.setdefault(fixture_id, set()).add(
            support_site_id
        )

    initial_state = payload.get("initial_state") or {}
    for object_state in (initial_state.get("objects") or {}).values():
        if isinstance(object_state, dict):
            _record(object_state.get("location"))

    for tool_spec in (payload.get("allowed_tool_specs") or {}).values():
        if not isinstance(tool_spec, dict):
            continue
        for location_key in (
            "allowed_support_ids",
            "allowed_source_ids",
            "allowed_receptacle_ids",
        ):
            for value in tool_spec.get(location_key) or []:
                _record(value)

    for condition in payload.get("goal_conditions") or []:
        if not isinstance(condition, dict):
            continue
        _record(condition.get("location"))
        for value in condition.get("locations") or []:
            _record(value)

    for condition in payload.get("task_preconditions") or []:
        if not isinstance(condition, dict):
            continue
        _record(condition.get("location"))
        _record(condition.get("required_location"))
        _record(condition.get("arg_value"))

    for effect in payload.get("task_effects") or []:
        if not isinstance(effect, dict):
            continue
        effect_args = effect.get("args") or {}
        if isinstance(effect_args, dict):
            for arg_name in (
                "source_id",
                "support_id",
                "receptacle_id",
                "target_id",
                "target_site_id",
                "reference_id",
            ):
                _record(effect_args.get(arg_name))
        for requirement in effect.get("required_object_locations") or []:
            if isinstance(requirement, dict):
                _record(requirement.get("location"))

    trajectory = payload.get("example_trajectory") or {}
    if isinstance(trajectory, dict):
        for step in trajectory.get("steps") or []:
            if not isinstance(step, dict):
                continue
            args = step.get("args") or {}
            if not isinstance(args, dict):
                continue
            for arg_name in (
                "source_id",
                "support_id",
                "receptacle_id",
                "target_id",
                "target_site_id",
                "reference_id",
            ):
                _record(args.get(arg_name))

    return referenced_support_sites_by_fixture


def _normalize_stove_control_aliases(
    *,
    payload: dict[str, Any],
    fixture_metadata_by_symbol: dict[str, FixtureSimulationMetadata],
    location_aliases: dict[str, str] | None,
) -> None:
    """Rewrite generic stove knob ids only when burner context is unique."""

    generic_control_aliases = frozenset(
        {"knob", "burner_knob", "stove_knob", "control", "knob_joint"}
    )
    referenced_support_sites_by_fixture = _collect_referenced_support_sites_by_fixture(
        payload=payload,
        fixture_metadata_by_symbol=fixture_metadata_by_symbol,
        location_aliases=location_aliases,
    )

    target_control_by_fixture: dict[str, str] = {}
    for fixture_id, metadata in fixture_metadata_by_symbol.items():
        if metadata.fixture_type != "stove":
            continue
        referenced_support_sites = referenced_support_sites_by_fixture.get(fixture_id) or set()
        if len(referenced_support_sites) != 1:
            continue
        support_site_id = next(iter(referenced_support_sites))
        candidate_control_ids = tuple(
            control_id
            for control_id in metadata.control_ids
            if isinstance(control_id, str)
        )
        direct_control_id = f"knob_{support_site_id}"
        if direct_control_id in candidate_control_ids:
            target_control_by_fixture[fixture_id] = direct_control_id
            continue
        matching_control_ids = [
            control_id
            for control_id in candidate_control_ids
            if control_id.startswith("knob_") and support_site_id in control_id
        ]
        if len(matching_control_ids) == 1:
            target_control_by_fixture[fixture_id] = matching_control_ids[0]

    if not target_control_by_fixture:
        return

    def _rewrite_control_id(value: Any, *, fixture_id: str | None) -> Any:
        if not isinstance(value, str) or not isinstance(fixture_id, str):
            return value
        if value.lower() not in generic_control_aliases:
            return value
        return target_control_by_fixture.get(fixture_id, value)

    initial_state = payload.get("initial_state") or {}
    for fixture_id, fixture_state in (initial_state.get("fixtures") or {}).items():
        if not isinstance(fixture_id, str) or not isinstance(fixture_state, dict):
            continue
        target_control_id = target_control_by_fixture.get(fixture_id)
        if not isinstance(target_control_id, str):
            continue
        controls = fixture_state.get("controls") or {}
        if not isinstance(controls, dict):
            continue
        rewritten_controls: dict[str, Any] = {}
        for control_id, control_state in controls.items():
            rewritten_control_id = _rewrite_control_id(control_id, fixture_id=fixture_id)
            rewritten_controls[rewritten_control_id] = control_state
        fixture_state["controls"] = rewritten_controls

    for tool_spec in (payload.get("allowed_tool_specs") or {}).values():
        if not isinstance(tool_spec, dict):
            continue
        target_ids = [
            target_id
            for target_id in tool_spec.get("allowed_target_ids") or []
            if isinstance(target_id, str) and target_id in target_control_by_fixture
        ]
        if len(target_ids) != 1:
            continue
        fixture_id = target_ids[0]
        if isinstance(tool_spec.get("allowed_control_ids"), list):
            tool_spec["allowed_control_ids"] = [
                _rewrite_control_id(control_id, fixture_id=fixture_id)
                for control_id in tool_spec["allowed_control_ids"]
            ]

    for condition in payload.get("goal_conditions") or []:
        if not isinstance(condition, dict):
            continue
        condition["control_id"] = _rewrite_control_id(
            condition.get("control_id"),
            fixture_id=condition.get("fixture_id")
            if isinstance(condition.get("fixture_id"), str)
            else None,
        )

    for effect in payload.get("task_effects") or []:
        if not isinstance(effect, dict):
            continue
        args = effect.get("args") or {}
        if isinstance(args, dict):
            args["control_id"] = _rewrite_control_id(
                args.get("control_id"),
                fixture_id=args.get("target_id")
                if isinstance(args.get("target_id"), str)
                else None,
            )
        for requirement in effect.get("required_fixture_controls") or []:
            if not isinstance(requirement, dict):
                continue
            requirement["control_id"] = _rewrite_control_id(
                requirement.get("control_id"),
                fixture_id=requirement.get("fixture_id")
                if isinstance(requirement.get("fixture_id"), str)
                else None,
            )

    trajectory = payload.get("example_trajectory") or {}
    if isinstance(trajectory, dict):
        for step in trajectory.get("steps") or []:
            if not isinstance(step, dict):
                continue
            args = step.get("args") or {}
            if not isinstance(args, dict):
                continue
            args["control_id"] = _rewrite_control_id(
                args.get("control_id"),
                fixture_id=args.get("target_id")
                if isinstance(args.get("target_id"), str)
                else None,
            )


def _normalize_goal_conditions_and_preconditions(
    *,
    payload: dict[str, Any],
    fixture_metadata_by_symbol: dict[str, FixtureSimulationMetadata],
    location_aliases: dict[str, str] | None,
    errors: list[str],
) -> None:
    initial_state = payload.get("initial_state")
    grounding = payload.get("grounding")

    for index, condition in enumerate(payload.get("goal_conditions") or []):
        if not isinstance(condition, dict):
            continue
        if "control_id" in condition:
            condition["control_id"] = _normalize_token_for_fixture(
                condition.get("control_id"),
                fixture_id=condition.get("fixture_id")
                if isinstance(condition.get("fixture_id"), str)
                else None,
                fixture_metadata_by_symbol=fixture_metadata_by_symbol,
                kind="control",
                location_aliases=location_aliases,
                errors=errors,
                context=f"goal_conditions[{index}]",
            )
        if "part_id" in condition:
            condition["part_id"] = _normalize_token_for_fixture(
                condition.get("part_id"),
                fixture_id=condition.get("fixture_id")
                if isinstance(condition.get("fixture_id"), str)
                else None,
                fixture_metadata_by_symbol=fixture_metadata_by_symbol,
                kind="part",
                location_aliases=location_aliases,
                errors=errors,
                context=f"goal_conditions[{index}]",
            )
        if "location" in condition:
            condition["location"] = _normalize_location_value(
                condition.get("location"),
                object_id=condition.get("object_id")
                if isinstance(condition.get("object_id"), str)
                else None,
                initial_state=initial_state,
                grounding=grounding,
                fixture_metadata_by_symbol=fixture_metadata_by_symbol,
                location_aliases=location_aliases,
                errors=errors,
                context=f"goal_conditions[{index}]",
            )
        if isinstance(condition.get("locations"), list):
            condition["locations"] = [
                _normalize_location_value(
                    location,
                    object_id=condition.get("object_id")
                    if isinstance(condition.get("object_id"), str)
                    else None,
                    initial_state=initial_state,
                    grounding=grounding,
                    fixture_metadata_by_symbol=fixture_metadata_by_symbol,
                    location_aliases=location_aliases,
                    errors=errors,
                    context=f"goal_conditions[{index}].locations",
                )
                for location in condition["locations"]
            ]

    for index, condition in enumerate(payload.get("task_preconditions") or []):
        if not isinstance(condition, dict):
            continue
        if "part_id" in condition:
            condition["part_id"] = _normalize_token_for_fixture(
                condition.get("part_id"),
                fixture_id=condition.get("fixture_id")
                if isinstance(condition.get("fixture_id"), str)
                else None,
                fixture_metadata_by_symbol=fixture_metadata_by_symbol,
                kind="part",
                location_aliases=location_aliases,
                errors=errors,
                context=f"task_preconditions[{index}]",
            )
        if "location" in condition:
            condition["location"] = _normalize_location_value(
                condition.get("location"),
                object_id=condition.get("object_id")
                if isinstance(condition.get("object_id"), str)
                else None,
                initial_state=initial_state,
                grounding=grounding,
                fixture_metadata_by_symbol=fixture_metadata_by_symbol,
                location_aliases=location_aliases,
                errors=errors,
                context=f"task_preconditions[{index}]",
            )
        if "required_location" in condition:
            condition["required_location"] = _normalize_location_value(
                condition.get("required_location"),
                object_id=condition.get("object_id")
                if isinstance(condition.get("object_id"), str)
                else None,
                initial_state=initial_state,
                grounding=grounding,
                fixture_metadata_by_symbol=fixture_metadata_by_symbol,
                location_aliases=location_aliases,
                errors=errors,
                context=f"task_preconditions[{index}]",
            )


def _normalize_initial_state(
    *,
    payload: dict[str, Any],
    fixture_metadata_by_symbol: dict[str, FixtureSimulationMetadata],
    location_aliases: dict[str, str] | None,
    errors: list[str],
) -> None:
    initial_state = payload.get("initial_state")
    grounding = payload.get("grounding")
    if not isinstance(initial_state, dict):
        return
    fixtures_by_id = initial_state.get("fixtures") or {}
    objects_by_id = initial_state.get("objects") or {}
    if not isinstance(fixtures_by_id, dict):
        return

    for fixture_id, fixture_state in fixtures_by_id.items():
        if not isinstance(fixture_id, str) or not isinstance(fixture_state, dict):
            continue
        metadata = fixture_metadata_by_symbol.get(fixture_id)
        if metadata is None:
            continue
        _rename_fixture_section_keys(
            fixture_state,
            section_name="controls",
            candidates=metadata.control_ids,
            kind="control",
            location_aliases=location_aliases,
            errors=errors,
            error_prefix=f"initial_state.fixtures[{fixture_id!r}]",
        )
        _rename_fixture_section_keys(
            fixture_state,
            section_name="parts",
            candidates=metadata.part_ids,
            kind="part",
            location_aliases=location_aliases,
            errors=errors,
            error_prefix=f"initial_state.fixtures[{fixture_id!r}]",
        )
        _rename_fixture_section_keys(
            fixture_state,
            section_name="support_sites",
            candidates=metadata.support_site_ids,
            kind="support_site",
            location_aliases=location_aliases,
            errors=errors,
            error_prefix=f"initial_state.fixtures[{fixture_id!r}]",
        )

    if not isinstance(objects_by_id, dict):
        return
    for object_id, object_state in objects_by_id.items():
        if not isinstance(object_id, str) or not isinstance(object_state, dict):
            continue
        object_state["location"] = _normalize_location_value(
            object_state.get("location"),
            object_id=object_id,
            initial_state=initial_state,
            grounding=grounding,
            fixture_metadata_by_symbol=fixture_metadata_by_symbol,
            location_aliases=location_aliases,
            errors=errors,
            context=f"initial_state.objects[{object_id!r}]",
        )


def _slot_pair_sites_from_control_id(control_id: str | None) -> tuple[str, str] | None:
    if not isinstance(control_id, str):
        return None
    if control_id.endswith("_0"):
        return ("sideL_slotL", "sideL_slotR")
    if control_id.endswith("_1"):
        return ("sideR_slotL", "sideR_slotR")
    return None


def _normalize_toaster_slot_targets(
    *,
    payload: dict[str, Any],
) -> None:
    """Pick a concrete free slot when the spec refers to a working slot-pair."""

    initial_state = payload.get("initial_state") or {}
    fixtures_by_id = initial_state.get("fixtures") or {}
    objects_by_id = initial_state.get("objects") or {}
    allowed_tool_specs = payload.get("allowed_tool_specs") or {}
    trajectory = payload.get("example_trajectory") or {}
    if not isinstance(fixtures_by_id, dict) or not isinstance(objects_by_id, dict):
        return

    active_pair_sites: tuple[str, str] | None = None
    press_lever_spec = allowed_tool_specs.get("press_lever")
    if isinstance(press_lever_spec, dict):
        for control_id in press_lever_spec.get("allowed_control_ids") or []:
            active_pair_sites = _slot_pair_sites_from_control_id(control_id)
            if active_pair_sites is not None:
                break
    if active_pair_sites is None and isinstance(trajectory, dict):
        for step in trajectory.get("steps") or []:
            if not isinstance(step, dict) or step.get("tool") != "press_lever":
                continue
            args = step.get("args") or {}
            if not isinstance(args, dict):
                continue
            active_pair_sites = _slot_pair_sites_from_control_id(args.get("control_id"))
            if active_pair_sites is not None:
                break
    if active_pair_sites is None:
        return

    active_pair_site_set = set(active_pair_sites)
    occupied_objects = [
        (object_id, object_state.get("location"))
        for object_id, object_state in objects_by_id.items()
        if isinstance(object_id, str)
        and isinstance(object_state, dict)
        and object_state.get("location") in active_pair_site_set
    ]
    if len(occupied_objects) != 1:
        return
    occupied_object_id, occupied_site = occupied_objects[0]
    if not isinstance(occupied_site, str):
        return
    destination_site = next(
        site_id for site_id in active_pair_sites if site_id != occupied_site
    )

    for fixture_state in fixtures_by_id.values():
        if not isinstance(fixture_state, dict):
            continue
        if fixture_state.get("fixture_type") != "toaster":
            continue
        support_sites = fixture_state.setdefault("support_sites", {})
        if isinstance(support_sites, dict):
            support_sites.setdefault(occupied_site, {"site_type": "support"})
            support_sites.setdefault(destination_site, {"site_type": "support"})

    def _rewrite_object_location_entry(entry: dict[str, Any]) -> None:
        if not isinstance(entry, dict):
            return
        object_id = entry.get("object_id")
        if object_id == occupied_object_id:
            if isinstance(entry.get("location"), str) and entry.get("location") != occupied_site:
                entry["location"] = occupied_site
            return
        if entry.get("location") == occupied_site:
            entry["location"] = destination_site
        if entry.get("support_id") == occupied_site:
            entry["support_id"] = destination_site

    place_on_surface_spec = allowed_tool_specs.get("place_on_surface")
    if isinstance(place_on_surface_spec, dict):
        allowed_object_ids = {
            object_id
            for object_id in place_on_surface_spec.get("allowed_object_ids") or []
            if isinstance(object_id, str)
        }
        if occupied_object_id not in allowed_object_ids and isinstance(
            place_on_surface_spec.get("allowed_support_ids"),
            list,
        ):
            place_on_surface_spec["allowed_support_ids"] = [
                destination_site if support_id == occupied_site else support_id
                for support_id in place_on_surface_spec["allowed_support_ids"]
            ]

    for condition in payload.get("goal_conditions") or []:
        _rewrite_object_location_entry(condition)
    for condition in payload.get("task_preconditions") or []:
        _rewrite_object_location_entry(condition)
    for effect in payload.get("task_effects") or []:
        if not isinstance(effect, dict):
            continue
        for requirement in effect.get("required_object_locations") or []:
            _rewrite_object_location_entry(requirement)
    if isinstance(trajectory, dict):
        for step in trajectory.get("steps") or []:
            if not isinstance(step, dict):
                continue
            args = step.get("args")
            if not isinstance(args, dict):
                continue
            _rewrite_object_location_entry(args)


def _normalize_sink_hot_water_controls(
    *,
    payload: dict[str, Any],
    fixture_metadata_by_symbol: dict[str, FixtureSimulationMetadata],
) -> None:
    """Make hot-water sink specs use the simulator's temperature control."""

    initial_state = payload.get("initial_state") or {}
    task_goal = str(payload.get("task_goal") or "").lower()
    rules_text = " ".join(
        str(rule).lower() for rule in (payload.get("extra_execution_rules") or [])
    )
    mentions_hot_water = "hot water" in task_goal or "hot water" in rules_text
    if not mentions_hot_water:
        return

    task_effects = payload.get("task_effects") or []
    trajectory = payload.get("example_trajectory") or {}
    allowed_tool_specs = payload.get("allowed_tool_specs") or {}

    def _is_hot_goal(goal: Any) -> bool:
        return str(goal).lower() in {"hot", "warm"}

    def _is_water_on_goal(goal: Any) -> bool:
        return str(goal).lower() in {
            "on",
            "open",
            "high",
            "1",
            "true",
            "hot",
            "warm",
        }

    for fixture_id, metadata in fixture_metadata_by_symbol.items():
        if metadata.fixture_type != "sink" or "handle_temp_joint" not in metadata.control_ids:
            continue
        fixture_state = (initial_state.get("fixtures") or {}).get(fixture_id)
        if not isinstance(fixture_state, dict):
            continue
        controls = fixture_state.setdefault("controls", {})
        if isinstance(controls, dict):
            controls.setdefault(
                "handle_temp_joint",
                {"control_type": "rotary_control", "state": "low"},
            )

        set_rotary_spec = allowed_tool_specs.get("set_rotary_control")
        if isinstance(set_rotary_spec, dict):
            allowed_targets = set_rotary_spec.setdefault("allowed_target_ids", [])
            if isinstance(allowed_targets, list) and fixture_id not in allowed_targets:
                allowed_targets.append(fixture_id)
            allowed_controls = set_rotary_spec.setdefault("allowed_control_ids", [])
            if isinstance(allowed_controls, list) and "handle_temp_joint" not in allowed_controls:
                allowed_controls.append("handle_temp_joint")

        for effect in task_effects:
            if not isinstance(effect, dict):
                continue
            args = effect.get("args")
            if isinstance(args, dict):
                if (
                    args.get("target_id") == fixture_id
                    and args.get("control_id") == "handle_joint"
                    and _is_hot_goal(args.get("goal"))
                ):
                    args["goal"] = "on"
            machine_path = effect.get("machine_path")
            if machine_path != [fixture_id, "water_hot"]:
                continue
            if not isinstance(args, dict):
                continue
            args["target_id"] = fixture_id
            args["control_id"] = "handle_temp_joint"
            args["goal"] = "high"

        steps = trajectory.get("steps") or []
        if not isinstance(steps, list):
            continue
        has_temp_step = any(
            isinstance(step, dict)
            and step.get("tool") == "set_rotary_control"
            and isinstance(step.get("args"), dict)
            and step["args"].get("target_id") == fixture_id
            and step["args"].get("control_id") == "handle_temp_joint"
            for step in steps
        )
        insert_index = None
        for index, step in enumerate(steps):
            if not isinstance(step, dict):
                continue
            args = step.get("args")
            if (
                step.get("tool") == "set_rotary_control"
                and isinstance(args, dict)
                and args.get("target_id") == fixture_id
                and args.get("control_id") == "handle_joint"
                and _is_hot_goal(args.get("goal"))
            ):
                args["goal"] = "on"
            if (
                step.get("tool") == "set_rotary_control"
                and isinstance(args, dict)
                and args.get("target_id") == fixture_id
                and args.get("control_id") == "handle_joint"
                and _is_water_on_goal(args.get("goal"))
            ):
                insert_index = index
                break
        if has_temp_step:
            continue
        if insert_index is None:
            continue

        source_step = steps[insert_index]
        temp_step = {
            "step": source_step.get("step"),
            "agent": source_step.get("agent"),
            "tool": "set_rotary_control",
            "args": {
                "target_id": fixture_id,
                "control_id": "handle_temp_joint",
                "goal": "high",
            },
            "reasoning": "Set the sink water temperature to hot before rinsing.",
        }
        steps.insert(insert_index, temp_step)
        for index, step in enumerate(steps):
            if isinstance(step, dict):
                step["step"] = index


def normalize_spec_payload_against_simulation(
    payload: dict[str, Any],
    *,
    task_name: str,
) -> tuple[dict[str, Any], list[str]]:
    """Rewrite simulator-facing IDs to the names exposed by RoboCasa."""

    normalized_payload = deepcopy(payload)
    reference_metadata = collect_simulation_reference_metadata(
        normalized_payload,
        task_name=task_name,
    )
    fixture_metadata_by_symbol = reference_metadata.fixture_metadata_by_symbol
    location_aliases = {
        **reference_metadata.location_aliases,
        **_build_sim_support_site_aliases(
            payload=normalized_payload,
            fixture_metadata_by_symbol=fixture_metadata_by_symbol,
            location_aliases=reference_metadata.location_aliases,
        ),
    }
    errors: list[str] = []
    _normalize_initial_state(
        payload=normalized_payload,
        fixture_metadata_by_symbol=fixture_metadata_by_symbol,
        location_aliases=location_aliases,
        errors=errors,
    )
    _normalize_allowed_tool_specs(
        payload=normalized_payload,
        fixture_metadata_by_symbol=fixture_metadata_by_symbol,
        location_aliases=location_aliases,
        errors=errors,
    )
    _normalize_goal_conditions_and_preconditions(
        payload=normalized_payload,
        fixture_metadata_by_symbol=fixture_metadata_by_symbol,
        location_aliases=location_aliases,
        errors=errors,
    )
    _normalize_task_effects_and_trajectory(
        payload=normalized_payload,
        fixture_metadata_by_symbol=fixture_metadata_by_symbol,
        location_aliases=location_aliases,
        errors=errors,
    )
    _normalize_stove_control_aliases(
        payload=normalized_payload,
        fixture_metadata_by_symbol=fixture_metadata_by_symbol,
        location_aliases=location_aliases,
    )
    _normalize_sink_hot_water_controls(
        payload=normalized_payload,
        fixture_metadata_by_symbol=fixture_metadata_by_symbol,
    )
    _normalize_toaster_slot_targets(payload=normalized_payload)
    return normalized_payload, errors


def collect_simulation_reference_metadata(
    payload: dict[str, Any],
    *,
    task_name: str,
) -> SimulationReferenceMetadata:
    """Return fixture/site lookup tables without mutating the payload."""

    source_python_module = (
        str(payload.get("source_python_module"))
        if payload.get("source_python_module")
        else None
    )
    snapshot = _load_task_simulation_snapshot(
        task_name,
        _infer_robot_count(payload),
        source_python_module,
    )
    object_aliases = _resolve_symbolic_object_aliases(
        payload=payload,
        scene=snapshot["scene"],
    )
    fixture_metadata_by_symbol = _resolve_symbolic_fixture_metadata(
        task_name=task_name,
        payload=payload,
        snapshot=snapshot,
        object_aliases=object_aliases,
        source_python_module=source_python_module,
    )
    location_aliases = _build_location_aliases(
        payload=payload,
        snapshot=snapshot,
        object_aliases=object_aliases,
    )
    return SimulationReferenceMetadata(
        fixture_metadata_by_symbol=fixture_metadata_by_symbol,
        location_aliases=location_aliases,
    )


def _collect_normalization_diffs(
    original: Any,
    normalized: Any,
    *,
    path: str = "payload",
    limit: int = 12,
) -> list[str]:
    if limit <= 0:
        return []
    if type(original) is not type(normalized):
        return [f"{path}: {type(original).__name__} -> {type(normalized).__name__}"]
    if isinstance(original, dict):
        diffs: list[str] = []
        keys = []
        for key in original.keys():
            if key not in keys:
                keys.append(key)
        for key in normalized.keys():
            if key not in keys:
                keys.append(key)
        for key in keys:
            next_path = f"{path}.{key!r}"
            if key not in original:
                diffs.append(f"{next_path}: added {normalized[key]!r}")
            elif key not in normalized:
                diffs.append(f"{next_path}: removed {original[key]!r}")
            else:
                diffs.extend(
                    _collect_normalization_diffs(
                        original[key],
                        normalized[key],
                        path=next_path,
                        limit=limit - len(diffs),
                    )
                )
            if len(diffs) >= limit:
                break
        return diffs[:limit]
    if isinstance(original, list):
        if original == normalized:
            return []
        diffs: list[str] = []
        max_len = max(len(original), len(normalized))
        for index in range(max_len):
            next_path = f"{path}[{index}]"
            if index >= len(original):
                diffs.append(f"{next_path}: added {normalized[index]!r}")
            elif index >= len(normalized):
                diffs.append(f"{next_path}: removed {original[index]!r}")
            else:
                diffs.extend(
                    _collect_normalization_diffs(
                        original[index],
                        normalized[index],
                        path=next_path,
                        limit=limit - len(diffs),
                    )
                )
            if len(diffs) >= limit:
                break
        return diffs[:limit]
    if original != normalized:
        return [f"{path}: {original!r} -> {normalized!r}"]
    return []


def collect_simulation_alignment_errors(
    payload: dict[str, Any],
    *,
    task_name: str,
) -> list[str]:
    """Return normalization/validation errors without mutating the input payload."""

    normalized_payload, errors = normalize_spec_payload_against_simulation(
        payload,
        task_name=task_name,
    )
    normalization_diffs = _collect_normalization_diffs(payload, normalized_payload)
    if normalization_diffs:
        errors.extend(
            f"simulation normalization would rewrite {diff}"
            for diff in normalization_diffs
        )
    return errors
