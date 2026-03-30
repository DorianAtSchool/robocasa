"""Adapter for external full-trajectory JSON into executor tool calls.

Symbolic IDs in trajectory steps (e.g. "bun", "serving_surface") are resolved
to concrete sim IDs (e.g. "hotdog_bun", "dining_dining_group") via:

  1. **Sim ground truth** (preferred) — uses ``env.fixture_refs`` and
     ``env.object_cfgs`` placement info exposed in the scene description.
     Combined with the trajectory's ``initial_state`` type information,
     this resolves all symbols unambiguously.

  2. **Heuristic fallback** — any symbols not resolved by ground truth
     fall through to the existing type-match / token-match / ordinal heuristics.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


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
        # Reverse maps: concrete env key → human-readable display name
        # Built during adapt() after resolution is complete.
        self._object_display_names: dict[str, str] = {}
        self._fixture_display_names: dict[str, str] = {}

    def _apply_sim_ground_truth(
        self,
        initial_state: dict[str, Any],
        grounding_symbols: dict[str, Any] | None = None,
    ) -> None:
        """Pre-populate alias caches from sim ground truth.

        Uses the trajectory's ``initial_state`` (object types and fixture
        types) combined with ``scene["object_placements"]`` and
        ``scene["fixture_refs"]`` to resolve symbolic IDs to concrete sim
        IDs with full confidence.

        ``grounding_symbols`` comes from ``trajectory["grounding_map"]["symbols"]``
        and carries richer resolver hints (e.g. ``anchor_fixture_symbol``,
        ``preferred_fixture_types``) that ``initial_state.fixtures`` doesn't have.
        """
        grounding_symbols = grounding_symbols or {}
        # object_placements: {env_obj_key: concrete_fixture_id} from env.object_cfgs
        object_placements = self.scene.get("object_placements", {})
        # fixture_refs: task-registered refs only (e.g. "coffee_machine", "cab"),
        # NOT all fixtures in the scene — just the ones the task explicitly registered
        # via register_fixture_ref() / get_fixture()
        fixture_refs = self.scene.get("fixture_refs", {})
        # scene_fixtures: ALL fixtures in the scene with positions, types, etc.
        # includes every counter, cabinet, appliance, etc. in the kitchen layout
        scene_fixtures = self.scene.get("fixtures", {})
        env_object_ids = set(self.scene.get("objects", {}).keys())

        if not object_placements and not fixture_refs:
            log.warning("No sim ground truth available; falling back to heuristics")
            return

        # --- Objects: match trajectory symbol → env.objects key by type ---
        traj_objects = initial_state.get("objects", {})
        # Build reverse map: object_type → env.objects key
        # env.objects keys are the task's obj_cfg names (e.g. "hotdog_bun")
        type_to_env_key = {}
        for env_key in env_object_ids:
            obj_info = self.scene.get("objects", {}).get(env_key, {})
            obj_type = obj_info.get("object_type", "")
            # Map both the type and the key itself
            type_to_env_key[obj_type] = env_key
            type_to_env_key[env_key] = env_key

        for symbol, obj_state in traj_objects.items():
            obj_type = obj_state.get("object_type", "")
            # Try: exact symbolic key match, then exact object_type-as-id match,
            # then generic type match, then substring match.
            resolved = None
            if symbol in env_object_ids:
                resolved = symbol
            elif obj_type in env_object_ids:
                resolved = obj_type
            elif obj_type in type_to_env_key:
                resolved = type_to_env_key[obj_type]
            else:
                for env_key in env_object_ids:
                    if obj_type in env_key or env_key in obj_type:
                        resolved = env_key
                        break

            if resolved is not None:
                self._object_aliases[symbol] = resolved
                self._resolution_log.append(ResolutionRecord(
                    entity_type="object",
                    requested_id=symbol,
                    resolved_id=resolved,
                    method="sim_ground_truth",
                    confidence=1.0,
                    reason=f"Matched object_type {obj_type!r} to env.objects[{resolved!r}]",
                ))

        # --- Fixtures: resolve via object placements and fixture refs ---
        traj_fixtures = initial_state.get("fixtures", {})

        # Build map: symbolic fixture → concrete ID by tracing object locations
        # If trajectory says object X is at fixture Y, and we resolved X to
        # env key K, then object_placements[K] gives the concrete fixture ID.
        traj_obj_locations = {}
        for symbol, obj_state in traj_objects.items():
            loc = obj_state.get("location")
            if isinstance(loc, str):
                traj_obj_locations.setdefault(loc, []).append(symbol)

        # --- Pass 1: strategies that don't depend on other fixtures ---
        for fixture_symbol, fixture_state in traj_fixtures.items():
            fixture_type = fixture_state.get("fixture_type", "")

            # Strategy 1: trace via object that lives at this fixture
            obj_symbols_here = traj_obj_locations.get(fixture_symbol, [])
            for obj_sym in obj_symbols_here:
                resolved_obj = self._object_aliases.get(obj_sym)
                if resolved_obj and resolved_obj in object_placements:
                    concrete_fixture = object_placements[resolved_obj]
                    self._fixture_aliases[fixture_symbol] = concrete_fixture
                    self._resolution_log.append(ResolutionRecord(
                        entity_type="fixture",
                        requested_id=fixture_symbol,
                        resolved_id=concrete_fixture,
                        method="sim_ground_truth",
                        confidence=1.0,
                        reason=f"Object {resolved_obj!r} placed on {concrete_fixture!r} by task config",
                    ))
                    break

            if fixture_symbol in self._fixture_aliases:
                continue

            # Strategy 2: match fixture_refs (task-registered only) by type
            for role, fxtr_id in fixture_refs.items():
                if role == fixture_type or fixture_type in role or role in fixture_type:
                    self._fixture_aliases[fixture_symbol] = fxtr_id
                    self._resolution_log.append(ResolutionRecord(
                        entity_type="fixture",
                        requested_id=fixture_symbol,
                        resolved_id=fxtr_id,
                        method="sim_ground_truth",
                        confidence=1.0,
                        reason=f"fixture_refs[{role!r}] matched fixture_type {fixture_type!r}",
                    ))
                    break

        # --- Pass 2: anchor-dependent resolution (needs other fixtures resolved first) ---
        for fixture_symbol, fixture_state in traj_fixtures.items():
            if fixture_symbol in self._fixture_aliases:
                continue

            fixture_type = fixture_state.get("fixture_type", "")

            # Strategy 3: find the counter/surface that the anchor fixture sits on
            # (e.g. "staging_surface" = the counter the coffee_machine is placed on)
            # Uses parent_fixture from scene description (containment-based, same
            # logic as env.get_fixture(ref=...)). Falls back to nearest-center.
            # anchor/preferred info lives in grounding_map.symbols, not initial_state
            gm_entry = grounding_symbols.get(fixture_symbol, {})
            anchor_symbol = gm_entry.get("anchor_fixture_symbol")
            preferred_types = gm_entry.get("preferred_fixture_types", [])
            if anchor_symbol:
                # anchor must have been resolved in pass 1
                anchor_id = self._fixture_aliases.get(anchor_symbol)
                if anchor_id:
                    anchor_info = scene_fixtures.get(anchor_id, {})
                    match_types = set(preferred_types) | {fixture_type} if preferred_types else {fixture_type}

                    # Strategy 3a: parent_fixture — the counter the anchor sits ON
                    # (computed via point_in_fixture containment in get_scene_description)
                    parent_id = anchor_info.get("parent_fixture")
                    if parent_id and parent_id in scene_fixtures:
                        parent_type = scene_fixtures[parent_id].get("fixture_type", "")
                        if parent_type in match_types:
                            self._fixture_aliases[fixture_symbol] = parent_id
                            self._resolution_log.append(ResolutionRecord(
                                entity_type="fixture",
                                requested_id=fixture_symbol,
                                resolved_id=parent_id,
                                method="sim_ground_truth",
                                confidence=1.0,
                                reason=(
                                    f"Parent {parent_type!r} of anchor "
                                    f"{anchor_symbol!r} ({anchor_id!r}) via containment"
                                ),
                            ))
                            continue

                    # Strategy 3b: fallback — nearest scene fixture of matching type
                    anchor_pos = anchor_info.get("position", [0, 0, 0])
                    best_id = None
                    best_dist = float("inf")
                    for fid, finfo in scene_fixtures.items():
                        ftype = finfo.get("fixture_type", "")
                        if ftype not in match_types:
                            continue
                        fpos = finfo.get("position", [0, 0, 0])
                        dist = ((fpos[0] - anchor_pos[0]) ** 2 + (fpos[1] - anchor_pos[1]) ** 2) ** 0.5
                        if dist < best_dist:
                            best_dist = dist
                            best_id = fid
                    if best_id is not None:
                        self._fixture_aliases[fixture_symbol] = best_id
                        self._resolution_log.append(ResolutionRecord(
                            entity_type="fixture",
                            requested_id=fixture_symbol,
                            resolved_id=best_id,
                            method="sim_ground_truth",
                            confidence=0.8,
                            reason=(
                                f"Nearest {fixture_type!r} to anchor "
                                f"{anchor_symbol!r} ({anchor_id!r}) at dist {best_dist:.3f}m "
                                f"(parent_fixture unavailable, fell back to nearest-center)"
                            ),
                        ))

        resolved_count = len([r for r in self._resolution_log if r.method == "sim_ground_truth"])
        total_symbols = len(traj_objects) + len(traj_fixtures)
        if resolved_count == total_symbols:
            log.info("Sim ground truth resolved all %d symbols", resolved_count)
        else:
            log.warning(
                "Sim ground truth resolved %d/%d symbols; remaining will use heuristics",
                resolved_count, total_symbols,
            )

    def adapt(
        self,
        trajectory: dict[str, Any],
        output_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        """Return a normalized executor-facing trajectory."""
        trajectory = deepcopy(trajectory)

        initial_state = trajectory.get("initial_state") or {}
        grounding_symbols = (trajectory.get("grounding_map") or {}).get("symbols") or {}
        self._apply_sim_ground_truth(initial_state, grounding_symbols)
        resolved_initial_state = self._adapt_initial_state(initial_state)

        # Build display name maps: env key → human-readable name
        # Objects: use object_type from scene (sourced from info.cat)
        for env_key, obj_info in self.scene.get("objects", {}).items():
            self._object_display_names[env_key] = obj_info.get("object_type", env_key)
        # Also map from symbolic names to display names via aliases
        for symbol, env_key in self._object_aliases.items():
            if env_key not in self._object_display_names:
                self._object_display_names[env_key] = symbol

        # Fixtures: use symbolic name from trajectory as display name,
        # fall back to fixture_type from scene
        for symbol, env_key in self._fixture_aliases.items():
            self._fixture_display_names[env_key] = symbol
        for env_key, fxtr_info in self.scene.get("fixtures", {}).items():
            if env_key not in self._fixture_display_names:
                self._fixture_display_names[env_key] = fxtr_info.get("fixture_type", env_key)

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
            "display_names": {
                "objects": dict(self._object_display_names),
                "fixtures": dict(self._fixture_display_names),
            },
        }

    def execute(
        self,
        trajectory: dict[str, Any],
        output_dir: str | Path | None = None,
        fps: int = 2,
        skip_videos: bool = False,
        save_debug_frames: bool = False,
    ) -> dict[str, Any]:
        """Adapt, load initial state, then execute with frames and video.

        Delegates to ``executor.run_tool_plan()`` so that before/after frames
        and per-camera MP4 videos are generated automatically.
        """
        adapted = self.adapt(trajectory, output_dir=output_dir)

        # Save pre-initial-state frames (before doors are closed / objects moved).
        # Useful for debugging: confirms objects are spawned correctly by the sim
        # even if the trajectory's initial_state hides them (e.g. closes cabinet).
        if output_dir is not None and save_debug_frames:
            self.executor.save_scene_frames(output_dir, prefix="pre_initial_state")

        load_summary = self.executor.load_initial_state(adapted["initial_state"])

        # In "trajectory" spawn mode, robots were just repositioned by
        # load_initial_state.  Re-save the initial map so it reflects the
        # trajectory's agent locations rather than the sim default.
        robot_spawn = getattr(self.executor, "_robot_spawn", "sim")
        if output_dir is not None and robot_spawn == "trajectory":
            self.executor.save_placement_map(
                output_dir, prefix="initial",
                clean_labels=getattr(self.executor, "_clean_map_labels", True),
            )
            if save_debug_frames:
                # Save post-initial-state rendered frames only when explicitly
                # debugging spawn / initial-state alignment.
                self.executor.save_scene_frames(output_dir, prefix="initial")

        if output_dir is not None:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            with open(output_path / "adapted_trajectory.json", "w") as f:
                json.dump(adapted, f, indent=2)

        # run_tool_plan handles rendering, frame saving, and video generation.
        plan_metadata = self.executor.run_tool_plan(
            tool_calls=adapted["tool_calls"],
            output_dir=output_dir or ".",
            fps=fps,
            skip_videos=skip_videos,
        )

        metadata = {
            "trajectory_id": adapted.get("trajectory_id"),
            "composite_task": adapted.get("composite_task"),
            "load_initial_state": load_summary,
            "resolution_log": adapted["resolution_log"],
            **plan_metadata,
        }

        if output_dir is not None:
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
                parts[self._normalize_part_id(part_id, part_state)] = deepcopy(
                    part_state
                )
            if parts:
                resolved_fixture_state["parts"] = parts
            resolved_fixtures[resolved_fixture_id] = resolved_fixture_state

        for requested_fixture_id, machine_cfg in machine_state.items():
            # machine_state keys may be task-state namespaces (e.g.
            # "hot_dog_setup") rather than fixture references.  Only attempt
            # resolution when the key looks like a known fixture or alias.
            if (
                requested_fixture_id in self.scene.get("fixtures", {})
                or requested_fixture_id in self._fixture_aliases
                or requested_fixture_id in fixture_context
            ):
                resolved_fixture_id = self._resolve_fixture_id(
                    requested_fixture_id,
                    requested_fixture_state=fixture_context.get(requested_fixture_id),
                )
            else:
                resolved_fixture_id = requested_fixture_id
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
                # Location can be a fixture ("mug_source_fixture") or another
                # object ("ingredient_bowl" for slices inside a bowl).  Check
                # object aliases first to avoid sending object names through
                # fixture resolution, which would fall back to a random fixture.
                if location in self._object_aliases or location in object_context:
                    resolved_state["location"] = self._resolve_object_id(
                        location,
                        requested_object_state=object_context.get(location),
                    )
                else:
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
            views = args.get("views") if isinstance(args.get("views"), list) else []
            if isinstance(image_paths, list) and image_paths:
                args["image_paths"] = [
                    str(
                        self._resolve_image_output_path(
                            image_path,
                            output_dir,
                            view_name=views[idx] if idx < len(views) else None,
                        )
                    )
                    for idx, image_path in enumerate(image_paths)
                ]
            else:
                image_path = args.pop("image_path", None)
                if not isinstance(image_path, str):
                    image_path = step.get("image_path")
                if isinstance(image_path, str):
                    args["image_paths"] = [
                        str(
                            self._resolve_image_output_path(
                                image_path,
                                output_dir,
                                view_name=views[0] if views else None,
                            )
                        )
                    ]
            tool_name = "get_image"

        args = self._resolve_step_args(
            tool_name,
            args,
            resolved_initial_state=resolved_initial_state,
        )

        # Build display_args: human-readable names for VLM consumption
        display_args = {}
        for arg_name, value in args.items():
            if not isinstance(value, str):
                continue
            if arg_name in self._OBJECT_ARG_NAMES:
                display_args[arg_name] = self._object_display_names.get(value, value)
            elif arg_name in self._FIXTURE_ARG_NAMES or arg_name == "receptacle_id":
                display_args[arg_name] = self._fixture_display_names.get(value, value)

        result = {
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
        if display_args:
            result["display_args"] = display_args
        return result

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
                        resolved_initial_state.get("fixtures", {}).get(value) or None
                    ),
                )
            elif arg_name in self._OBJECT_ARG_NAMES:
                # The value may actually be a fixture (e.g. "toaster_oven"
                # passed as reference_object_id in place_next_to).  Check
                # fixture aliases first to avoid a bad object fallback.
                if value in self._fixture_aliases or value in self.scene.get("fixtures", {}):
                    resolved_args[arg_name] = self._resolve_fixture_id(
                        value,
                        requested_fixture_state=(
                            resolved_initial_state.get("fixtures", {}).get(value) or None
                        ),
                    )
                else:
                    resolved_args[arg_name] = self._resolve_object_id(
                        value,
                        requested_object_state=(
                            resolved_initial_state.get("objects", {}).get(value) or None
                        ),
                    )
            elif arg_name == "receptacle_id":
                if (
                    value in self.scene.get("fixtures", {})
                    or value in self._fixture_aliases
                ):
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

    def _resolve_image_output_path(
        self,
        image_path: str,
        output_dir: str | Path | None,
        *,
        view_name: str | None = None,
    ) -> Path:
        path = self._resolve_output_path(image_path, output_dir)
        normalized_view = str(view_name).strip().lower() if view_name is not None else None
        if normalized_view != "map" and path.suffix.lower() == ".png":
            return path.with_suffix(".jpg")
        return path

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
            candidate_id
            for candidate_id in candidate_ids
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
    fps: int = 2,
    skip_videos: bool = False,
    save_debug_frames: bool = False,
) -> dict[str, Any]:
    """Adapt and execute one external trajectory."""
    return TrajectoryAdapter(
        executor=executor,
        allow_approximate_ids=allow_approximate_ids,
    ).execute(
        trajectory,
        output_dir=output_dir,
        fps=fps,
        skip_videos=skip_videos,
        save_debug_frames=save_debug_frames,
    )


__all__ = [
    "ResolutionRecord",
    "TrajectoryAdapter",
    "adapt_trajectory",
    "execute_trajectory",
]
