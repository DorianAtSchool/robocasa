"""Phase 0a: Static candidate filtering via AST analysis.

Parses each composite task .py file without importing it, extracts object
configs and fixture references, then applies exclusion rules to identify
tasks suitable for automated 2-agent TaskSpec generation.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

from .models import FixtureRef, ObjConfig, TaskAnalysis

COMPOSITE_DIR = (
    Path(__file__).resolve().parents[3]
    / "robocasa"
    / "environments"
    / "kitchen"
    / "composite"
)

# Object group categories that are too broad for deterministic spec generation.
BROAD_CATEGORIES = frozenset(
    {
        "fruit",
        "vegetable",
        "all",
        "meat",
        "dairy",
        "drink",
        "snack",
        "cereal",
        "bread",
        "food",
        "sweets",
        "packaged_food",
    }
)

# Words that indicate rng.choice is being used for position, not object type.
_POSITION_KEYWORDS = frozenset(
    {
        "slot",
        "direction",
        "pos",
        "side",
        "knob",
        "pair",
        "loc",
        "location",
        "x_pos",
        "y_pos",
        "rack",
        "index",
        "idx",
    }
)


def _class_name_to_module_path(file_path: Path) -> str:
    """Convert a file path to a dotted module path."""
    parts = file_path.relative_to(COMPOSITE_DIR.parents[3]).with_suffix("").parts
    return ".".join(parts)


def _extract_string_constant(node: ast.expr) -> str | None:
    """Extract a string value from an AST constant node."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _extract_string_or_tuple(node: ast.expr) -> str | list[str] | None:
    """Extract a string or tuple/list of strings from an AST node."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, (ast.Tuple, ast.List)):
        values = []
        for elt in node.elts:
            s = _extract_string_constant(elt)
            if s is None:
                return None
            values.append(s)
        return values if len(values) != 1 else values[0]
    return None


def _is_rng_choice_call(node: ast.expr) -> bool:
    """Check if a node is a call to self.rng.choice(...)."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if not isinstance(func, ast.Attribute) or func.attr != "choice":
        return False
    value = func.value
    return (
        isinstance(value, ast.Attribute)
        and value.attr == "rng"
        and isinstance(value.value, ast.Name)
        and value.value.id == "self"
    )


def _is_position_rng_choice(node: ast.Call, assign_target: str | None = None) -> bool:
    """Heuristic: does this rng.choice select a position/direction, not an object type?

    Checks:
    1. If the assignment target name contains position keywords.
    2. If the choice list contains only numeric values.
    """
    if assign_target:
        target_lower = assign_target.lower()
        if any(kw in target_lower for kw in _POSITION_KEYWORDS):
            return True

    if node.args:
        arg = node.args[0]
        if isinstance(arg, (ast.List, ast.Tuple)):
            all_numeric = all(
                isinstance(elt, ast.Constant) and isinstance(elt.value, (int, float))
                for elt in arg.elts
            )
            if all_numeric:
                return True
    return False


def _extract_dict_call_kwargs(node: ast.Call) -> dict[str, ast.expr]:
    """Extract keyword arguments from a dict(...) call."""
    result: dict[str, ast.expr] = {}
    for kw in node.keywords:
        if kw.arg is not None:
            result[kw.arg] = kw.value
    return result


def _is_dict_call(node: ast.expr) -> bool:
    """Check if node is a call to dict(...)."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "dict"
    )


def _extract_fixture_type(node: ast.expr) -> str | None:
    """Extract FixtureType enum name from id=FixtureType.X."""
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        if node.value.id == "FixtureType":
            return node.attr
    return None


class CompositeTaskVisitor(ast.NodeVisitor):
    """AST visitor that extracts task metadata from a composite task source file."""

    def __init__(self) -> None:
        self.class_name: str | None = None
        self.obj_configs: list[ObjConfig] = []
        self.fixture_refs: list[FixtureRef] = []
        self.has_hinged_parts: bool = False
        self.exclusion_reasons: list[str] = []

        # Track which methods we're inside.
        self._current_method: str | None = None
        # Track rng.choice calls that feed into obj_groups via instance vars.
        self._rng_choice_for_objects: bool = False

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        # Only process classes that inherit from Kitchen (or other bases).
        # We take the first class defined in the file.
        if self.class_name is None:
            for base in node.bases:
                base_name = None
                if isinstance(base, ast.Name):
                    base_name = base.id
                elif isinstance(base, ast.Attribute):
                    base_name = base.attr
                if base_name and base_name == "Kitchen":
                    self.class_name = node.name
                    break
            # If no Kitchen base found, still take the class if it looks
            # like a composite task (has _get_obj_cfgs method).
            if self.class_name is None:
                for item in ast.walk(node):
                    if (
                        isinstance(item, ast.FunctionDef)
                        and item.name == "_get_obj_cfgs"
                    ):
                        self.class_name = node.name
                        break

        if self.class_name == node.name:
            self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        prev_method = self._current_method
        self._current_method = node.name

        if node.name == "_get_obj_cfgs":
            self._visit_get_obj_cfgs(node)
        elif node.name == "_setup_kitchen_references":
            self._visit_setup_kitchen_references(node)
        elif node.name == "_setup_scene":
            self._visit_setup_scene(node)

        # Also scan ALL methods for sample_object and get_cats_by_type.
        self._scan_for_disqualifiers(node)

        self._current_method = prev_method

    def _visit_get_obj_cfgs(self, node: ast.FunctionDef) -> None:
        """Extract object configs from _get_obj_cfgs method body."""
        for child in ast.walk(node):
            # Look for cfgs.append(dict(...)) calls.
            if not isinstance(child, ast.Call):
                continue

            # Check for .append() call.
            if isinstance(child.func, ast.Attribute) and child.func.attr == "append":
                if child.args and _is_dict_call(child.args[0]):
                    self._extract_obj_config(child.args[0])

            # Also check for direct dict() in list literal: cfgs = [dict(...), ...]
            # (less common but possible).

            # Check for rng.choice inside _get_obj_cfgs (R3/R5).
            if _is_rng_choice_call(child):
                # Determine the assignment target if possible.
                # This is tricky in a walk — we check if the choice feeds
                # into obj_groups.
                if not _is_position_rng_choice(child):
                    self._rng_choice_for_objects = True

        # Also handle list literal return: return [dict(...), dict(...)]
        for child in ast.walk(node):
            if isinstance(child, ast.Return) and child.value is not None:
                if isinstance(child.value, ast.List):
                    for elt in child.value.elts:
                        if _is_dict_call(elt):
                            self._extract_obj_config(elt)

    def _extract_obj_config(self, dict_call: ast.Call) -> None:
        """Extract one ObjConfig from a dict(...) call."""
        kwargs = _extract_dict_call_kwargs(dict_call)

        name_node = kwargs.get("name")
        name = _extract_string_constant(name_node) if name_node else None
        if name is None:
            # Name might be an f-string (loop pattern) — extract the prefix.
            if name_node and isinstance(name_node, ast.JoinedStr):
                for val in name_node.values:
                    if isinstance(val, ast.Constant) and isinstance(val.value, str):
                        name = val.value.rstrip("_")
                        break
            if name is None:
                return

        obj_groups_node = kwargs.get("obj_groups")
        obj_groups: str | list[str] | None = None
        obj_groups_is_dynamic = False

        if obj_groups_node is not None:
            resolved = _extract_string_or_tuple(obj_groups_node)
            if resolved is not None:
                obj_groups = resolved
            else:
                obj_groups_is_dynamic = True

        is_distractor = name.startswith("distr") or name.startswith("distractor")

        placement_fixture_attr: str | None = None
        has_try_to_place_in = False
        try_to_place_in: str | None = None

        placement_node = kwargs.get("placement")
        if placement_node and _is_dict_call(placement_node):
            placement_kwargs = _extract_dict_call_kwargs(placement_node)
            fixture_node = placement_kwargs.get("fixture")
            if fixture_node and isinstance(fixture_node, ast.Attribute):
                placement_fixture_attr = fixture_node.attr
            ttp = placement_kwargs.get("try_to_place_in")
            if ttp is not None:
                has_try_to_place_in = True
                try_to_place_in = _extract_string_constant(ttp)

        self.obj_configs.append(
            ObjConfig(
                name=name,
                obj_groups=obj_groups,
                obj_groups_is_dynamic=obj_groups_is_dynamic,
                is_distractor=is_distractor,
                placement_fixture_attr=placement_fixture_attr,
                has_try_to_place_in=has_try_to_place_in,
                try_to_place_in=try_to_place_in,
            )
        )

    def _visit_setup_kitchen_references(self, node: ast.FunctionDef) -> None:
        """Extract fixture references from _setup_kitchen_references."""
        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            func = child.func
            if not (
                isinstance(func, ast.Attribute)
                and func.attr == "register_fixture_ref"
            ):
                continue
            if len(child.args) < 2:
                continue
            ref_name = _extract_string_constant(child.args[0])
            if ref_name is None:
                continue

            fixture_type: str | None = None
            if _is_dict_call(child.args[1]):
                fkwargs = _extract_dict_call_kwargs(child.args[1])
                id_node = fkwargs.get("id")
                if id_node:
                    fixture_type = _extract_fixture_type(id_node)
                    # Handle self.cab_id style references.
                    if fixture_type is None and isinstance(id_node, ast.Attribute):
                        attr_name = id_node.attr
                        # Try to infer type from the attribute name.
                        fixture_type = attr_name.upper().replace("_ID", "")

            self.fixture_refs.append(
                FixtureRef(
                    name=ref_name,
                    fixture_type=fixture_type,
                )
            )

            # Check for rng.choice in this method (for object type selection).
            if _is_rng_choice_call(child):
                if not _is_position_rng_choice(child):
                    self._rng_choice_for_objects = True

        # Scan assignments for rng.choice that feed into obj_groups.
        for child in ast.walk(node):
            if isinstance(child, ast.Assign) and len(child.targets) == 1:
                target = child.targets[0]
                if isinstance(target, ast.Attribute) and isinstance(
                    child.value, ast.Call
                ):
                    if _is_rng_choice_call(child.value):
                        target_name = target.attr
                        if not _is_position_rng_choice(child.value, target_name):
                            self._rng_choice_for_objects = True

    def _visit_setup_scene(self, node: ast.FunctionDef) -> None:
        """Detect hinged part operations (open_door/close_door) in _setup_scene."""
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
                if child.func.attr in ("open_door", "close_door"):
                    self.has_hinged_parts = True
                    # Try to associate with a fixture ref.
                    if isinstance(child.func.value, ast.Attribute):
                        fixture_attr = child.func.value.attr
                        for ref in self.fixture_refs:
                            if ref.name == fixture_attr:
                                ref.has_hinged_parts = True

    def _scan_for_disqualifiers(self, node: ast.FunctionDef) -> None:
        """Scan a method body for R1 (sample_object) and R2 (get_cats_by_type)."""
        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            func = child.func

            # R1: self.sample_object(...)
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "sample_object"
                and isinstance(func.value, ast.Name)
                and func.value.id == "self"
            ):
                reason = "R1: Uses sample_object() for runtime object selection"
                if reason not in self.exclusion_reasons:
                    self.exclusion_reasons.append(reason)

            # R2: get_cats_by_type(...)
            if isinstance(func, ast.Name) and func.id == "get_cats_by_type":
                reason = "R2: Uses get_cats_by_type() for category enumeration"
                if reason not in self.exclusion_reasons:
                    self.exclusion_reasons.append(reason)
            if isinstance(func, ast.Attribute) and func.attr == "get_cats_by_type":
                reason = "R2: Uses get_cats_by_type() for category enumeration"
                if reason not in self.exclusion_reasons:
                    self.exclusion_reasons.append(reason)


def _apply_exclusion_rules(analysis: TaskAnalysis) -> None:
    """Apply exclusion rules R3-R6 based on extracted metadata."""
    visitor_reasons = list(analysis.exclusion_reasons)

    for cfg in analysis.obj_configs:
        if cfg.is_distractor:
            continue

        # R4: Broad obj_groups.
        if isinstance(cfg.obj_groups, str) and cfg.obj_groups in BROAD_CATEGORIES:
            visitor_reasons.append(
                f"R4: Broad obj_groups category '{cfg.obj_groups}' on object '{cfg.name}'"
            )
        elif isinstance(cfg.obj_groups, list):
            for group in cfg.obj_groups:
                if group in BROAD_CATEGORIES:
                    visitor_reasons.append(
                        f"R4: Broad obj_groups category '{group}' on object '{cfg.name}'"
                    )

        # R5: Dynamic obj_groups.
        if cfg.obj_groups_is_dynamic:
            visitor_reasons.append(
                f"R5: Dynamic obj_groups on object '{cfg.name}'"
            )

        # R6: Abstract names, only flagged when obj_groups is also unclear.
        # A task with name="obj" but obj_groups="mug" is still fully deterministic
        # (the object type is specific; only the variable name is generic). We
        # only exclude if the abstract name coincides with dynamic or broad groups.
        if re.match(r"^obj\d*$", cfg.name):
            obj_groups_is_specific = (
                isinstance(cfg.obj_groups, str)
                and cfg.obj_groups not in BROAD_CATEGORIES
                and not cfg.obj_groups_is_dynamic
            )
            if not obj_groups_is_specific:
                visitor_reasons.append(
                    f"R6: Abstract object name '{cfg.name}' with non-specific obj_groups"
                )

    # Deduplicate.
    seen: set[str] = set()
    unique_reasons: list[str] = []
    for reason in visitor_reasons:
        if reason not in seen:
            seen.add(reason)
            unique_reasons.append(reason)
    analysis.exclusion_reasons = unique_reasons


def analyze_task_file(file_path: Path) -> TaskAnalysis | None:
    """Analyze one composite task source file via AST parsing.

    Returns None if the file does not contain a composite task class.
    """
    try:
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(file_path))
    except (SyntaxError, UnicodeDecodeError):
        return None

    visitor = CompositeTaskVisitor()
    visitor.visit(tree)

    if visitor.class_name is None:
        return None

    activity = file_path.parent.name
    module_path = _class_name_to_module_path(file_path)

    analysis = TaskAnalysis(
        task_name=visitor.class_name,
        module_path=module_path,
        file_path=str(file_path),
        activity=activity,
        obj_configs=visitor.obj_configs,
        fixture_refs=visitor.fixture_refs,
        has_hinged_parts=visitor.has_hinged_parts,
        exclusion_reasons=visitor.exclusion_reasons,
    )

    # R3: rng.choice for object types (detected by visitor).
    if visitor._rng_choice_for_objects:
        analysis.exclusion_reasons.append(
            "R3: Uses rng.choice() for object type selection"
        )

    _apply_exclusion_rules(analysis)
    return analysis


def analyze_all_tasks(
    composite_dir: Path | None = None,
) -> tuple[list[TaskAnalysis], list[TaskAnalysis]]:
    """Run Phase 0a on all composite task files.

    Returns (candidates, excluded) where candidates have no exclusion reasons.
    """
    if composite_dir is None:
        composite_dir = COMPOSITE_DIR

    candidates: list[TaskAnalysis] = []
    excluded: list[TaskAnalysis] = []

    for py_file in sorted(composite_dir.rglob("*.py")):
        if py_file.name == "__init__.py":
            continue
        analysis = analyze_task_file(py_file)
        if analysis is None:
            continue
        if analysis.is_candidate:
            candidates.append(analysis)
        else:
            excluded.append(analysis)

    return candidates, excluded


def run_phase0a(
    output_dir: Path,
    composite_dir: Path | None = None,
    dry_run: bool = False,
) -> tuple[list[TaskAnalysis], list[TaskAnalysis]]:
    """Execute Phase 0a and write results to output_dir/phase0a/."""
    candidates, excluded = analyze_all_tasks(composite_dir)

    if dry_run:
        return candidates, excluded

    phase_dir = output_dir / "phase0a"
    phase_dir.mkdir(parents=True, exist_ok=True)

    with (phase_dir / "candidates.json").open("w", encoding="utf-8") as f:
        json.dump([c.to_dict() for c in candidates], f, indent=2)

    with (phase_dir / "excluded.json").open("w", encoding="utf-8") as f:
        json.dump([e.to_dict() for e in excluded], f, indent=2)

    # Summary by batch.
    batch_counts: dict[str, int] = {}
    for c in candidates:
        batch_counts[c.batch] = batch_counts.get(c.batch, 0) + 1

    summary = {
        "total_files_scanned": len(candidates) + len(excluded),
        "candidates": len(candidates),
        "excluded": len(excluded),
        "candidates_by_batch": batch_counts,
        "exclusion_reason_counts": _count_exclusion_reasons(excluded),
    }
    with (phase_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return candidates, excluded


def _count_exclusion_reasons(
    excluded: list[TaskAnalysis],
) -> dict[str, int]:
    """Count how many tasks hit each exclusion rule."""
    counts: dict[str, int] = {}
    for task in excluded:
        rules_seen: set[str] = set()
        for reason in task.exclusion_reasons:
            rule_id = reason.split(":")[0]
            if rule_id not in rules_seen:
                rules_seen.add(rule_id)
                counts[rule_id] = counts.get(rule_id, 0) + 1
    return counts
