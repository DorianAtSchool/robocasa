"""Shared dataclasses for pipeline artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


TASK_TYPE_BATCH_BY_ACTIVITY: dict[str, str] = {
    # batch1: countertop prep / station setup / object arrangement
    "arranging_buffet": "batch1",
    "arranging_condiments": "batch1",
    "brewing": "batch1",
    "chopping_food": "batch1",
    "chopping_vegetables": "batch1",
    "clearing_table": "batch1",
    "filling_serving_dishes": "batch1",
    "garnishing_dishes": "batch1",
    "making_juice": "batch1",
    "making_salads": "batch1",
    "measuring_ingredients": "batch1",
    "meat_preparation": "batch1",
    "mixing_and_blending": "batch1",
    "mixing_ingredients": "batch1",
    "packing_lunches": "batch1",
    "plating_food": "batch1",
    "portioning_meals": "batch1",
    "preparing_marinade": "batch1",
    "preparing_sandwiches": "batch1",
    "seasoning_food": "batch1",
    "serving_beverages": "batch1",
    "serving_food": "batch1",
    "setting_the_table": "batch1",
    "slicing_meat": "batch1",
    "snack_preparation": "batch1",
    "sorting_ingredients": "batch1",
    # batch2: appliance / sink / heat / control-heavy tasks
    "adding_ice_to_beverages": "batch2",
    "boiling": "batch2",
    "broiling_fish": "batch2",
    "cleaning_appliances": "batch2",
    "cleaning_sink": "batch2",
    "defrosting_food": "batch2",
    "frying": "batch2",
    "making_smoothies": "batch2",
    "making_tea": "batch2",
    "microwaving_food": "batch2",
    "preparing_hot_chocolate": "batch2",
    "reheating_food": "batch2",
    "sanitizing_cutting_board": "batch2",
    "sanitizing_surface": "batch2",
    "sauteing_vegetables": "batch2",
    "simmering_sauces": "batch2",
    "slow_cooking": "batch2",
    "steaming_food": "batch2",
    "steaming_vegetables": "batch2",
    "toasting_bread": "batch2",
    "washing_dishes": "batch2",
    "washing_fruits_and_vegetables": "batch2",
    # batch3: articulated storage / appliance interior / rack-drawer tasks
    "arranging_cabinets": "batch3",
    "baking": "batch3",
    "loading_dishwasher": "batch3",
    "loading_fridge": "batch3",
    "making_toast": "batch3",
    "managing_freezer_space": "batch3",
    "organizing_dishes_and_containers": "batch3",
    "organizing_recycling": "batch3",
    "organizing_utensils": "batch3",
    "restocking_supplies": "batch3",
    "storing_leftovers": "batch3",
    "tidying_cabinets_and_drawers": "batch3",
}


@dataclass
class ObjConfig:
    """Parsed object configuration from _get_obj_cfgs()."""

    name: str
    obj_groups: str | list[str] | None = None
    obj_groups_is_dynamic: bool = False
    is_distractor: bool = False
    placement_fixture_attr: str | None = None
    has_try_to_place_in: bool = False
    try_to_place_in: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ObjConfig:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class FixtureRef:
    """Parsed fixture reference from _setup_kitchen_references()."""

    name: str
    fixture_type: str | None = None
    has_hinged_parts: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FixtureRef:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class TaskAnalysis:
    """Result of static analysis on one composite task source file."""

    task_name: str
    module_path: str
    file_path: str
    activity: str
    obj_configs: list[ObjConfig] = field(default_factory=list)
    fixture_refs: list[FixtureRef] = field(default_factory=list)
    has_hinged_parts: bool = False
    exclusion_reasons: list[str] = field(default_factory=list)

    @property
    def is_candidate(self) -> bool:
        return len(self.exclusion_reasons) == 0

    @property
    def active_fixture_refs(self) -> list[FixtureRef]:
        """Fixture refs that participate in task actions (not passive anchors).

        Stools and similar seating references are registered for spatial
        anchoring but are not interacted with. We exclude them from the
        complexity count so batch classification reflects task complexity.
        """
        passive_name_patterns = ("stool",)
        return [
            ref
            for ref in self.fixture_refs
            if not any(ref.name.startswith(pattern) for pattern in passive_name_patterns)
        ]

    @property
    def batch(self) -> str:
        """Classify into task-type batches, with a complexity fallback."""
        activity_key = self.activity.strip().lower()
        task_type_batch = TASK_TYPE_BATCH_BY_ACTIVITY.get(activity_key)
        if task_type_batch is not None:
            return task_type_batch
        if self.has_hinged_parts:
            return "batch3"
        if len(self.active_fixture_refs) > 1:
            return "batch2"
        return "batch1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_name": self.task_name,
            "module_path": self.module_path,
            "file_path": self.file_path,
            "activity": self.activity,
            "batch": self.batch,
            "obj_configs": [c.to_dict() for c in self.obj_configs],
            "fixture_refs": [f.to_dict() for f in self.fixture_refs],
            "has_hinged_parts": self.has_hinged_parts,
            "exclusion_reasons": self.exclusion_reasons,
            "is_candidate": self.is_candidate,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskAnalysis:
        return cls(
            task_name=data["task_name"],
            module_path=data["module_path"],
            file_path=data["file_path"],
            activity=data["activity"],
            obj_configs=[ObjConfig.from_dict(c) for c in data.get("obj_configs", [])],
            fixture_refs=[FixtureRef.from_dict(f) for f in data.get("fixture_refs", [])],
            has_hinged_parts=data.get("has_hinged_parts", False),
            exclusion_reasons=data.get("exclusion_reasons", []),
        )
