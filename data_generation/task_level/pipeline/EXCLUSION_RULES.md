# Phase 0a Exclusion Rules

Reference for the static filters in `phase0a.py`. Each rule excludes tasks
where the object identity cannot be determined from source code alone.

## Rules

### R1 — `sample_object()`
Task calls `self.sample_object(groups=[...])` to dynamically pick an object
category at runtime.

**Example** (`choose_measuring_cup.py`):
```python
measuring_cup_info = self.sample_object(
    groups=["measuring_cup"], obj_registries=self.obj_registries
)
measuring_cup_inst = measuring_cup_info[1]["mjcf_path"]
cfgs.append(dict(name="measuring_cup_small", obj_groups=measuring_cup_inst, ...))
```

### R2 — `get_cats_by_type()`
Task enumerates a category list from a type filter, then samples from it.

**Example** (`pack_food_by_temp.py`):
```python
hot_categories = get_cats_by_type(types=["cooked_food"], obj_registries=self.obj_registries)
selected_hot = self.rng.choice(graspable_hot_categories, size=2, replace=False)
cfgs.append(dict(name="hot0", obj_groups=selected_hot[0], ...))
```

### R3 — `rng.choice()` for object types
Task uses `self.rng.choice([...])` to pick a category. Position-only choices
(`rng.choice([-1.0, 1.0])` for direction) are NOT flagged.

**Example** (`cutting_tool_selection.py`):
```python
self.food = self.rng.choice(list(self._CUTTING_MAP.keys()))
cfgs.append(dict(name="food", obj_groups=self.food, ...))
```

### R4 — Broad `obj_groups`
Task sets `obj_groups` to a category that expands to many specific types at
runtime: `fruit`, `vegetable`, `meat`, `dairy`, `drink`, `snack`, `cereal`,
`bread`, `food`, `sweets`, `packaged_food`, `all`.

**Example** (`arrange_buffet_dessert.py`):
```python
cfgs.append(dict(name="sweet1", obj_groups="sweets", ...))
```

### R5 — Dynamic `obj_groups`
Task sets `obj_groups=self.something` (computed value, not a literal).

**Example** (`create_child_friendly_fridge.py`):
```python
if self.third_item_type == "fruit_vegetable":
    obj_groups = ("fruit", "vegetable")
else:
    obj_groups = "alcohol"
cfgs.append(dict(name="item3", obj_groups=obj_groups, ...))
```

### R6 — Abstract names with non-specific groups
Task uses generic names (`obj`, `obj1`, `obj_2`) AND `obj_groups` is also
dynamic/broad. If `name="obj"` but `obj_groups="mug"`, the task is still
deterministic and is NOT excluded — only the variable name is generic.

**Example** (`bread_setup_slicing.py`):
```python
for i in range(self.num_bread):
    cfgs.append(dict(name=f"obj_{i}", obj_groups="bread", ...))
```

## Reclaiming Excluded Tasks

| Rule | Root cause | Fix path |
|------|------------|----------|
| R1, R2, R3, R4, R5 | RNG-driven object identity | Pin the RNG seed / pin `obj_groups` in `_filter_object_cfgs_for_trajectory()` (kitchen.py) so the trajectory sees a deterministic instance |
| R6 | Cosmetic + R3/R4 underneath | Fix the underlying R3/R4; renaming is automatic |

The unifying fix: extend `_filter_object_cfgs_for_trajectory()` to pin
`obj_groups`, `exclude_obj_groups`, and the task RNG to whatever the spec
declares. This is documented in
`data_generation/task_level/tasks/specs/deferred_scene_matching/README.md`.

If implemented, this would mechanically reclaim ~80-100 of the 127 currently
excluded tasks (the R1/R2/R3/R4/R5 ones). Tasks where RNG controls topology
(e.g., `num_bread = self.rng.randint(...)` changes the object count) cannot
be reclaimed by sampling fixes alone — they would need spec-per-variation
generation.

## Current Counts (2026-04-09)

- Total composite task files scanned: 301
- Candidates after Phase 0a: 174 (batch1: 31, batch2: 73, batch3: 70)
- Excluded: 127
  - R4 (broad groups): 98
  - R5 (dynamic groups): 25
  - R3 (rng.choice for objects): 23
  - R6 (abstract names + dynamic): 18
  - R2 (get_cats_by_type): 5
  - R1 (sample_object): 4
