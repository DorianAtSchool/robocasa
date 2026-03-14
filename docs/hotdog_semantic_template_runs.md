# HotDog Semantic Template Runs

This document explains the batch script for running `HotDogSetup` across
multiple layouts and how the semantic plan template is grounded at runtime.

## Batch Script

Use:

```bash
bash experiments/run_hotdog_5layouts.sh
```

By default, the script runs `HotDogSetup` on these validated layout / style
pairs:

- `11 / 34`
- `24 / 7`
- `56 / 42`
- `8 / 12`
- `15 / 20`

Outputs are written to:

```text
tmp/hotdog_5layouts/
```

with one folder per scene:

- `tmp/hotdog_5layouts/layout11_style34/`
- `tmp/hotdog_5layouts/layout24_style7/`
- `tmp/hotdog_5layouts/layout56_style42/`
- `tmp/hotdog_5layouts/layout8_style12/`
- `tmp/hotdog_5layouts/layout15_style20/`

Each run folder contains:

- `plan.json`
- `metadata.json`
- `room_view.mp4`
- `top_view.mp4`
- robot agent-view videos
- robot wrist / `eye_in_hand` videos
- `frames/`

## Environment Variables

The script can be adjusted without editing it:

- `PYTHON_BIN`
- `MUJOCO_GL_BACKEND`
- `SEED`
- `WIDTH`
- `HEIGHT`
- `FPS`
- `OUTPUT_ROOT`

Example:

```bash
MUJOCO_GL_BACKEND=egl WIDTH=640 HEIGHT=360 OUTPUT_ROOT=/tmp/hotdog_batch \
  bash experiments/run_hotdog_5layouts.sh
```

## How The Semantic Template Works

The hotdog demo is no longer stored as a fully grounded scene-specific plan.
Instead, the executor has two stages in
[sim_tool_executor.py](/home/dorian/Projects/robocasa/robocasa/utils/sim_tool_executor.py):

1. `build_demo_plan_template("cooperative_hotdog_setup")`
2. `ground_plan_template(template)`

The template uses semantic references such as:

```json
{
  "$ref": "source_fixture",
  "object_id": "hotdog_bun"
}
```

and:

```json
{
  "$ref": "object_anchor_fixture",
  "object_id": "plate",
  "preferred_fixture_types": ["dining_counter", "island", "counter_non_dining"],
  "require_placeable": true
}
```

Those references are resolved against the live scene at runtime.

The grounded hotdog plan now makes the robot workspace explicit for support
object placement by adding `anchor_fixture_id` to `place_on_object(...)` steps.

## What Gets Grounded

For `HotDogSetup`, the runtime grounding step determines:

- where the `hotdog_bun` currently is
- which fridge fixture should be used for `sausage`
- where the `condiment` currently is
- which placeable fixture should be used as the assembly area near the `plate`

That means the same semantic template can ground differently across layouts.

Example:

- on one layout, the plate may resolve to `dining_dining_group`
- on another layout, the plate may be nearest to a `stool`, and the executor
  will instead choose a nearby placeable anchor like `island_island_group_1`

## Why This Scales Better

A saved grounded plan like `tmp/sim_tool_hotdog_demo/plan.json` is tied to one
scene because it contains literal fixture ids.

A semantic template scales better because it stores task intent rather than
scene ids:

- `source_fixture(hotdog_bun)`
- `source_fixture(sausage, preferred_type=fridge)`
- `object_anchor_fixture(plate, preferred_types=[...])`

The executor then grounds those references into the actual fixture ids for the
current realized layout.

## What `plan.json` Contains

The `plan.json` written into each output directory is the grounded plan that was
actually executed for that layout. It is not the unresolved semantic template.

For `place_on_object(...)`, that grounded plan includes:

- `support_object_id`
- `anchor_fixture_id`

This means the executor no longer has to re-guess the robot workspace for that
step if the grounded plan already specifies it.

If you want the unresolved template itself, use Python:

```python
from robocasa.utils.sim_tool_executor import SimToolExecutor

executor = SimToolExecutor(task_name="HotDogSetup", robots=2, layout=11, style=34, seed=42)
template = executor.build_demo_plan_template("cooperative_hotdog_setup")
grounded = executor.build_demo_plan("cooperative_hotdog_setup")
executor.close()
```

## Current Scope

This semantic-template path is implemented for `cooperative_hotdog_setup`.
The same pattern can be extended to other tasks by:

- authoring a task-level template with semantic refs
- grounding those refs against live scene objects and fixtures
- executing the grounded plan normally through the existing tool executor
