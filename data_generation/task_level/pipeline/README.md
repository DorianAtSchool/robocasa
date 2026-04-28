## Task-Level Pipeline

This package turns RoboCasa composite tasks into JSON `TaskSpec`s, validates
them statically, generates trajectories, and sweeps them through the simulator.

User-facing `TaskSpec` reference:

- `docs/sim_tools/task_spec.md`

### Phases

- `0a`: static candidate filtering
- `0b`: LLM transferability scoring
- `1`: LLM spec generation, with optional simulator-aware normalization
- `2`: static spec validation and FSM dry-run of the example trajectory
- `2.5`: LLM review of Phase 2-passing specs
- `3`: trajectory generation from approved specs
- `4`: pre-image export and simulator sweep, optionally with videos
- `5`: aggregate remaining failures

### Key Behaviors

- When `--phase1-sim-normalization` is enabled, Phase 1 does not trust
  simulator-facing IDs from the model. It normalizes fixture parts, controls,
  support sites, and selected symbolic locations against live simulator
  metadata in `sim_normalization.py`.
- Phase 2 always runs schema, referential, and symbolic FSM validation. Slow
  live-simulator alignment checks are opt-in with `--phase2-sim-alignment`;
  this is separate from `--phase1-sim-normalization`.
- Stove controls keep burner-specificity. Generic knob aliases are rewritten
  only when the spec references exactly one concrete burner for that stove.
  If multiple burners are in play, normalization does not guess.
- Guarded `communicate` task effects ignore exact message text so specs do not
  depend on one brittle string literal.
- Phase 2 validates both the JSON spec structure and the example trajectory by
  replaying it through the shared symbolic FSM.
- Phase 2 and Phase 2.5 can repair failures by regenerating specs with compact
  feedback from the validator or review step. Rejected Phase 2.5 specs do not
  continue to trajectory generation.

### Simulator Sweep

- `scripts/sweep_trajectories.py` supports `--videos` to render sweep videos.
- On macOS, sweep bootstraps `MUJOCO_GL=cgl` when the inherited value is an
  incompatible Linux backend such as `osmesa` or `egl`. This is a local
  compatibility fix for viewer-backed rendering and does not change Linux
  behavior.

### Typical Commands

Run a full batch:

```bash
python -m data_generation.task_level.pipeline.cli \
  --phase all \
  --batch batch1 \
  --workers 4 \
  --num-runs 1 \
  --videos
```

Resume an existing run root from a selected subset of phases:

```bash
python -m data_generation.task_level.pipeline.cli \
  --phase 1 3 4 5 \
  --resume data_generation/task_level/data/pipeline_runs/<run_id> \
  --batch batch1 \
  --workers 4
```

Phase ids can be listed in any order; the CLI deduplicates them and runs them
in canonical pipeline order.

Add `--phase1-sim-normalization` when you want Phase 1 to pay the extra cost of
live simulator ID normalization instead of leaving simulator grounding to later
phases.

Add `--phase2-sim-alignment` when you want Phase 2 to pay the extra cost of
live simulator reference/alignment checks. Leave it off for fast group-level
spec validation.

When optional intermediate phases are missing, downstream phases use the best
available upstream artifact automatically:

- Phase `1` falls back from Phase `0b` filtered candidates to Phase `0a`
  candidates.
- Phase `3` falls back from Phase `2.5` approved specs to Phase `2` passed
  specs, and then to Phase `1` specs when neither validation phase has run.

### Task Groups

Validation groups are recorded in
`data_generation/task_level/pipeline/task_groups.py` as `TASK_GROUPS_BY_BATCH`,
with convenience aliases `BATCH1_TASK_GROUPS`, `BATCH2_TASK_GROUPS`, and
`BATCH3_TASK_GROUPS`. The groups intentionally overlap so specs and sweeps can
be checked by behavior/fixture family instead of by a strict partition.

Verified TaskSpecs are now stored canonically as one flat JSON file per task
slug under `data_generation/task_level/tasks/specs/verified/`. Group
membership is preserved separately:

- programmatically in `data_generation/task_level/pipeline/task_groups.py`
- as checked-in documentation in
  `data_generation/task_level/tasks/specs/verified/README.md`

Print the available groups:

```bash
python - <<'PY'
from data_generation.task_level.pipeline.task_groups import TASK_GROUPS_BY_BATCH
for batch, groups in TASK_GROUPS_BY_BATCH.items():
    print(f"[{batch}]")
    for name, tasks in groups.items():
        print(name, " ".join(tasks))
PY
```

Reverse-lookup the groups for one canonical task slug:

```bash
python - <<'PY'
from data_generation.task_level.pipeline.task_groups import groups_for_task
print(groups_for_task("portionhotdogs", batch="batch1"))
PY
```

Run fast Phase 2 validation for one or more groups:

```bash
BATCH=batch1
GROUPS="bowls plates"
TASKS=$(python - <<PY
from data_generation.task_level.pipeline.task_groups import tasks_for_groups
print(" ".join(tasks_for_groups("$GROUPS", batch="$BATCH")))
PY
)

python -m data_generation.task_level.pipeline.cli \
  --resume "$RUN" \
  --phase 2 \
  --phase2-repair-retries 0 \
  --tasks $TASKS
```

Generate trajectories and sweep videos for one or more groups:

```bash
BATCH=batch1
GROUPS="bowls plates"
TASKS=$(python - <<PY
from data_generation.task_level.pipeline.task_groups import tasks_for_groups
print(" ".join(tasks_for_groups("$GROUPS", batch="$BATCH")))
PY
)

python -m data_generation.task_level.pipeline.cli \
  --resume "$RUN" \
  --phase 3 4 \
  --workers 4 \
  --num-runs 1 \
  --videos \
  --tasks $TASKS
```
