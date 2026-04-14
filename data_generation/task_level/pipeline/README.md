## Task-Level Pipeline

This package turns RoboCasa composite tasks into JSON `TaskSpec`s, validates
them statically, generates trajectories, and sweeps them through the simulator.

### Phases

- `0a`: static candidate filtering
- `0b`: LLM transferability scoring
- `1`: LLM spec generation plus simulator-aware normalization
- `2`: static spec validation and FSM dry-run of the example trajectory
- `2.5`: LLM review of Phase 2-passing specs
- `3`: trajectory generation from approved specs
- `4`: pre-image export and simulator sweep, optionally with videos
- `5`: aggregate remaining failures

### Key Behaviors

- Phase 1 does not trust simulator-facing IDs from the model. It normalizes
  fixture parts, controls, support sites, and selected symbolic locations
  against live simulator metadata in `sim_normalization.py`.
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

Resume an existing run root from Phase 2:

```bash
python -m data_generation.task_level.pipeline.cli \
  --phase 2 \
  --resume data_generation/task_level/data/pipeline_runs/<run_id> \
  --batch batch1 \
  --workers 4
```
