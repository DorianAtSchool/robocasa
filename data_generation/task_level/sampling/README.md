# Sampling Method Data Generation

Use these commands from the repository root to generate comparable raw
trajectory data under `raw/sampling_methods/` for post-hoc sampling analysis.

Set the output root first:

```bash
export ROBOCASA_TASK_LEVEL_DATA_ROOT="$PWD/data_generation/task_level/data"
```

## All Analysis Methods

This runs `base`, `random`, `verbalized`, and `high_temperature` into sibling
directories under `$ROBOCASA_TASK_LEVEL_DATA_ROOT/raw/sampling_methods/`.
Existing method directories are resumed in place.

```bash
bash scripts/generate_raw_sampling_methods.sh \
  --num-trajectories 30 \
  --verbalized-k 3 \
  --max-workers 4 \
  --max-retries 5 \
  -- --enable-validation
```

## Individual Methods

Use individual commands when you only need to resume or regenerate one method.

```bash
# Base: one trajectory per run.
python -m data_generation.task_level.generation.raw.cli \
  --tasks verified \
  --num-runs 30 \
  --random-start-location true \
  --sampling base \
  --temperature 0.6 \
  --model gemini-3-flash-preview \
  --location global \
  --thinking-level low \
  --max-workers 4 \
  --max-retries 5 \
  --enable-validation \
  --resume "$ROBOCASA_TASK_LEVEL_DATA_ROOT/raw/sampling_methods/base"

# Random: UUID prompt tag, one trajectory per run.
python -m data_generation.task_level.generation.raw.cli \
  --tasks verified \
  --num-runs 30 \
  --random-start-location true \
  --sampling random \
  --temperature 0.6 \
  --model gemini-3-flash-preview \
  --location global \
  --thinking-level low \
  --max-workers 4 \
  --max-retries 5 \
  --enable-validation \
  --resume "$ROBOCASA_TASK_LEVEL_DATA_ROOT/raw/sampling_methods/random"

# Verbalized: K trajectories per run, so num-runs = num-trajectories / K.
python -m data_generation.task_level.generation.raw.cli \
  --tasks verified \
  --num-runs 10 \
  --random-start-location true \
  --sampling verbalized \
  --verbalized-k 3 \
  --temperature 0.6 \
  --model gemini-3-flash-preview \
  --location global \
  --thinking-level low \
  --max-workers 4 \
  --max-retries 5 \
  --enable-validation \
  --resume "$ROBOCASA_TASK_LEVEL_DATA_ROOT/raw/sampling_methods/verbalized"

# High temperature: base prompt shape with temperature 1.0.
python -m data_generation.task_level.generation.raw.cli \
  --tasks verified \
  --num-runs 30 \
  --random-start-location true \
  --sampling high_temperature \
  --temperature 1.0 \
  --model gemini-3-flash-preview \
  --location global \
  --thinking-level low \
  --max-workers 4 \
  --max-retries 5 \
  --enable-validation \
  --resume "$ROBOCASA_TASK_LEVEL_DATA_ROOT/raw/sampling_methods/high_temperature"
```

If a method directory does not exist yet, replace `--resume ...` with
`--summary-path "$ROBOCASA_TASK_LEVEL_DATA_ROOT/raw/sampling_methods/<method>/summary.json"`.

## Validity Checks

After generation, check completion and saved invalid trajectory counts:

```bash
jq '{is_complete, pending_tasks, num_trajectories}' \
  "$ROBOCASA_TASK_LEVEL_DATA_ROOT/raw/sampling_methods/base/summary.json"

jq -r '.task_summaries[] | select((.pending_run_indices | length) > 0 or (.trajectory_stats.invalid_trajectories // 0) > 0) | [.composite_task, (.pending_run_indices | length), (.trajectory_stats.invalid_trajectories // 0)] | @tsv' \
  "$ROBOCASA_TASK_LEVEL_DATA_ROOT/raw/sampling_methods/base/summary.json"
```

`summary_errors.json` is historical attempt metadata. It can contain validation
errors from failed attempts even when the saved trajectory files are valid.
