Set up a local `.venv` and the in-repo `robosuite` dependency before running the examples using [uv](https://docs.astral.sh/uv/):

```bash
uv venv .venv --python 3.11
source .venv/bin/activate
git submodule update --init --recursive
uv pip install -e ./robosuite
uv pip install -e .
```

If you are cloning the repo for the first time, you can also use `git clone --recurse-submodules ...` to fetch `robosuite/` immediately.

## Vertex AI trajectory generation

The task-level generation code lives under `data_generation/task_level/generation`
and is isolated from the core RoboCasa task definitions:
- `data_generation.task_level.generation.raw`: raw trajectory generation
- `data_generation.task_level.generation.image`: post-processing that inserts canonical image observation steps

The raw-generation CLI entrypoint is `data_generation.task_level.generation.raw.cli`.

Install the required Google SDK into your active environment:

```bash
uv pip install google-genai
```

If you plan to use Vertex batch mode, install the GCS client too:

```bash
uv pip install google-cloud-storage
```

Create a repo-root `.env` file:

```bash
cat > .env <<'EOF'
GOOGLE_CLOUD_PROJECT=your-gcp-project
GOOGLE_CLOUD_LOCATION=global
# Optional on headless machines if you want a service account instead of
# `gcloud auth application-default login`:
# GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/service-account.json
EOF
```

The generator automatically loads the repo-root `.env` before CLI parsing. Shell
environment variables still win if you already exported a value manually.

First install [gcloud](https://docs.cloud.google.com/sdk/docs/install-sdk#linux), then authenticate with Application Default Credentials:

```bash
gcloud init
gcloud auth application-default login
```

Test the client wiring before you generate trajectories:

```bash
python test_google_cloud.py
```

Generate validated symbolic two-agent trajectories for `PrepareCoffee` with base sampling:

```bash
python -m data_generation.task_level.generation.raw.cli \
  --tasks PrepareCoffee \
  --num-runs 2 \
  --random-start-location true \
  --sampling base \
  --model gemini-3.1-flash-lite-preview \
  --location global \
  --thinking-level low \
  --max-workers 4 \
  --max-retries 5 \
  --enable-validation
```

Generate verbalized samples, where each run asks the model for multiple full
trajectories plus a probability label for each one:

```bash
python -m data_generation.task_level.generation.raw.cli \
  --tasks PrepareCoffee \
  --num-runs 2 \
  --random-start-location true \
  --sampling verbalized \
  --verbalized-k 3 \
  --model gemini-3.1-flash-lite-preview \
  --location global \
  --thinking-level low \
  --max-workers 4 \
  --max-retries 5 \
  --enable-validation
```

Generate both supported tasks with shared runtime settings. `--num-runs` applies
to each task, so the example below runs 10 model calls total:

```bash
python -m data_generation.task_level.generation.raw.cli \
  --tasks PrepareCoffee HotDogSetup \
  --num-runs 5 \
  --random-start-location true \
  --paralleize-tasks \
  --sampling base \
  --model gemini-3.1-flash-lite-preview \
  --location global \
  --thinking-level low \
  --max-workers 4 \
  --max-retries 5 \
  --enable-validation
```

Use `--paralleize-tasks` to run the selected tasks concurrently. `--max-workers`
still controls the per-task run workers inside each task.

Raw generation randomizes each agent's initial symbolic fixture location by
default. Use `--random-start-location false` to keep the canonical task
template positions instead (note that this maintains the same start position for N verbalized samples of a single run, if verablized sampling is enabled). When randomization is enabled, agents are sampled
independently, so some runs may start them at the same fixture and others may
start them at different fixtures.

For Gemini 3 models, you can optionally tune reasoning depth with
`--thinking-level minimal|low|medium|high`.

When you pass multiple tasks with `--tasks`, the generator writes:

```text
data/raw/{timestamp}/
├── summary.json
├── cost_summary.json
├── summary_errors.json
├── {task_a}/
│   ├── summary.json
│   ├── cost_summary.json
│   ├── summary_errors.json
│   ├── trajectories/
│   ├── prompts/
│   └── outputs/
├── {task_b}/
│   ├── summary.json
│   ├── cost_summary.json
│   ├── summary_errors.json
│   ├── trajectories/
│   ├── prompts/
│   └── outputs/
└── ...
    ├── summary.json
    ├── cost_summary.json
    ├── summary_errors.json
    ├── trajectories/
    ├── prompts/
    └── outputs/
```
### Batch trajectory generation

Trajectory generation supports a Vertex AI batch mode for large offline sweeps. It will be 50% cheaper, but significantly slower--as much as 10x slower estimated from past runs.

Use `--batch-processing` when launching the task-level generator CLI.

```bash
python -m data_generation.task_level.generation.raw.cli \
  --tasks PrepareCoffee HotDogSetup \
  --num-runs 4 \
  --random-start-location true \
  --paralleize-tasks \
  --sampling verbalized \
  --verbalized-k 3 \
  --model gemini-3-flash-preview \
  --location global \
  --thinking-level medium \
  --max-workers 10 \
  --max-retries 2 \
  --batch-processing \
  --batch-gcs-prefix gs://your-bucket/robocasa-batch
```

If a request stops with some runs still incomplete, do not start from scratch. The
generator now supports in-place resume for both batch and on-demand runs:

```bash
python -m data_generation.task_level.generation.raw.cli \
  --tasks PrepareCoffee HotDogSetup \
  --num-runs 4 \
  --paralleize-tasks \
  --sampling verbalized \
  --verbalized-k 3 \
  --model gemini-3-flash-preview \
  --location global \
  --thinking-level medium \
  --max-workers 10 \
  --max-retries 2 \
  --batch-processing \
  --batch-gcs-prefix gs://your-bucket/robocasa-batch \
  --resume data_generation/task_level/data/raw/{timestamp}
```

For a single-task run, pass that task directory to `--resume`. For a multi-task
run, pass the timestamped run directory. Resume reuses the existing directory in
place, skips completed runs, and retries only the pending run indices.


### Adding Images via Post-Processing of Raw Data

After the raw data is generated via LLM, run post-processing to add default multi-view
`get_image` observation steps and deterministic `image_paths` fields in a copied
dataset tree under `data/image/`. The output path mirrors the source tree after
`data/raw/`, so
`data/raw/{timestamp}/{task}/summary.json` becomes
`data/image/{timestamp}/{task}/summary.json`. The source dataset stays unchanged:

```bash
python -m data_generation.task_level.generation.image.cli \
  --dataset data_generation/task_level/data/raw/{timestamp}/{task}/summary.json
```

Default post-processing behavior:
- Add one initial `get_image(views=[top_view, room_view, map])` step per agent after the opening coordination block and before the first task action.
- Wrap each navigation action with one `get_image` step before and after using
  `agentview_center`, `agentview_left`, and `agentview_right`.
- Wrap each non-navigation action with one `get_image` step before and after using
  `wrist` and `agentview_center`.
- Store the rendered artifacts for each inserted observation step in `image_paths`,
  ordered to match the requested `views`.

To post-process every task summary inside one multi-task run directory, pass the
raw run timestamp to the helper script:

```bash
bash scripts/post_process_task_level_images.sh {timestamp}
```

The wrapper expands `{timestamp}` to
`data_generation/task_level/data/raw/{timestamp}/*/summary.json` and runs the
image post-processing CLI once per task. It writes the copied post-processed
trajectories to `data_generation/task_level/data/pre_image/{timestamp}/...`.
You can pass shared CLI flags after the timestamp, for example
`--disable-progress`.

To sweep one post-processed timestamp through the simulator, use the matching
timestamp wrapper:

```bash
bash scripts/generate_and_insert_images.sh {timestamp}

# Parallelize across trajectories with 4 workers.
bash scripts/generate_and_insert_images.sh {timestamp} --workers 4
```

That wrapper reads trajectories from
`data_generation/task_level/data/pre_image/{timestamp}` and writes the rendered
outputs to `data_generation/task_level/data/image/{timestamp}`. It exits with an
error if `data_generation/task_level/data/pre_image/{timestamp}` does not exist.
Pass `--workers N` or `-j N` to parallelize trajectory execution within that
timestamped sweep. The wrapper shows a progress bar by default and suppresses
the sweep CLI's normal stdout unless you pass `--verbose`, but it still prints
the final `Done` and `Summary` lines. The wrapper owns `--workers` and
`--verbose`, then forwards any remaining arguments to
`python scripts/sweep_trajectories.py`.

For multi-GPU cluster runs, request the GPUs from your scheduler and forward
GPU allocation flags to the sweep CLI. Example: 8 workers spread evenly across
4 GPUs with EGL rendering:

```bash
srun --gpus=4 bash scripts/generate_and_insert_images.sh {timestamp} \
  --workers 8 \
  --gpu-ids 0 1 2 3 \
  --procs-per-gpu 2 2 2 2 \
  --gl-backend egl
```

If your scheduler masks GPUs per job, use the GPU ordinals visible inside the
job shell, which are usually `0..N-1`. You can verify that with
`echo $CUDA_VISIBLE_DEVICES`.

The raw generator writes artifacts next to the dataset summary:
- `trajectories/`: saved trajectory JSON
- `prompts/`: the exact prompt used for each saved trajectory
- `outputs/`: the raw successful model output text for each saved trajectory
- `summary_errors.json`: aggregated error events for that task or request
- `cost_summary.json`: costs of generation per trajectory and aggregated totals

Post-processing keeps those copied artifacts and also prepares:
- `images/`: sibling image root referenced by inserted observation steps via `image_paths`

Sampling notes:
- `--num-runs` is the number of runs, not always the number of saved trajectories.
- With multiple tasks, `--num-runs` applies to each task. For example,
  `--tasks PrepareCoffee HotDogSetup --num-runs 5` launches 10 runs total.
- `--sampling base` saves one trajectory per successful run.
- `--sampling verbalized` saves `--verbalized-k` flattened trajectories per successful run.
- Verbalized trajectories include `sampling_metadata` with the parsed probability.
- The task files may define a template agent location such as `staging_area`, but
  actual per-run agent start positions are sampled from the task's allowed fixture
  locations before prompt generation and validation. Because the agents are
  sampled independently, a run may start them at the same fixture or at
  different fixtures.

Notes:

- Supported tasks currently include `PrepareCoffee` and `HotDogSetup`.
- Batch mode supports both `--sampling base` and `--sampling verbalized`.
- With `--sampling verbalized`, each successful batch row can save multiple
  flattened trajectories from one model response.
- Validation is now disabled by default. Without `--enable-validation`, the generator
  keeps invalid trajectories, records the validation error in the output, and omits
  retry-attempt metadata from the saved JSON.
- With `--enable-validation`, invalid or duplicate trajectories are retried up to
  `--max-retries`. If some runs still fail after the retry budget is exhausted, the
  generator writes the successful runs, records the failed run indices in the task
  summary, continues to the remaining tasks, and exits non-zero at the end.
- Task summaries now include explicit run state:
  `completed_run_indices`, `failed_run_indices`, `pending_run_indices`, and
  `is_complete`. Request summaries also surface whether the overall request is
  complete.
- `--resume` works for both on-demand and batch mode. It validates that the saved
  directory matches the requested task/model/sampling configuration before reusing it.
- The output `.json` stores interleaved tool-call steps for `agent_0` and `agent_1`,
  plus short explicit reasoning text per step.
- The main output path stores a compact trajectory summary plus `cost_summary`. A
  sibling cost summary file is written to `cost_summary.json` by default and includes
  per-trajectory `generation_usage`, retry-inclusive rollup costs, average trajectory
  cost, and model pricing (`input_usd_per_million_tokens` /
  `output_usd_per_million_tokens`).
- A sibling `summary_errors.json` file records every observed error in the run,
  including retry failures and saved invalid trajectories, plus per-error counts.
- The generator prints an initial projected cost before execution starts. While
  the CLI runs, the overall progress bar shows both the accumulated observed cost
  so far and a single projected total computed from the remaining trajectories
  and the current average cost per saved trajectory. Until at least one trajectory
  finishes, that live projected value is shown as `NaN`. In batch mode, the overall
  status text also shows `trajectories=x/y` so you can see how many saved
  trajectories are complete so far.
- The generator now uses a single supported client path:
  `genai.Client(http_options=HttpOptions(api_version="v1"))` after the runtime
  configures the Vertex environment variables internally.
- `GOOGLE_API_KEY` and `GOOGLE_GENAI_USE_VERTEXAI` are not required for the generator.
- The smoke test in `test_google_cloud.py` now loads `.env` and explicitly passes
  `vertexai=True`, `project`, and `location` directly because it exercises the
  raw SDK path outside the generator runtime wrapper.
- The ADC principal also needs Vertex AI permission to call the publisher model.
  If you see `aiplatform.endpoints.predict` denied, grant a role such as Vertex AI User.
- Keep the shell clean while testing. Old exported Google variables can override `.env`.
