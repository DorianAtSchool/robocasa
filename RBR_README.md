Set up a local `.venv` and the in-repo `robosuite` dependency before running the examples:

```bash
uv venv .venv --python 3.11
source .venv/bin/activate
git submodule update --init --recursive
uv pip install -e ./robosuite
uv pip install -e .
```

If you are cloning the repo for the first time, you can also use `git clone --recurse-submodules ...` to fetch `robosuite/` immediately.

Run random policy rollout:

```bash
python policies/random.py
```

This writes a rollout video to `test.mp4` at the repo root.

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

Authenticate with Application Default Credentials:

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
  --sampling base \
  --model gemini-3.1-flash-lite-preview \
  --location global \
  --thinking-level low \
  --max-workers 4 \
  --max-retries 5 \
  --enable-validation
```

For Gemini 3 models, you can optionally tune reasoning depth with
`--thinking-level minimal|low|medium|high`.

When you pass multiple tasks with `--tasks`, the generator writes:

```text
data/raw/requests/{timestamp}/
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
  --sampling verbalized \
  --verbalized-k 3 \
  --model gemini-3-flash-preview \
  --location global \
  --thinking-level medium \
  --max-workers 10 \
  --max-retries 2 \
  --batch-processing \
  --batch-gcs-prefix gs://your-bucket/robocasa-batch \
  --resume data_generation/task_level/data/gemini-3-flash-preview/requests/{timestamp}
```

For a single-task run, pass that task directory to `--resume`. For a multi-task
request, pass the request directory. Resume reuses the existing directory in place,
skips completed runs, and retries only the pending run indices.


### Adding Images via Post-Processing of Raw Data

After the raw data is generated via LLM, run post-processing to add default multi-view
`get_image` observation steps and deterministic `image_paths` fields in a copied
dataset tree under `data/w_images/`. The output path mirrors the source tree after
`data/raw/`, so
`data/raw/requests/{timestamp}/{task}/summary.json` becomes
`data/w_images/requests/{timestamp}/{task}/summary.json`. The source dataset stays unchanged:

```bash
python -m data_generation.task_level.generation.image.cli \
  --dataset data_generation/task_level/data/raw/requests/{timestamp}/{task}/summary.json
```

Default post-processing behavior:
- Add one initial `get_image(views=[top_view, room_view, map])` step before any task action.
- Wrap each navigation action with one `get_image` step before and after using
  `agentview_center`, `agentview_left`, and `agentview_right`.
- Wrap each non-navigation action with one `get_image` step before and after using
  `wrist` and `agentview_center`.
- Store the rendered artifacts for each inserted observation step in `image_paths`,
  ordered to match the requested `views`.

To post-process every task summary inside one request directory, run directly in CLI:

```bash
for summary in data_generation/task_level/data/raw/requests/{timestamp}/*/summary.json; do
  python -m data_generation.task_level.generation.image.cli \
    --dataset "$summary"
done
```

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
  locations before prompt generation and validation.

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
