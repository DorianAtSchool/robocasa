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

The multi-agent task-level generator lives under `data_generation/task_level` and is
isolated from the core RoboCasa task definitions.

Install the required Google SDK into your active environment:

```bash
uv pip install google-genai
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

Generate `N` strictly validated symbolic two-agent trajectories for `PrepareCoffee`:

```bash
python -m data_generation.task_level.trajectory_generation \
  --task PrepareCoffee \
  --num-trajectories 8 \
  --output /tmp/prepare_coffee_trajectories.json \
  --sdk google-genai \
  --model gemini-3-flash-preview \
  --location global \
  --max-workers 4 \
  --max-retries 5 \
  --enable-validation
```
## Batch trajectory generation

Trajectory generation now supports a Vertex AI batch mode for large offline sweeps.

### Why use batch mode

Batch mode is useful when you want to:
- run large trajectory sweeps more cheaply than online generation
- avoid many concurrent live API calls
- be able to cancel outstanding remote work cleanly

The generated local outputs are the same as the normal path:
- trajectory JSON
- summary JSON
- cost JSON

### Enabling batch mode

Use `--batch-processing` when launching trajectory generation.

```bash
PYTHONPATH=. uv run python -m data_generation.task_level.trajectory_generation \
  --task PrepareCoffee \
  --num-trajectories 100 \
  --model gemini-2.5-flash \
  --batch-processing \
  --batch-gcs-prefix gs://YOUR_BUCKET/robocasa-batch
```

Notes:

- V1 only supports `PrepareCoffee`.
- Validation is now disabled by default. Without `--enable-validation`, the generator
  keeps invalid trajectories, records the validation error in the output, and omits
  retry-attempt metadata from the saved JSON.
- With `--enable-validation`, invalid or duplicate trajectories are retried up to
  `--max-retries` and only validated trajectories are written.
- The output `.json` stores interleaved tool-call steps for `agent_0` and `agent_1`,
  plus short explicit reasoning text per step.
- The main output path stores a compact trajectory summary plus `cost_summary`. A
  sibling sidecar file is written to `<output>_costs.json` by default and includes
  per-trajectory `generation_usage`, observed rollup costs, and model pricing
  (`input_usd_per_million_tokens` / `output_usd_per_million_tokens`).
- The generator prints a projected cost before execution starts. With validation
  disabled this is a single projected total; with `--enable-validation` it is shown
  as a best-case / worst-case range based on the retry budget.
- The generator now uses a single supported client path:
  `genai.Client(http_options=HttpOptions(api_version="v1"))`.
- `GOOGLE_API_KEY` and `GOOGLE_GENAI_USE_VERTEXAI` are not required for the generator.
- The smoke test in `test_google_cloud.py` now loads `.env` and explicitly passes
  `vertexai=True`, `project`, and `location`.
- The ADC principal also needs Vertex AI permission to call the publisher model.
  If you see `aiplatform.endpoints.predict` denied, grant a role such as Vertex AI User.
- Keep the shell clean while testing. Old exported Google variables can override `.env`.
