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

Generate `N` symbolic two-agent trajectories for `PrepareCoffee`:

```bash
python -m data_generation.task_level.trajectory_generation \
  --composite-task PrepareCoffee \
  --n 8 \
  --output /tmp/prepare_coffee_trajectories.json \
  --sdk google-genai \
  --model gemini-3-flash-preview \
  --location global \
  --max-workers 4 \
  --max-retries 5
```

Notes:

- V1 only supports `PrepareCoffee`.
- The output `.json` stores interleaved tool-call steps for `agent_0` and `agent_1`,
  plus short explicit reasoning text per step.
- The generator now uses a single supported client path:
  `genai.Client(http_options=HttpOptions(api_version="v1"))`.
- `GOOGLE_API_KEY` and `GOOGLE_GENAI_USE_VERTEXAI` are not required for the generator.
- The smoke test in `test_google_cloud.py` now loads `.env` and explicitly passes
  `vertexai=True`, `project`, and `location`.
- The ADC principal also needs Vertex AI permission to call the publisher model.
  If you see `aiplatform.endpoints.predict` denied, grant a role such as Vertex AI User.
- Keep the shell clean while testing. Old exported Google variables can override `.env`.
