# pi05 Backend

This backend runs the RoboCasa leaderboard pi0.5 checkpoint through the shared
`model_evals/common/` evaluator.

Leaderboard metadata:

```text
Model: pi0.5
Policy family: openpi
Config: pi05_pretrain_human300
Checkpoint:
  https://huggingface.co/robocasa/robocasa365_checkpoints/tree/main/pi05_pretrain_human300/multitask_learning/75000
Code:
  https://github.com/robocasa-benchmark/openpi
```

## Setup

Unlike the GWP backend, the pi0.5 runtime is not vendored into this repository.
GWP only needed a small `world_action_model` package, which we keep under
`model_evals/backends/gwp/src`. pi0.5 depends on the RoboCasa openpi fork policy
server, configs, model code, and transforms, so we keep that external checkout
under an ignored repo-local folder:

```bash
git clone https://github.com/robocasa-benchmark/openpi external/openpi
cd external/openpi

uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e ../..
uv pip install -e .
uv pip install -e packages/openpi-client/
uv pip install chex==0.1.89
```

The pi05 server does not need a conda environment. Using a repo-local `uv`
environment keeps the OpenPI/JAX dependency stack separate from the RoboCasa
simulation environment. The `uv pip install -e ../..` step installs this local
RoboCasa checkout into the server environment; OpenPI needs that because its
RoboCasa policy config imports `robocasa.macros` and dataset registry modules
when loading `pi05_pretrain_human300`. That is also what provides dependencies
such as `mujoco==3.3.1`.

`chex==0.1.89` is included because this OpenPI checkout imports `chex` from the
tokenizer code and pins that version in `uv.lock`.

`external/` is ignored by git. You may still override the checkout path with
`OPENPI_ROOT=/path/to/openpi`, but the default expected path is:

```text
/home/dorian/Projects/robocasa/external/openpi
```

Download the checkpoint into this backend, or set `CHECKPOINT` directly:

```bash
hf download robocasa/robocasa365_checkpoints \
  --repo-type model \
  --include "pi05_pretrain_human300/multitask_learning/75000/**" \
  --local-dir model_evals/backends/pi05/ckpts
```

## Run

Terminal 1: start the pi05 policy server from the OpenPI `uv` environment.

```bash
cd /home/dorian/Projects/robocasa
source external/openpi/.venv/bin/activate
bash model_evals/backends/pi05/server.sh 0 \
  "$PWD/model_evals/backends/pi05/ckpts/pi05_pretrain_human300/multitask_learning/75000"
```

Terminal 2: run RoboCasa simulation from the environment that already has
RoboCasa, robosuite, and MuJoCo working.

The pi05 policy returns 50 actions per websocket request by default. The client
uses only the first `REPLAN_STEPS` actions from each chunk, matching the official
OpenPI RoboCasa eval pattern.

```bash
conda activate robocasa
BACKEND=pi05 \
NUM_TRIALS=5 \
BASE_LOG_DIR="$PWD/model_evals/backends/pi05/logs/robot_interference/composite_seen_5trials" \
bash model_evals/scripts/run_composite_seen_conditions.sh
```

## Troubleshooting

If the server fails with `ModuleNotFoundError: No module named mujoco`, the
OpenPI `uv` environment is missing the local RoboCasa install. Run:

```bash
cd /home/dorian/Projects/robocasa/external/openpi
source .venv/bin/activate
uv pip install -e ../..
```


If the server restores the model but fails while downloading
`gs://big_vision/paligemma_tokenizer.model` with a protobuf or `gcsfs` import
error, make sure the OpenPI lockfile protobuf version is installed:

```bash
cd /home/dorian/Projects/robocasa/external/openpi
source .venv/bin/activate
uv pip install protobuf==4.25.8 google-api-core==2.24.2 gcsfs==2025.3.0 fsspec==2025.3.0
```

You can verify the protobuf version with:

```bash
python -c "import google.protobuf; print(google.protobuf.__version__)"
```


If the server restores the model and tokenizer but then looks for local training
stats under `datasets/v1.0/.../lerobot/meta/stats.json`, it means OpenPI fell
back to training-dataset normalization stats instead of using the checkpoint
assets. `server.sh` sets `OPENPI_SKIP_LOCAL_NORM_STATS_FALLBACK=1`, and the local
ignored OpenPI checkout is patched so serving loads
`<checkpoint>/assets/norm_stats.json` through `policy_config.py`.

`pi05/inference_client.py` follows the official openpi RoboCasa eval input
format: 224x224 padded images, state order
`ee_pos, ee_rot, base_pos, base_rot, gripper`, and actions passed directly to
RoboCasa `convert_action`.
