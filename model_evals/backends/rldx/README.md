# RLDX-1 Backend

This backend runs the RoboCasa leaderboard RLDX-1 checkpoint through the shared
`model_evals/common/` evaluator.

Leaderboard metadata:

```text
Model: RLDX-1
Policy family: RLDX
Code: https://github.com/RLWRLD/RLDX-1
Commit: ef05cd4ae634ff97d672d42275febbc0b92cc192
Checkpoint: https://huggingface.co/RLWRLD/RLDX-1-FT-RC365
```

## Setup

RLDX-1 uses a separate model-server environment. Keep the external checkout
under the ignored repo-local `external/` folder:

```bash
git clone https://github.com/RLWRLD/RLDX-1 external/rldx-1
cd external/rldx-1
git checkout ef05cd4ae634ff97d672d42275febbc0b92cc192
```

### RTX 5090 / Blackwell

On RTX 5090 / Blackwell (`sm_120`), use RLDX's `pixi` environment. The plain
`uv sync` path tries to build `flash-attn` without a CUDA toolkit and fails with
`nvcc was not found` / `CUDA_HOME environment variable is not set`.

```bash
cd /home/dorian/Projects/robocasa/external/rldx-1
pixi install
pixi run --environment rldx postinstall
pixi run --environment rldx python -c "import rldx; print(rldx.__version__)"
```

This pulls the CUDA 12.8 toolkit through pixi and source-builds flash-attn for
`sm_120`. The first `postinstall` can take 10-20 minutes.

### Non-Blackwell GPUs

For GPUs with available flash-attn wheels, the standard RLDX uv path should work:

```bash
cd /home/dorian/Projects/robocasa/external/rldx-1
uv sync --python 3.10
uv pip install -e .
```

The server can load the checkpoint directly from Hugging Face, or you can
download it locally and set `MODEL_PATH=/path/to/RLDX-1-FT-RC365`.


The scripts intentionally use `RLDX_HOST` instead of `HOST` because pixi/conda
may define `HOST=x86_64-conda-linux-gnu` for build tooling. Leave `RLDX_HOST`
unset for local eval, or set `RLDX_HOST=127.0.0.1` explicitly.

## Run

Terminal 1: start the RLDX-1 ZeroMQ policy server from the RLDX environment.
For RTX 5090 / Blackwell, run it through pixi:

```bash
cd /home/dorian/Projects/robocasa/external/rldx-1
pixi run --environment rldx bash /home/dorian/Projects/robocasa/model_evals/backends/rldx/server.sh 0
```

For a standard uv environment:

```bash
cd /home/dorian/Projects/robocasa
source external/rldx-1/.venv/bin/activate
bash model_evals/backends/rldx/server.sh 0
```

Terminal 2: run RoboCasa simulation from the environment that already has
RoboCasa, robosuite, and MuJoCo working. The RLDX client also needs `pyzmq` for
ZeroMQ communication with the server.

```bash
conda activate robocasa
python -m pip install pyzmq

BACKEND=rldx bash model_evals/backends/rldx/client.sh 1 \
  --single_task CloseBlenderLid \
  --cameras default \
  --trajectory_init_mode none
```

RLDX uses its own ZeroMQ protocol, not the WebSocket protocol used by pi05/GWP.
The adapter sends RoboCasa observations in the temporal format expected by the
RLDX-1 RoboCasa checkpoint: four video frames shaped `(B=1, T=4, H, W, C)` and
single-frame state shaped `(B=1, T=1, D)`. Early episode steps are padded by
repeating the first frame. It also passes `session_ids` and `reset_memory`
options for the memory cache, and extracts the returned 16-step action
dictionary into RoboCasa action order: `ee_pos, ee_rot, gripper, base_motion,
control_mode`.

## Troubleshooting

If the server fails with `unsupported operand type(s) for /: 'NoneType' and
'int'` inside `rldx/data/augmentations.py`, the checkpoint has loaded
`image_max_area=None`. The local ignored RLDX checkout is patched to coerce
`image_max_area` to the documented default `65536` and `image_resize_m` to `32`.
Restart the RLDX server after applying or pulling this workspace change.
