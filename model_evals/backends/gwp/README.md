# GWP Backend

This backend contains the GigaWorld-Policy RoboCasa server/client entrypoints and
its small `world_action_model` source package. It is self-contained inside the
canonical RoboCasa repo except for external checkpoints, base model weights, and
normal Python/CUDA dependencies.

```text
model_evals/backends/gwp/
  pyproject.toml              # server-side GWP dependency metadata
  src/world_action_model/     # vendored GWP model code required by server
  inference_server.py         # model server
  inference_client.py         # RoboCasa rollout client
  server.sh
  client.sh
```

The shared batch scripts in `model_evals/scripts/` call this backend's
`client.sh` when `BACKEND=gwp` or when `BACKEND` is unset. Future backends
should be added as siblings, for example:

```text
model_evals/backends/pi05/
model_evals/backends/gr00t_n1_5/
model_evals/backends/rldx_1/
```

## Server Environment

Use a separate server environment for the model stack. The exact CUDA-tagged
PyTorch wheel should match the machine.

```bash
conda create -n gwp python=3.11 -y
conda activate gwp

# Example CUDA 12.8 install. Adjust for the machine.
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128

# Install this backend's model package and server dependencies.
pip install -e /home/dorian/Projects/robocasa/model_evals/backends/gwp
```

The server needs a GWP checkpoint and norm stats. For this workspace, they are copied into backend-local ignored artifact folders:

```text
model_evals/backends/gwp/ckpts/model.pt
model_evals/backends/gwp/assets/norm_stats_delta.json
model_evals/backends/gwp/logs/
```

These folders are ignored by git because they contain large checkpoints and generated results.

## Client Environment

Use the normal RoboCasa environment for simulation and video logging:

```bash
conda activate robocasa
pip install -e /home/dorian/Projects/robocasa
pip install gymnasium imageio imageio-ffmpeg msgpack websockets numpy pillow tqdm
```

## Typical Debug Flow

Terminal 1, server env:

```bash
conda activate gwp
cd /home/dorian/Projects/robocasa
bash model_evals/backends/gwp/server.sh 0
```

Terminal 2, RoboCasa env:

```bash
conda activate robocasa
cd /home/dorian/Projects/robocasa
NUM_TRIALS=5 \
BASE_LOG_DIR="$PWD/model_evals/backends/gwp/logs/robot_interference/atomic_seen_5trials" \
bash model_evals/scripts/run_atomic_seen_conditions.sh
```

The backend scripts prepend `model_evals/backends/gwp/src`, the repo root, and
the vendored `robosuite/` checkout to `PYTHONPATH`, so running from source works
even before installing the backend package editable.
