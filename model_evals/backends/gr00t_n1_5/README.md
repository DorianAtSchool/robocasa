# GR00T N1.5 RoboCasa Eval Backend

This backend runs the RoboCasa leaderboard GR00T N1.5 checkpoint through the RoboCasa Isaac-GR00T fork and the shared `model_evals` client.

## Setup

Use a separate GR00T environment for the server. The client still runs in the normal RoboCasa environment.

### RTX 5090 / Blackwell

Do not use `pip install -e .[base]` on RTX 50-series / Blackwell. The GR00T
`base` extra pins `torch==2.5.1`, which can install a CUDA 12.4 stack and fail
with kernel / architecture errors on `sm_120`.

Use the repo-local pixi environment instead:

```bash
cd /home/dorian/Projects/robocasa/external/Isaac-GR00T
pixi install
pixi run postinstall
pixi run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

This uses CUDA 12.8, `torch==2.8.0+cu128`, and source-builds flash-attn for
`sm_120`, matching the RLDX-1 Blackwell setup strategy.

### Non-Blackwell GPUs

```bash
cd /home/dorian/Projects/robocasa
git clone https://github.com/robocasa-benchmark/Isaac-GR00T external/Isaac-GR00T

cd external/Isaac-GR00T
pip install -e .[base]
pip install --no-build-isolation flash-attn==2.7.1.post4
```

Download the checkpoint:

```bash
cd /home/dorian/Projects/robocasa
hf download robocasa/robocasa365_checkpoints --repo-type model \
  --include 'gr00t_n1-5/multitask_learning/checkpoint-120000/**' \
  --local-dir model_evals/backends/gr00t_n1_5/ckpts
```

## Run

Start the server from the GR00T environment. For RTX 5090 / Blackwell, run it
through pixi:

```bash
cd /home/dorian/Projects/robocasa/external/Isaac-GR00T
pixi run bash /home/dorian/Projects/robocasa/model_evals/backends/gr00t_n1_5/server.sh 0
```

For a standard Python/conda environment:

```bash
cd /home/dorian/Projects/robocasa
bash model_evals/backends/gr00t_n1_5/server.sh 0
```

Run the RoboCasa client from the RoboCasa environment:

```bash
conda activate robocasa
python -m pip install pyzmq

BACKEND=gr00t_n1_5 bash model_evals/backends/gr00t_n1_5/client.sh 1 \
  --single_task CloseBlenderLid \
  --cameras default \
  --trajectory_init_mode none
```

To evaluate with the passive second robot initialized from your generated trajectories:

```bash
EXTRA_ROBOT=1 BACKEND=gr00t_n1_5 bash model_evals/backends/gr00t_n1_5/client.sh 1 \
  --single_task DeliverStraw \
  --cameras default \
  --trajectory_init_mode passive_other \
  --trajectory_index 0
```

## Notes

The adapter uses the `panda_omron` GR00T data config:

- Cameras: `video.robot0_agentview_left`, `video.robot0_agentview_right`, `video.robot0_eye_in_hand`
- States: end-effector position/rotation, gripper qpos, base position/rotation
- Language: `annotation.human.task_description`
- Actions: end-effector position/rotation, gripper close, base motion, control mode

GR00T returns a 16-step action dictionary. The adapter concatenates those action fields into the same flat action order used by RoboCasa demos before the shared evaluator calls `convert_action`.
