# Task-Level BC VLM Training

## Important TODOs
 - [x] Verify that the image observations are being sent through a ViT or whatever encoder the model uses
 - [x] Verify that the images are excluded from the cross-entropy loss
 - [x] Verify that the ouptut is structured wrt the finetunned model
 - [x] Split joint demonstrations into one training conversation per agent so the loss only applies to that agent's tool-call turns.
 - [] Verify wandb works
 - [] Verify pretraining works on the NCSA cluster with the small amount of data we have.
 - [] Fix the NCSA cluster CUDA/PyTorch environment mismatch; the current multi-GPU launch warns that the NVIDIA driver is too old for the installed torch build and then fails precision validation during Trainer startup.
 - [] Verify that the task-vlm prompt is inlcuded in the context. We will have to construct this. I don't remember doing this.
 

This directory contains a multi-GPU supervised fine-tuning pipeline for image-conditioned next-step prediction on the rendered task-level dataset.

The default checkpoint is `Qwen/Qwen3.5-0.8B`, not `Qwen/Qwen3.5-0.8B-Base`. This
pipeline uses chat-formatted supervision with tool schemas and assistant
tool-call targets, so the post-trained checkpoint is the better default when
you need structured outputs during SFT and evaluation.

Training uses task-local JSON Schema tool definitions passed through
`apply_chat_template(..., tools=...)`. By default, each joint two-agent
demonstration is converted into two training conversations, one per agent. The
global mixed-agent execution history stays in the user prompt, but only the
selected agent's assistant tool-call turns contribute to the loss. Validation
still uses one-step next-tool prediction, and the predicted action itself is
trained as an actual Qwen/Hugging Face tool call rather than a custom
`{"steps":[...]}` blob.

## Training Modes

`--train-example-granularity decentralized` is the default.

- `centralized`: one training example per successful non-`get_image` global action step. Each sample asks for the single next tool call in the joint trajectory, including whichever agent acts next.
- `decentralized`: one training example per `(trajectory, agent)`. A single joint two-agent demonstration becomes two training conversations, one for `agent_0` and one for `agent_1`. The other agent's actions stay in the prompt history, but the loss is only applied to the selected agent's assistant tool-call turns.

Example:

- Joint trajectory actions: `agent_0 -> communicate`, `agent_1 -> communicate`, `agent_1 -> pick_up_object`, `agent_0 -> navigate_to_fixture`
- `centralized`: 4 separate next-step training samples
- `decentralized`: 2 training samples total

Validation remains step-level in both cases.

## Training Sample Build Progress And Cache

When training samples are built from raw trajectories, the main process now shows a
trajectory-level `tqdm` progress bar for each task. This appears during cache
misses, for example when you point at a new dataset root or when the cache has
been invalidated.

Caching is enabled by default through `--use-example-cache`. Cached serialized
training samples are stored under:

```text
.cache/bc_task_vlm/examples/
```

The cache is keyed by dataset root, task name, and granularity
(`centralized` vs `decentralized`). It is automatically invalidated when any of
these change:

- `original_trajectory.json`
- `plan.json`
- `metadata.json`
- the BC task VLM example-building code or prompt/schema helpers

On repeated runs, training and validation sample construction will load from
cache and skip the expensive rebuild step. In multi-GPU launches, rank 0 builds
or refreshes the cache and the other ranks wait, then reuse the cached result.

Useful flags:

- `--use-example-cache` / `--no-use-example-cache`
- `--training-samples-cache-dir /path/to/cache`

Use the post-trained checkpoint for launches unless you have a specific reason
to compare against the pre-trained-only base model:

```bash
accelerate launch \
  --config_file training/bc_task_vlm/accelerate_multigpu.yaml \
  --num_processes 2 \
  -m training.bc_task_vlm.main \
  --model-name-or-path Qwen/Qwen3.5-0.8B \
  --output-dir training/bc_task_vlm/runs/qwen35_08b_lora \
  --wandb-project robocasa-bc-task-vlm \
  --wandb-run-name qwen35-08b-lora-prepare-coffee-holdout
```

## Install

If you are using the repo's `uv` workflow, install the training dependencies into
your active environment with:

```bash
uv pip install -r training/bc_task_vlm/requirements.txt
```

If you still need to create the base environment, follow `RBR_README.md` first.
Run from the repo root so `python -m training.bc_task_vlm.main` can import the
local `training/` package.

### NCSA Cluster Notes

On shared clusters, do not assume `uv pip install -r training/bc_task_vlm/requirements.txt`
will pick a torch wheel that matches the node driver. The failure mode you saw
is exactly what happens when pip resolves a newer CUDA build than the host
driver supports.

Check the node first:

```bash
nvidia-smi --query-gpu=name,driver_version --format=csv
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

If `torch.cuda.is_available()` is `False` and torch warns that the NVIDIA
driver is too old, reinstall torch and torchvision with a CUDA build the node
driver supports, then install the rest of the repo requirements. PyTorch's
official wheel indexes for older CUDA builds are documented here:
https://docs.pytorch.org/get-started/previous-versions/

Examples:

```bash
uv pip uninstall -y torch torchvision
uv pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install -r training/bc_task_vlm/requirements.txt
```

```bash
uv pip uninstall -y torch torchvision
uv pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu118
uv pip install -r training/bc_task_vlm/requirements.txt
```

After reinstalling, verify that CUDA is usable before launching distributed
training:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.device_count(), torch.cuda.is_available())"
```

## Dataset Split

Default leave-one-task-out split:

- train: `hot_dog_setup`, `prepare_sandwich_station`
- validation: `prepare_coffee`

## Multi-GPU Launch

Use the helper script or call `accelerate` directly.

```bash
training/bc_task_vlm/launch_multigpu.sh \
  --output-dir training/bc_task_vlm/runs/qwen35_08b_lora \
  --wandb-project robocasa-bc-task-vlm \
  --wandb-run-name qwen35-08b-lora-prepare-coffee-holdout
```

Equivalent direct launch:

```bash
accelerate launch \
  --config_file training/bc_task_vlm/accelerate_multigpu.yaml \
  --num_processes 4 \
  -m training.bc_task_vlm.main \
  --train-example-granularity decentralized \
  --output-dir training/bc_task_vlm/runs/qwen35_08b_lora
```

Override `--num_processes` to match the number of visible GPUs.
The helper script now counts GPUs from `CUDA_VISIBLE_DEVICES`, Slurm GPU env
vars, or `nvidia-smi` so it does not need to import `torch` before launch.

Precision is selected by `training.bc_task_vlm.main`, not by the Accelerate
YAML. The script will keep `bf16` on supported GPUs and fall back to `fp16`
when the node does not support `bf16`.

If you want to move or disable the training-samples cache during launch:

```bash
training/bc_task_vlm/launch_multigpu.sh \
  --training-samples-cache-dir /tmp/robocasa_bc_task_vlm_cache
```

```bash
training/bc_task_vlm/launch_multigpu.sh \
  --no-use-example-cache
```

## W&B

Create a repo-root `.env` file:

```bash
cat > .env <<'EOF'
WANDB_API_KEY=your-wandb-api-key
WANDB_PROJECT=robocasa-bc-task-vlm
# Optional:
# WANDB_ENTITY=your-team-or-username
EOF
```

`training.bc_task_vlm.main` automatically loads the repo-root `.env` before CLI
parsing. Shell environment variables still win if you already exported a value
manually.

Online tracking:

```bash
training/bc_task_vlm/launch_multigpu.sh --wandb-mode online
```

Offline tracking:

```bash
training/bc_task_vlm/launch_multigpu.sh --wandb-mode offline
```

Disable external tracking:

```bash
training/bc_task_vlm/launch_multigpu.sh --report-to none
```

## Outputs

Each run directory stores:

- `run_config.json`
- `split_manifest.json`
- LoRA checkpoints saved by `Trainer`
- `processor/`
- `structured_eval_metrics.json`
- `structured_eval_predictions.jsonl`
- `final_metrics.json`
