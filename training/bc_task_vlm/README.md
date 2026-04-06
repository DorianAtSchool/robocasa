# Task-Level BC VLM Training

This directory contains a multi-GPU supervised fine-tuning pipeline for image-conditioned next-step prediction on the rendered task-level dataset.

## Install

```bash
pip install -r training/bc_task_vlm/requirements.txt
```

Run from the repo root so `python -m training.bc_task_vlm.main` can import the local `training/` package.

## Dataset Split

Default leave-one-task-out split:

- train: `hot_dog_setup`, `prepare_sandwich_station`
- validation: `prepare_coffee`

## Multi-GPU Launch

Use the helper script or call `accelerate` directly.

```bash
training/bc_task_vlm/launch_multigpu.sh \
  --output-dir training/bc_task_vlm/runs/qwen35_9b_lora \
  --wandb-project robocasa-bc-task-vlm \
  --wandb-run-name qwen35-9b-lora-prepare-coffee-holdout
```

Equivalent direct launch:

```bash
accelerate launch \
  --config_file training/bc_task_vlm/accelerate_multigpu.yaml \
  --num_processes 4 \
  -m training.bc_task_vlm.main \
  --output-dir training/bc_task_vlm/runs/qwen35_9b_lora
```

Override `--num_processes` to match the number of visible GPUs.

## W&B

Online tracking:

```bash
export WANDB_API_KEY=...
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
