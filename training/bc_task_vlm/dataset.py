"""Dataset loading and multimodal collation for task-level VLM SFT."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor

from training.bc_task_vlm.prompting import build_messages, build_user_prompt
from training.bc_task_vlm.schema_utils import (
    build_single_step_response_schema,
    compact_json_dumps,
    validate_single_step_payload,
)
from training.bc_task_vlm.task_registry import AGENT_IDS, get_task_metadata


@dataclass(frozen=True)
class TrajectoryStepExample:
    """One supervised next-step prediction example."""

    sample_id: str
    task_name: str
    composite_task: str
    trajectory_id: str
    step_index: int
    agent_id: str
    task_instruction: str
    observation_views: list[str]
    image_paths: list[str]
    history_steps: list[dict[str, Any]]
    allowed_tool_specs: dict[str, dict[str, Any]]
    response_schema: dict[str, Any]
    target_payload: dict[str, Any]
    target_text: str
    messages: list[dict[str, Any]]

    def to_manifest_entry(self) -> dict[str, Any]:
        """Returns the compact sample metadata persisted with the run."""

        return {
            "sample_id": self.sample_id,
            "task_name": self.task_name,
            "trajectory_id": self.trajectory_id,
            "step_index": self.step_index,
            "agent_id": self.agent_id,
            "num_images": len(self.image_paths),
            "observation_views": list(self.observation_views),
        }


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalize_history_step(step: dict[str, Any]) -> dict[str, Any]:
    return {
        "step": step["step"],
        "agent": step["agent"],
        "tool": step["tool"],
        "args": dict(step["args"]),
        "reasoning": step["reasoning"],
    }


def _find_latest_same_agent_observation(
    plan_steps: list[dict[str, Any]],
    *,
    before_index: int,
    agent_id: str,
) -> tuple[list[str], list[str]]:
    for step in reversed(plan_steps[:before_index]):
        if step.get("tool") != "get_image":
            continue
        metadata = step.get("metadata", {})
        if metadata.get("source_agent") != agent_id:
            continue
        args = step.get("args", {})
        image_paths = list(args.get("image_paths", ()))
        views = list(args.get("views", ()))
        if not image_paths:
            continue
        return image_paths, views
    raise ValueError(
        f"Missing preceding observation image for agent {agent_id!r} before step {before_index}."
    )


def _validate_alignment(
    *,
    task_name: str,
    trajectory_id: str,
    raw_steps: list[dict[str, Any]],
    plan_steps: list[dict[str, Any]],
    executed_steps: list[dict[str, Any]],
) -> None:
    if not (len(raw_steps) == len(plan_steps) == len(executed_steps)):
        raise ValueError(
            f"Step count mismatch for {task_name}/{trajectory_id}: "
            f"raw={len(raw_steps)} plan={len(plan_steps)} executed={len(executed_steps)}."
        )
    for index, (raw_step, plan_step, executed_step) in enumerate(
        zip(raw_steps, plan_steps, executed_steps, strict=True)
    ):
        if raw_step.get("step") != index:
            raise ValueError(
                f"Unexpected raw step index in {task_name}/{trajectory_id}: "
                f"expected {index}, got {raw_step.get('step')}."
            )
        plan_index = plan_step.get("metadata", {}).get("step_index")
        if plan_index != index:
            raise ValueError(
                f"Unexpected plan step index in {task_name}/{trajectory_id}: "
                f"expected {index}, got {plan_index}."
            )
        executed_index = executed_step.get("step_index")
        if executed_index != index:
            raise ValueError(
                f"Unexpected executed step index in {task_name}/{trajectory_id}: "
                f"expected {index}, got {executed_index}."
            )
        if raw_step.get("tool") != plan_step.get("tool"):
            raise ValueError(
                f"Tool mismatch at {task_name}/{trajectory_id} step {index}: "
                f"raw={raw_step.get('tool')} plan={plan_step.get('tool')}."
            )


def build_examples(
    *,
    dataset_root: Path,
    task_names: list[str],
) -> list[TrajectoryStepExample]:
    """Builds one SFT example per successful non-image action step."""

    examples: list[TrajectoryStepExample] = []

    for task_name in task_names:
        task_metadata = get_task_metadata(task_name)
        task_root = dataset_root / task_metadata.dataset_name
        if not task_root.is_dir():
            raise FileNotFoundError(f"Task directory does not exist: {task_root}")

        response_schema = build_single_step_response_schema(
            agent_ids=AGENT_IDS,
            allowed_tool_specs=task_metadata.allowed_tool_specs,
        )

        for trajectory_dir in sorted(
            path for path in task_root.iterdir() if path.is_dir()
        ):
            original_trajectory = _load_json(
                trajectory_dir / "original_trajectory.json"
            )
            plan_steps = _load_json(trajectory_dir / "plan.json")
            metadata = _load_json(trajectory_dir / "metadata.json")

            raw_steps = list(original_trajectory["steps"])
            executed_steps = list(metadata["steps"])
            trajectory_id = str(original_trajectory["trajectory_id"])

            _validate_alignment(
                task_name=task_name,
                trajectory_id=trajectory_id,
                raw_steps=raw_steps,
                plan_steps=plan_steps,
                executed_steps=executed_steps,
            )

            history_steps: list[dict[str, Any]] = []
            for raw_step, executed_step in zip(raw_steps, executed_steps, strict=True):
                if raw_step["tool"] == "get_image":
                    continue

                if not executed_step.get("success", False):
                    continue

                image_paths, observation_views = _find_latest_same_agent_observation(
                    plan_steps,
                    before_index=raw_step["step"],
                    agent_id=raw_step["agent"],
                )

                missing_images = [
                    image_path
                    for image_path in image_paths
                    if not Path(image_path).exists()
                ]
                if missing_images:
                    missing = ", ".join(missing_images)
                    raise FileNotFoundError(
                        f"Missing rendered images for {task_name}/{trajectory_id}: {missing}"
                    )

                target_payload = validate_single_step_payload(
                    {
                        "steps": [
                            {
                                "step": raw_step["step"],
                                "agent": raw_step["agent"],
                                "tool": raw_step["tool"],
                                "args": raw_step["args"],
                                "reasoning": raw_step["reasoning"],
                            }
                        ]
                    },
                    agent_ids=AGENT_IDS,
                    allowed_tool_specs=task_metadata.allowed_tool_specs,
                )
                target_text = compact_json_dumps(target_payload)
                user_prompt = build_user_prompt(
                    composite_task=task_metadata.composite_task,
                    task_instruction=metadata["task"],
                    agent_id=raw_step["agent"],
                    next_step_index=raw_step["step"],
                    observation_views=observation_views,
                    history_steps=history_steps,
                    allowed_tool_specs=task_metadata.allowed_tool_specs,
                )
                sample_id = f"{task_metadata.dataset_name}/{trajectory_id}/step_{raw_step['step']:06d}"
                examples.append(
                    TrajectoryStepExample(
                        sample_id=sample_id,
                        task_name=task_metadata.dataset_name,
                        composite_task=task_metadata.composite_task,
                        trajectory_id=trajectory_id,
                        step_index=raw_step["step"],
                        agent_id=raw_step["agent"],
                        task_instruction=metadata["task"],
                        observation_views=observation_views,
                        image_paths=list(image_paths),
                        history_steps=list(history_steps),
                        allowed_tool_specs=task_metadata.allowed_tool_specs,
                        response_schema=response_schema,
                        target_payload=target_payload,
                        target_text=target_text,
                        messages=build_messages(
                            user_prompt=user_prompt,
                            num_images=len(image_paths),
                            target_text=target_text,
                        ),
                    )
                )
                history_steps.append(_normalize_history_step(raw_step))

    return examples


class TrajectoryStepDataset(Dataset):
    """Thin PyTorch dataset wrapper around precomputed step examples."""

    def __init__(self, examples: list[TrajectoryStepExample]) -> None:
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        return {
            "sample_id": example.sample_id,
            "task_name": example.task_name,
            "composite_task": example.composite_task,
            "trajectory_id": example.trajectory_id,
            "step_index": example.step_index,
            "agent_id": example.agent_id,
            "observation_views": list(example.observation_views),
            "image_paths": list(example.image_paths),
            "allowed_tool_specs": example.allowed_tool_specs,
            "response_schema": example.response_schema,
            "target_payload": example.target_payload,
            "target_text": example.target_text,
            "messages": example.messages,
        }


def build_split_manifest(
    *,
    dataset_root: Path,
    train_examples: list[TrajectoryStepExample],
    val_examples: list[TrajectoryStepExample],
) -> dict[str, Any]:
    """Builds a compact manifest describing the train/validation split."""

    def summarize_split(examples: list[TrajectoryStepExample]) -> dict[str, Any]:
        counts_by_task: dict[str, dict[str, Any]] = {}
        for example in examples:
            entry = counts_by_task.setdefault(
                example.task_name,
                {"num_samples": 0, "trajectory_ids": set()},
            )
            entry["num_samples"] += 1
            entry["trajectory_ids"].add(example.trajectory_id)

        return {
            "num_samples": len(examples),
            "tasks": {
                task_name: {
                    "num_samples": task_summary["num_samples"],
                    "num_trajectories": len(task_summary["trajectory_ids"]),
                }
                for task_name, task_summary in sorted(counts_by_task.items())
            },
            "samples": [example.to_manifest_entry() for example in examples],
        }

    return {
        "dataset_root": str(dataset_root.resolve()),
        "train": summarize_split(train_examples),
        "validation": summarize_split(val_examples),
    }


class LazyVisionSFTCollator:
    """Collates multimodal SFT batches and masks loss to assistant tokens only."""

    def __init__(
        self,
        *,
        processor_name_or_path: str,
        max_length: int | None,
        trust_remote_code: bool,
    ) -> None:
        self.processor_name_or_path = processor_name_or_path
        self.max_length = max_length
        self.trust_remote_code = trust_remote_code
        self._processor = None

    def _get_processor(self):
        if self._processor is None:
            processor = AutoProcessor.from_pretrained(
                self.processor_name_or_path,
                trust_remote_code=self.trust_remote_code,
            )
            tokenizer = getattr(processor, "tokenizer", None)
            if tokenizer is not None:
                tokenizer.padding_side = "right"
            self._processor = processor
        return self._processor

    @staticmethod
    def _load_images(image_paths: list[str]) -> list[Image.Image]:
        images: list[Image.Image] = []
        for image_path in image_paths:
            with Image.open(image_path) as image:
                images.append(image.convert("RGB"))
        return images

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        processor = self._get_processor()
        messages = [feature["messages"] for feature in features]
        prompt_messages = [message[:-1] for message in messages]
        images = [self._load_images(feature["image_paths"]) for feature in features]

        full_texts = [
            processor.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=False,
            )
            for message in messages
        ]
        prompt_texts = [
            processor.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True,
            )
            for message in prompt_messages
        ]

        tokenizer_kwargs = {
            "text": full_texts,
            "images": images,
            "padding": True,
            "return_tensors": "pt",
        }
        prompt_tokenizer_kwargs = {
            "text": prompt_texts,
            "images": images,
            "padding": True,
            "return_tensors": "pt",
        }
        if self.max_length is not None:
            tokenizer_kwargs["max_length"] = self.max_length
            tokenizer_kwargs["truncation"] = True
            prompt_tokenizer_kwargs["max_length"] = self.max_length
            prompt_tokenizer_kwargs["truncation"] = True

        batch = processor(**tokenizer_kwargs)
        prompt_batch = processor(**prompt_tokenizer_kwargs)
        batch.pop("token_type_ids", None)
        prompt_batch.pop("token_type_ids", None)

        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        prompt_lengths = prompt_batch["attention_mask"].sum(dim=1).tolist()
        for row_index, prompt_length in enumerate(prompt_lengths):
            labels[row_index, :prompt_length] = -100
        batch["labels"] = labels
        return batch
