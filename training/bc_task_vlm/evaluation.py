"""Structured generation evaluation for task-level VLM fine-tuning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoProcessor

from training.bc_task_vlm.schema_utils import (
    canonicalize_for_comparison,
    compact_json_dumps,
)
from training.bc_task_vlm.task_registry import AGENT_IDS
from training.bc_task_vlm.tool_calling import (
    extract_pre_tool_call_text,
    parse_first_qwen_tool_call,
    tool_call_to_single_step_payload,
)


class VisionGenerationCollator:
    """Collates prompt-only multimodal batches for structured generation eval."""

    def __init__(
        self,
        *,
        processor_name_or_path: str,
        max_length: int | None,
        trust_remote_code: bool,
    ) -> None:
        self.processor = AutoProcessor.from_pretrained(
            processor_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None:
            tokenizer.padding_side = "left"
        self.max_length = max_length

    @staticmethod
    def _load_images(image_paths: list[str]):
        from PIL import Image

        images = []
        for image_path in image_paths:
            with Image.open(image_path) as image:
                images.append(image.convert("RGB"))
        return images

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        prompt_messages = [feature["messages"][:-1] for feature in features]
        images = [self._load_images(feature["image_paths"]) for feature in features]
        prompt_texts = [
            self.processor.apply_chat_template(
                message,
                tools=feature["tool_schemas"],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for message, feature in zip(prompt_messages, features, strict=True)
        ]

        tokenizer_kwargs = {
            "text": prompt_texts,
            "images": images,
            "padding": True,
            "return_tensors": "pt",
        }
        if self.max_length is not None:
            tokenizer_kwargs["max_length"] = self.max_length
            tokenizer_kwargs["truncation"] = True

        batch = self.processor(**tokenizer_kwargs)
        batch.pop("token_type_ids", None)
        batch["sample_metadata"] = [
            {
                "sample_id": feature["sample_id"],
                "task_name": feature["task_name"],
                "trajectory_id": feature["trajectory_id"],
                "step_index": feature["step_index"],
                "agent_id": feature["agent_id"],
                "target_payload": feature["target_payload"],
                "target_text": feature["target_text"],
                "allowed_tool_specs": feature["allowed_tool_specs"],
            }
            for feature in features
        ]
        return batch


def _move_batch_to_device(
    batch: dict[str, Any], device: torch.device
) -> dict[str, Any]:
    tensor_batch: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            tensor_batch[key] = value.to(device)
        else:
            tensor_batch[key] = value
    return tensor_batch


def evaluate_structured_generation(
    *,
    model,
    eval_dataset,
    processor_name_or_path: str,
    output_dir: Path,
    max_length: int | None,
    max_new_tokens: int,
    batch_size: int,
    trust_remote_code: bool,
    max_samples: int | None = None,
) -> dict[str, float]:
    """Runs generation over the validation split and computes structured metrics."""

    if len(eval_dataset) == 0:
        return {}

    if max_samples is not None:
        max_samples = max(0, min(max_samples, len(eval_dataset)))
        dataset = torch.utils.data.Subset(eval_dataset, range(max_samples))
    else:
        dataset = eval_dataset

    collator = VisionGenerationCollator(
        processor_name_or_path=processor_name_or_path,
        max_length=max_length,
        trust_remote_code=trust_remote_code,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
    )

    processor = collator.processor
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "structured_eval_predictions.jsonl"

    unwrapped_model = model
    if hasattr(model, "module"):
        unwrapped_model = model.module
    device = next(unwrapped_model.parameters()).device

    total_samples = 0
    parsed_tool_calls = 0
    valid_tool_calls = 0
    exact_tool_matches = 0
    exact_args_matches = 0
    exact_action_matches = 0

    previous_use_cache = getattr(unwrapped_model.config, "use_cache", None)
    if previous_use_cache is not None:
        unwrapped_model.config.use_cache = True
    unwrapped_model.eval()

    with predictions_path.open("w", encoding="utf-8") as handle:
        with torch.inference_mode():
            for batch in tqdm(
                dataloader,
                desc="Structured validation",
                leave=False,
            ):
                sample_metadata = batch.pop("sample_metadata")
                batch = _move_batch_to_device(batch, device)
                generated_ids = unwrapped_model.generate(
                    **batch,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
                input_width = batch["input_ids"].shape[1]
                trimmed_ids = [output_ids[input_width:] for output_ids in generated_ids]
                decoded_outputs = processor.batch_decode(
                    trimmed_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )

                for metadata, decoded_text in zip(
                    sample_metadata,
                    decoded_outputs,
                    strict=True,
                ):
                    total_samples += 1
                    target_payload = metadata["target_payload"]
                    parsed_tool_call = None
                    normalized_prediction = None
                    parse_error = None
                    validation_error = None

                    try:
                        parsed_tool_call = parse_first_qwen_tool_call(decoded_text)
                        parsed_tool_calls += 1
                    except Exception as exc:  # pragma: no cover - defensive eval logging
                        parse_error = str(exc)

                    if parsed_tool_call is not None:
                        try:
                            normalized_prediction = tool_call_to_single_step_payload(
                                parsed_tool_call,
                                step_index=metadata["step_index"],
                                agent_id=metadata["agent_id"],
                                agent_ids=AGENT_IDS,
                                allowed_tool_specs=metadata["allowed_tool_specs"],
                                reasoning_text=(
                                    extract_pre_tool_call_text(decoded_text)
                                    or "tool_call"
                                ),
                            )
                            valid_tool_calls += 1
                        except Exception as exc:  # pragma: no cover - defensive eval logging
                            validation_error = str(exc)

                    if normalized_prediction is not None:
                        predicted_step = normalized_prediction["steps"][0]
                        target_step = target_payload["steps"][0]
                        if predicted_step["tool"] == target_step["tool"]:
                            exact_tool_matches += 1
                        if canonicalize_for_comparison(
                            predicted_step["args"]
                        ) == canonicalize_for_comparison(target_step["args"]):
                            exact_args_matches += 1
                        if canonicalize_for_comparison(
                            {
                                "step": predicted_step["step"],
                                "agent": predicted_step["agent"],
                                "tool": predicted_step["tool"],
                                "args": predicted_step["args"],
                            }
                        ) == canonicalize_for_comparison(
                            {
                                "step": target_step["step"],
                                "agent": target_step["agent"],
                                "tool": target_step["tool"],
                                "args": target_step["args"],
                            }
                        ):
                            exact_action_matches += 1

                    handle.write(
                        json.dumps(
                            {
                                "sample_id": metadata["sample_id"],
                                "task_name": metadata["task_name"],
                                "trajectory_id": metadata["trajectory_id"],
                                "step_index": metadata["step_index"],
                                "prediction_text": decoded_text,
                                "target_text": metadata["target_text"],
                                "target_payload": compact_json_dumps(target_payload),
                                "parsed_tool_call": parsed_tool_call,
                                "parse_error": parse_error,
                                "validation_error": validation_error,
                            },
                            ensure_ascii=True,
                        )
                        + "\n"
                    )

    if previous_use_cache is not None:
        unwrapped_model.config.use_cache = previous_use_cache

    def rate(count: int) -> float:
        if total_samples == 0:
            return 0.0
        return count / total_samples

    metrics = {
        "structured_eval_num_samples": float(total_samples),
        "structured_eval_tool_call_parse_rate": rate(parsed_tool_calls),
        "structured_eval_tool_call_valid_rate": rate(valid_tool_calls),
        "structured_eval_exact_tool_accuracy": rate(exact_tool_matches),
        "structured_eval_exact_args_match_rate": rate(exact_args_matches),
        "structured_eval_exact_action_step_match_rate": rate(exact_action_matches),
    }
    metrics_path = output_dir / "structured_eval_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metrics
