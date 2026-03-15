"""Implement verbalized sampling over task-level trajectories."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from data_generation.task_level.sampling.base import SampledTrajectoryCandidate
from data_generation.task_level.tasks import ResponseFormatValidationError
from data_generation.task_level.tasks.base import (
    PreflightTokenEstimate,
    TrajectoryStructureValidationError,
)
from data_generation.utils import stable_json_sha256

if TYPE_CHECKING:
    from data_generation.task_level.tasks import TaskDefinition
    from data_generation.task_level.trajectory_generation import RuntimeConfig


class VerbalizedSamplingValidationError(TrajectoryStructureValidationError):
    """Raised when a verbalized multi-trajectory response is malformed."""


@dataclass(frozen=True)
class VerbalizedTrajectorySequenceValidator:
    """Validates the outer verbalized response before per-trajectory validation."""

    expected_count: int

    def validate(self, response_payload: dict[str, Any]) -> list[SampledTrajectoryCandidate]:
        """Parses one verbalized response object into distinct candidate trajectories."""

        responses = response_payload.get("responses")
        if not isinstance(responses, list):
            raise VerbalizedSamplingValidationError(
                "verbalized response must contain a responses array."
            )
        if len(responses) != self.expected_count:
            raise VerbalizedSamplingValidationError(
                f"verbalized response must contain exactly {self.expected_count} trajectories."
            )

        sampled_candidates: list[SampledTrajectoryCandidate] = []
        seen_signatures: set[str] = set()
        for index, response_entry in enumerate(responses):
            if not isinstance(response_entry, dict):
                raise VerbalizedSamplingValidationError(
                    f"responses[{index}] must be an object."
                )

            probability = response_entry.get("probability")
            if not isinstance(probability, (int, float)) or isinstance(probability, bool):
                raise VerbalizedSamplingValidationError(
                    f"responses[{index}].probability must be a number."
                )
            normalized_probability = float(probability)
            if normalized_probability < 0.0 or normalized_probability > 1.0:
                raise VerbalizedSamplingValidationError(
                    f"responses[{index}].probability must be between 0 and 1."
                )

            trajectory = response_entry.get("trajectory")
            if not isinstance(trajectory, dict):
                raise VerbalizedSamplingValidationError(
                    f"responses[{index}].trajectory must be an object."
                )

            trajectory_signature = stable_json_sha256(trajectory)
            if trajectory_signature in seen_signatures:
                raise VerbalizedSamplingValidationError(
                    "verbalized response contained duplicate trajectories."
                )
            seen_signatures.add(trajectory_signature)

            sampled_candidates.append(
                SampledTrajectoryCandidate(
                    candidate=trajectory,
                    raw_output={
                        "probability": normalized_probability,
                        "trajectory": trajectory,
                    },
                    probability=normalized_probability,
                )
            )

        return sampled_candidates


class VerbalizedSamplingStrategy:
    """Asks the model for multiple labeled trajectories in one response."""

    name = "verbalized"

    def trajectories_per_run(self, runtime_config: RuntimeConfig) -> int:
        """Returns the configured number of trajectories saved per verbalized run."""

        return runtime_config.verbalized_k

    def build_prompt(
        self,
        *,
        task_definition: TaskDefinition,
        runtime_config: RuntimeConfig,
        variation_key: str,
        retry_feedback: str | None = None,
    ) -> str:
        """Wraps the task prompt with verbalized multi-trajectory response instructions."""

        base_prompt = task_definition.build_prompt(
            variation_key,
            retry_feedback=retry_feedback,
        )
        return (
            f"{base_prompt}\n\n"
            "Verbalized sampling instructions:\n"
            f"- Return exactly {runtime_config.verbalized_k} distinct candidate trajectories.\n"
            "- Return one JSON object with key responses.\n"
            f"- Responses must be an array with exactly {runtime_config.verbalized_k} items.\n"
            "- Each item must be an object with keys probability and trajectory.\n"
            "- Probability must be a number between 0 and 1 representing the model's estimated likelihood for that full trajectory.\n"
            "- Trajectory must be one complete trajectory object that satisfies the trajectory object requirements above.\n"
            "- For each trajectory, also vary amount of communication between agents. The range is between communicating with high frequency (every step) to communicating infrequently (just at the beginning). You should include both extremes as trajectories."
            "- Do not return a standalone top-level steps object.\n"
            "- Make the trajectories meaningfully distinct from each other.\n"
            "- Output JSON only and do not include markdown."
        )

    def response_schema(
        self,
        *,
        task_definition: TaskDefinition,
        runtime_config: RuntimeConfig,
    ) -> dict[str, Any]:
        """Builds the nested response schema expected by verbalized sampling."""

        return {
            "type": "OBJECT",
            "required": ["responses"],
            "properties": {
                "responses": {
                    "type": "ARRAY",
                    "minItems": runtime_config.verbalized_k,
                    "maxItems": runtime_config.verbalized_k,
                    "items": {
                        "type": "OBJECT",
                        "required": ["probability", "trajectory"],
                        "properties": {
                            "probability": {"type": "NUMBER"},
                            "trajectory": task_definition.response_schema,
                        },
                    },
                }
            },
        }

    def preflight_token_estimate(
        self,
        *,
        task_definition: TaskDefinition,
        runtime_config: RuntimeConfig,
    ) -> PreflightTokenEstimate:
        """Scales the per-run output estimate to account for K trajectories."""

        base_estimate = task_definition.preflight_token_estimate
        return PreflightTokenEstimate(
            prompt_tokens=base_estimate.prompt_tokens,
            output_tokens=base_estimate.output_tokens * runtime_config.verbalized_k,
            reasoning_tokens=base_estimate.reasoning_tokens,
        )

    def extract_candidates(
        self,
        *,
        raw_response: Any,
        task_definition: TaskDefinition,
        runtime_config: RuntimeConfig,
    ) -> list[SampledTrajectoryCandidate]:
        """Parses one verbalized response into per-trajectory candidate objects."""

        del task_definition
        response_payload = _extract_json_object(raw_response)
        validator = VerbalizedTrajectorySequenceValidator(
            expected_count=runtime_config.verbalized_k
        )
        return validator.validate(response_payload)


def _extract_json_object(raw_response: Any) -> dict[str, Any]:
    """Parses one raw model response into a JSON object for sequence validation."""

    if isinstance(raw_response, dict):
        return raw_response
    if not isinstance(raw_response, str):
        raise ResponseFormatValidationError(
            f"Unsupported model response type: {type(raw_response).__name__}"
        )

    stripped = raw_response.strip()
    fenced_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", stripped, re.DOTALL)
    if fenced_match:
        stripped = fenced_match.group(1)
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as exc:
        json_match = re.search(r"(\{.*\})", stripped, re.DOTALL)
        if not json_match:
            raise ResponseFormatValidationError(
                "Model response did not contain JSON."
            ) from exc
        try:
            parsed = json.loads(json_match.group(1))
        except json.JSONDecodeError as inner_exc:
            raise ResponseFormatValidationError(
                "Model response contained invalid JSON."
            ) from inner_exc
    if not isinstance(parsed, dict):
        raise ResponseFormatValidationError(
            "Model response must decode to a JSON object."
        )
    return parsed
