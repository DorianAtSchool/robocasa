"""Define the shared sampling strategy interface and base sampling behavior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from data_generation.task_level.tasks import TaskDefinition
from data_generation.task_level.tasks.base import PreflightTokenEstimate

if TYPE_CHECKING:
    from data_generation.task_level.trajectory_generation import RuntimeConfig
    from data_generation.task_level.tasks.base import TaskInstance


@dataclass(frozen=True)
class SampledTrajectoryCandidate:
    """Stores one parsed candidate trajectory emitted by a sampling strategy."""

    candidate: dict[str, Any]
    raw_output: Any
    probability: float | None = None


class SamplingStrategy(Protocol):
    """Protocol implemented by task-level response sampling strategies."""

    name: str

    def trajectories_per_run(self, runtime_config: RuntimeConfig) -> int:
        """Returns how many saved trajectories one successful run yields."""

    def build_prompt(
        self,
        *,
        task_definition: TaskDefinition,
        runtime_config: RuntimeConfig,
        task_instance: TaskInstance,
        variation_key: str,
        retry_feedback: str | None = None,
    ) -> str:
        """Builds the prompt for one run."""

    def response_schema(
        self,
        *,
        task_definition: TaskDefinition,
        runtime_config: RuntimeConfig,
    ) -> dict[str, Any]:
        """Builds the expected model response schema for one run."""

    def preflight_token_estimate(
        self,
        *,
        task_definition: TaskDefinition,
        runtime_config: RuntimeConfig,
    ) -> PreflightTokenEstimate:
        """Returns the manual token estimate for one run."""

    def extract_candidates(
        self,
        *,
        raw_response: Any,
        task_definition: TaskDefinition,
        runtime_config: RuntimeConfig,
    ) -> list[SampledTrajectoryCandidate]:
        """Parses one model response into saved candidate trajectories."""


class BaseSamplingStrategy:
    """Preserves the existing single-trajectory generation contract."""

    name = "base"

    def trajectories_per_run(self, runtime_config: RuntimeConfig) -> int:
        """Returns the single saved trajectory emitted by base sampling."""

        return 1

    def build_prompt(
        self,
        *,
        task_definition: TaskDefinition,
        runtime_config: RuntimeConfig,
        task_instance: TaskInstance,
        variation_key: str,
        retry_feedback: str | None = None,
    ) -> str:
        """Builds the unchanged task prompt for one base-sampled run."""

        del runtime_config
        return task_definition.build_prompt(
            variation_key,
            task_instance=task_instance,
            retry_feedback=retry_feedback,
        )

    def response_schema(
        self,
        *,
        task_definition: TaskDefinition,
        runtime_config: RuntimeConfig,
    ) -> dict[str, Any]:
        """Uses the task's original single-trajectory response schema."""

        del runtime_config
        return task_definition.response_schema

    def preflight_token_estimate(
        self,
        *,
        task_definition: TaskDefinition,
        runtime_config: RuntimeConfig,
    ) -> PreflightTokenEstimate:
        """Uses the task's original manual token estimate."""

        del runtime_config
        return task_definition.preflight_token_estimate

    def extract_candidates(
        self,
        *,
        raw_response: Any,
        task_definition: TaskDefinition,
        runtime_config: RuntimeConfig,
    ) -> list[SampledTrajectoryCandidate]:
        """Parses the unchanged single-trajectory response payload."""

        del task_definition
        del runtime_config
        from data_generation.task_level.trajectory_generation import extract_json_candidate

        candidate = extract_json_candidate(raw_response)
        return [SampledTrajectoryCandidate(candidate=candidate, raw_output=raw_response)]
