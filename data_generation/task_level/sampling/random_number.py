"""Implement UUID prompt tagging over base sampling."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from data_generation.task_level.sampling.base import BaseSamplingStrategy

if TYPE_CHECKING:
    from data_generation.task_level.generation.raw.config import RuntimeConfig
    from data_generation.task_level.tasks import TaskDefinition
    from data_generation.task_level.tasks.base import TaskInstance


class RandomNumberSamplingStrategy(BaseSamplingStrategy):
    """Prepends a UUID sample id while preserving base sampling behavior."""

    name = "random_number"

    def build_prompt(
        self,
        *,
        task_definition: TaskDefinition,
        runtime_config: RuntimeConfig,
        task_instance: TaskInstance,
        variation_key: str,
        retry_feedback: str | None = None,
    ) -> str:
        """Builds the base prompt with a UUID sample id as the first line."""

        base_prompt = super().build_prompt(
            task_definition=task_definition,
            runtime_config=runtime_config,
            task_instance=task_instance,
            variation_key=variation_key,
            retry_feedback=retry_feedback,
        )
        sample_id = uuid.uuid4()
        return f"Sample ID: {sample_id}\n\n Condition your generation based on this sample ID. \n\n {base_prompt}"


class RandomSamplingStrategy(RandomNumberSamplingStrategy):
    """Uses UUID prompt tagging under the public random sampling name."""

    name = "random"
