"""Expose task-level sampling strategies."""

from __future__ import annotations

from data_generation.task_level.runtime.client import TrajectoryGenerationError
from data_generation.task_level.sampling.base import (
    BaseSamplingStrategy,
    SampledTrajectoryCandidate,
    SamplingStrategy,
)
from data_generation.task_level.sampling.high_temperature import (
    HighTemperatureSamplingStrategy,
)
from data_generation.task_level.sampling.random_number import (
    RandomNumberSamplingStrategy,
    RandomSamplingStrategy,
)
from data_generation.task_level.sampling.verbalized import (
    VerbalizedSamplingStrategy,
    VerbalizedSamplingValidationError,
)


SAMPLING_STRATEGIES: dict[str, SamplingStrategy] = {
    "base": BaseSamplingStrategy(),
    "high_temperature": HighTemperatureSamplingStrategy(),
    "random": RandomSamplingStrategy(),
    "random_number": RandomNumberSamplingStrategy(),
    "verbalized": VerbalizedSamplingStrategy(),
}


def get_sampling_strategy(name: str) -> SamplingStrategy:
    """Returns the requested sampling strategy or raises a config error."""

    strategy = SAMPLING_STRATEGIES.get(name)
    if strategy is None:
        supported_names = ", ".join(sorted(SAMPLING_STRATEGIES))
        raise TrajectoryGenerationError(
            f"Unsupported sampling strategy '{name}'. Available strategies: {supported_names}."
        )
    return strategy


__all__ = [
    "BaseSamplingStrategy",
    "HighTemperatureSamplingStrategy",
    "RandomSamplingStrategy",
    "RandomNumberSamplingStrategy",
    "SampledTrajectoryCandidate",
    "SAMPLING_STRATEGIES",
    "SamplingStrategy",
    "VerbalizedSamplingStrategy",
    "VerbalizedSamplingValidationError",
    "get_sampling_strategy",
]
