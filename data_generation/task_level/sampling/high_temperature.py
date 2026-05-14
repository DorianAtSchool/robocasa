"""Expose base prompting under the high-temperature sampling label."""

from __future__ import annotations

from data_generation.task_level.sampling.base import BaseSamplingStrategy


class HighTemperatureSamplingStrategy(BaseSamplingStrategy):
    """Uses base sampling behavior while recording a high-temperature strategy."""

    name = "high_temperature"
