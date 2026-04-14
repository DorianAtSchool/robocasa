"""Few-shot example loader for Phase 1 spec generation.

Selects a small set of existing manually-built TaskSpec JSONs together with
their source Python files. The goal is to expose the LLM to the variety of
patterns it will need to handle: a single-fixture parallel task, a
multi-fixture flexible-assignment task, and a multi-fixture task with a
hinged-part dependency and a machine-flag effect.
"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path

from data_generation.task_level.tasks import specs as specs_pkg


@dataclass(frozen=True)
class FewShotExample:
    """One few-shot example: source Python + JSON spec."""

    task_name: str
    source_python: str
    spec_json: str

    @property
    def short_label(self) -> str:
        return self.task_name


# Curated examples covering distinct patterns. Order matters: the prompt
# always shows them in this order so the model anchors on the simplest
# pattern first and generalizes outward.
_CURATED_EXAMPLES: tuple[tuple[str, str], ...] = (
    # (composite_task, spec_json_filename)
    ("MeatSkewerAssembly", "meatskewerassembly.json"),
    ("SetBowlsForSoup", "setbowlsforsoup.json"),
    ("GarnishCupcake", "garnishcupcake.json"),
)


_SPECS_DIR = Path(specs_pkg.__file__).resolve().parent


def _resolve_source_python_path(spec_payload: dict) -> Path:
    """Find the source .py file referenced by a spec's source_python_module."""

    module_name = spec_payload["source_python_module"]
    spec = importlib.util.find_spec(module_name)
    if spec is None or not spec.origin:
        raise FileNotFoundError(
            f"Could not locate source module {module_name!r} "
            f"for few-shot example."
        )
    return Path(spec.origin)


def _load_one_example(spec_filename: str) -> FewShotExample:
    spec_path = _SPECS_DIR / spec_filename
    spec_text = spec_path.read_text(encoding="utf-8")
    spec_payload = json.loads(spec_text)
    source_path = _resolve_source_python_path(spec_payload)
    source_text = source_path.read_text(encoding="utf-8")
    return FewShotExample(
        task_name=str(spec_payload["composite_task"]),
        source_python=source_text,
        spec_json=spec_text,
    )


def load_few_shot_examples() -> tuple[FewShotExample, ...]:
    """Load all curated few-shot examples in their canonical order.

    The set is small (3) so callers can include all of them in the prompt.
    Token cost is bounded because spec JSONs are small relative to model
    context windows. Returns a tuple so it is safe to cache.
    """

    return tuple(_load_one_example(filename) for _, filename in _CURATED_EXAMPLES)
