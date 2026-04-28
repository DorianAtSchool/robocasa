"""Few-shot example loader for Phase 1 spec generation.

Selects a small set of existing manually-built TaskSpec JSONs together with
their source Python files. The goal is to expose the LLM to a few distinct
patterns it will need to handle: a simple retrieval-and-placement task, a
multi-fixture cooperative staging task, and a machine-trigger task with a
hinged-part dependency.
"""

from __future__ import annotations

import importlib
import json
from dataclasses import asdict, dataclass

from data_generation.task_level.tasks.specs import load_task_spec


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
_CURATED_EXAMPLES: tuple[str, ...] = (
    "PrepareSausageCheese",
    "PrepareSandwichStation",
    "PrepareCoffee",
)


def _resolve_source_python_path(module_name: str):
    """Find the source .py file referenced by a spec's source_python_module."""

    spec = importlib.util.find_spec(module_name)
    if spec is None or not spec.origin:
        raise FileNotFoundError(
            f"Could not locate source module {module_name!r} "
            f"for few-shot example."
        )
    return spec.origin


def _load_one_example(task_name: str) -> FewShotExample:
    spec_payload = load_task_spec(task_name)
    source_path = _resolve_source_python_path(spec_payload.source_python_module)
    with open(source_path, "r", encoding="utf-8") as handle:
        source_text = handle.read()
    spec_dict = asdict(spec_payload)
    allowed_tool_specs = spec_dict.get("allowed_tool_specs")
    if isinstance(allowed_tool_specs, dict):
        for tool_name, tool_spec in list(allowed_tool_specs.items()):
            if not isinstance(tool_spec, dict):
                continue
            allowed_tool_specs[tool_name] = {
                key: value
                for key, value in tool_spec.items()
                if key
                not in {
                    "description",
                    "tool_args",
                    "optional_tool_args",
                    "tool_arg_types",
                    "tool_arg_any_of",
                }
            }
    spec_text = json.dumps(spec_dict, indent=2)
    return FewShotExample(
        task_name=spec_payload.composite_task,
        source_python=source_text,
        spec_json=spec_text,
    )


def load_few_shot_examples() -> tuple[FewShotExample, ...]:
    """Load all curated few-shot examples in their canonical order.

    The set is small (3) so callers can include all of them in the prompt.
    Token cost is bounded because spec JSONs are small relative to model
    context windows. Returns a tuple so it is safe to cache.
    """

    return tuple(_load_one_example(task_name) for task_name in _CURATED_EXAMPLES)
