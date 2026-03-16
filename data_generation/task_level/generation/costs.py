"""Build retry-aware token and cost summaries for trajectory generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from data_generation.task_level.generation.config import (
    COST_SUMMARY_OUTPUT_FILENAME,
    DEFAULT_OUTPUT_DIR,
    RuntimeConfig,
)
from data_generation.task_level.generation.runtime_support import (
    _default_traffic_type_for_runtime,
    _sampling_strategy_for_runtime,
)
from data_generation.task_level.runtime.client import (
    COST_DECIMAL_PLACES,
    _resolve_pricing_tier,
    reprice_generation_usage,
)
from data_generation.task_level.tasks import TaskDefinition
from data_generation.utils import camel_to_snake_case, coerce_int, round_cost


def _successful_attempt_count(generation_usage: dict[str, Any]) -> int:
    successful_attempt_number = generation_usage.get("successful_attempt_number", 1)
    if not isinstance(successful_attempt_number, int) or successful_attempt_number < 1:
        return 1
    return successful_attempt_number


def _projected_attempt_count(runtime_config: RuntimeConfig) -> int:
    """Choose the attempt multiplier used for the single preflight projection."""

    if runtime_config.disable_validation:
        return 1
    return max(runtime_config.max_retries, 1)


def _attempt_counts_for_saved_trajectories(
    generation_usages: list[dict[str, Any]],
) -> list[int]:
    """Read the persisted successful attempt number for each saved trajectory."""

    return [
        (
            1
            if generation_usage.get("retry_costs_included") is True
            else _successful_attempt_count(generation_usage)
        )
        for generation_usage in generation_usages
    ]


def _reasoning_token_count(generation_usage: dict[str, Any]) -> int:
    """Read reasoning tokens from persisted usage, defaulting old payloads to zero."""
    return coerce_int(generation_usage.get("reasoning_tokens")) or 0


def _billable_output_token_count(generation_usage: dict[str, Any]) -> int:
    """Reasoning tokens share the standard output-token billing tier."""
    return generation_usage["output_tokens"] + _reasoning_token_count(generation_usage)


def _scaled_token_totals(
    generation_usages: list[dict[str, Any]],
    attempt_counts: list[int],
) -> dict[str, int]:
    return {
        "prompt": sum(
            generation_usage["prompt_tokens"] * attempt_count
            for generation_usage, attempt_count in zip(
                generation_usages, attempt_counts
            )
        ),
        "output": sum(
            generation_usage["output_tokens"] * attempt_count
            for generation_usage, attempt_count in zip(
                generation_usages, attempt_counts
            )
        ),
        "reasoning": sum(
            _reasoning_token_count(generation_usage) * attempt_count
            for generation_usage, attempt_count in zip(
                generation_usages, attempt_counts
            )
        ),
        "total": sum(
            generation_usage["total_tokens"] * attempt_count
            for generation_usage, attempt_count in zip(
                generation_usages, attempt_counts
            )
        ),
    }


def _scaled_total_cost(
    generation_usages: list[dict[str, Any]],
    attempt_counts: list[int],
) -> float | None:
    scaled_costs: list[float] = []
    for generation_usage, attempt_count in zip(generation_usages, attempt_counts):
        observed_cost_usd = generation_usage.get("observed_cost_usd")
        if observed_cost_usd is None:
            return None
        scaled_costs.append(observed_cost_usd * attempt_count)
    return sum(scaled_costs)


def _observed_cost_total(generation_usages: list[dict[str, Any]]) -> float | None:
    """Sums observed costs only when every usage payload includes one."""

    observed_costs = [
        generation_usage.get("observed_cost_usd")
        for generation_usage in generation_usages
    ]
    if any(observed_cost is None for observed_cost in observed_costs):
        return None
    return sum(observed_costs)


def _shared_pricing(
    generation_usages: list[dict[str, Any]],
    *,
    model: str | None = None,
) -> dict[str, Any] | None:
    if not generation_usages:
        return None

    resolved_pricings: list[dict[str, Any]] = []
    for generation_usage in generation_usages:
        pricing = generation_usage.get("pricing")
        if not isinstance(pricing, dict):
            if model is None:
                return None
            pricing_tier = _resolve_pricing_tier(
                model,
                generation_usage.get("traffic_type"),
            )
            if pricing_tier is None:
                return None
            pricing = {
                "model": pricing_tier.model,
                "input_usd_per_million_tokens": pricing_tier.input_usd_per_million_tokens,
                "output_usd_per_million_tokens": pricing_tier.output_usd_per_million_tokens,
            }
        resolved_pricings.append(pricing)

    first_pricing = resolved_pricings[0]
    if any(pricing != first_pricing for pricing in resolved_pricings[1:]):
        return None

    return dict(first_pricing)


def _best_case_cost_estimate_note(attempt_counts: list[int]) -> str:
    if any(attempt_count > 1 for attempt_count in attempt_counts):
        return (
            "Best case uses the observed successful attempt number for each "
            "saved trajectory and assumes earlier failed attempts had the "
            "same token profile as the successful attempt."
        )
    return "Best case assumes each trajectory succeeds on the first attempt."


def _append_sampling_cost_note(
    summary: dict[str, Any],
    runtime_config: RuntimeConfig,
) -> dict[str, Any]:
    """Adds sampling-specific cost notes when totals are shared across candidates."""

    if runtime_config.sampling != "verbalized":
        return summary
    verbalized_note = (
        "For verbalized sampling, token and cost totals are counted once per "
        "model response and apportioned across the saved trajectories from "
        "that response."
    )
    if verbalized_note not in summary["notes"]:
        summary["notes"].append(verbalized_note)
    return summary


def _load_json_payload(path: Path) -> dict[str, Any] | None:
    """Loads one JSON payload from disk when the file exists and is valid."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _matches_historical_preflight_config(
    payload: dict[str, Any],
    runtime_config: RuntimeConfig,
) -> bool:
    """Checks whether one saved run matches the active preflight settings."""

    if payload.get("model") != runtime_config.model:
        return False
    if payload.get("sdk") != runtime_config.sdk:
        return False

    model_config = payload.get("model_config")
    if not isinstance(model_config, dict):
        return False

    sampling_payload = model_config.get("sampling")
    if not isinstance(sampling_payload, dict):
        return False
    if sampling_payload.get("strategy", "base") != runtime_config.sampling:
        return False
    if runtime_config.sampling == "verbalized":
        if sampling_payload.get("verbalized_k") != runtime_config.verbalized_k:
            return False
    elif sampling_payload.get("verbalized_k", 1) != 1:
        return False
    if sampling_payload.get("temperature") != runtime_config.temperature:
        return False

    reasoning_payload = model_config.get("reasoning")
    if not isinstance(reasoning_payload, dict):
        return runtime_config.thinking_level is None
    return reasoning_payload.get("thinking_level") == runtime_config.thinking_level


def _combine_split_generation_usages(
    usage_entries: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Reassembles one run-level usage record from per-candidate saved usage."""

    if not usage_entries:
        return None

    successful_attempt_number = _successful_attempt_count(usage_entries[0])
    if any(
        _successful_attempt_count(usage_entry) != successful_attempt_number
        for usage_entry in usage_entries[1:]
    ):
        return None

    combined_usage = dict(usage_entries[0])
    combined_usage["successful_attempt_number"] = successful_attempt_number
    combined_usage["prompt_tokens"] = sum(
        usage_entry["prompt_tokens"] for usage_entry in usage_entries
    )
    combined_usage["output_tokens"] = sum(
        usage_entry["output_tokens"] for usage_entry in usage_entries
    )
    combined_usage["reasoning_tokens"] = sum(
        _reasoning_token_count(usage_entry) for usage_entry in usage_entries
    )
    combined_usage["total_tokens"] = (
        combined_usage["prompt_tokens"]
        + combined_usage["output_tokens"]
        + combined_usage["reasoning_tokens"]
    )
    combined_usage["observed_cost_usd"] = _observed_cost_total(usage_entries)
    return combined_usage


def _historical_preflight_generation_usages(
    runtime_config: RuntimeConfig,
    task_definition: TaskDefinition,
) -> list[dict[str, Any]]:
    """Loads prior matching run usages to calibrate preflight estimates."""

    if runtime_config.summary_path is None:
        return []

    current_run_dir = runtime_config.summary_path.parent
    task_output_directory_name = camel_to_snake_case(task_definition.composite_task)
    trajectories_per_run = _sampling_strategy_for_runtime(
        runtime_config
    ).trajectories_per_run(runtime_config)
    matched_generation_usages: list[dict[str, Any]] = []

    candidate_cost_summary_paths = sorted(
        DEFAULT_OUTPUT_DIR.rglob(COST_SUMMARY_OUTPUT_FILENAME),
        reverse=True,
    )
    for cost_summary_path in candidate_cost_summary_paths:
        parent_directory = cost_summary_path.parent
        grandparent_directory = parent_directory.parent
        if (
            parent_directory.name != task_output_directory_name
            and grandparent_directory.name != task_output_directory_name
        ):
            continue
        if cost_summary_path.parent == current_run_dir:
            continue
        payload = _load_json_payload(cost_summary_path)
        if payload is None or not _matches_historical_preflight_config(
            payload,
            runtime_config,
        ):
            continue

        trajectory_costs = payload.get("trajectory_costs")
        if not isinstance(trajectory_costs, list) or not trajectory_costs:
            continue

        for index in range(0, len(trajectory_costs), trajectories_per_run):
            trajectory_group = trajectory_costs[index : index + trajectories_per_run]
            if len(trajectory_group) != trajectories_per_run:
                continue
            usage_entries = [
                trajectory_cost.get("generation_usage")
                for trajectory_cost in trajectory_group
                if isinstance(trajectory_cost, dict)
            ]
            if any(not isinstance(usage_entry, dict) for usage_entry in usage_entries):
                continue
            combined_usage = _combine_split_generation_usages(usage_entries)
            if combined_usage is not None:
                matched_generation_usages.append(
                    reprice_generation_usage(
                        combined_usage,
                        model=runtime_config.model,
                        traffic_type=combined_usage.get("traffic_type")
                        or _default_traffic_type_for_runtime(runtime_config),
                        round_observed_cost=False,
                    )
                )

    return matched_generation_usages


def _projected_generation_usages_from_history(
    observed_generation_usages: list[dict[str, Any]],
    *,
    num_runs: int,
) -> list[dict[str, Any]]:
    """Repeats the latest observed run profile until the requested run count is filled."""

    return [
        dict(observed_generation_usages[index % len(observed_generation_usages)])
        for index in range(num_runs)
    ]


def _build_cost_estimate_summary_from_generation_usages(
    generation_usages: list[dict[str, Any]],
    runtime_config: RuntimeConfig,
) -> dict[str, Any]:
    # Post-run estimates should honor how many attempts each saved trajectory took.
    best_case_attempt_counts = _attempt_counts_for_saved_trajectories(generation_usages)
    worst_case_attempt_counts = [runtime_config.max_retries for _ in generation_usages]

    best_case_tokens = _scaled_token_totals(
        generation_usages,
        best_case_attempt_counts,
    )
    worst_case_tokens = _scaled_token_totals(
        generation_usages,
        worst_case_attempt_counts,
    )
    best_case_total_cost = _scaled_total_cost(
        generation_usages,
        best_case_attempt_counts,
    )
    worst_case_total_cost = _scaled_total_cost(
        generation_usages,
        worst_case_attempt_counts,
    )
    shared_pricing = _shared_pricing(
        generation_usages,
        model=runtime_config.model,
    )

    summary = {
        "best_case_total_usd": round_cost(
            best_case_total_cost,
            decimal_places=COST_DECIMAL_PLACES,
        ),
        "best_case_per_trajectory_usd": round_cost(
            best_case_total_cost / max(len(generation_usages), 1)
            if best_case_total_cost is not None
            else None,
            decimal_places=COST_DECIMAL_PLACES,
        ),
        "worst_case_total_usd": round_cost(
            worst_case_total_cost,
            decimal_places=COST_DECIMAL_PLACES,
        ),
        "worst_case_per_trajectory_usd": round_cost(
            worst_case_total_cost / max(len(generation_usages), 1)
            if worst_case_total_cost is not None
            else None,
            decimal_places=COST_DECIMAL_PLACES,
        ),
        "best_case_tokens": best_case_tokens,
        "worst_case_tokens": worst_case_tokens,
        "max_retries_assumed_for_worst_case": runtime_config.max_retries,
        "notes": [
            _best_case_cost_estimate_note(best_case_attempt_counts),
            "Worst case assumes every trajectory consumes the full retry budget and each attempt has the same token profile as the successful attempt.",
            "Token counts use Vertex usage metadata when available and otherwise fall back to a local character-based estimate.",
        ],
    }
    if shared_pricing is not None:
        summary["pricing"] = shared_pricing
    return _append_sampling_cost_note(summary, runtime_config)


def _build_cost_summary_from_generation_usages(
    generation_usages: list[dict[str, Any]],
) -> dict[str, Any]:
    # Scale saved trajectories by their successful attempt number so totals include
    # the retry work required to produce each persisted output.
    trajectory_count = len(generation_usages)
    attempt_counts = _attempt_counts_for_saved_trajectories(generation_usages)
    scaled_tokens = _scaled_token_totals(
        generation_usages,
        attempt_counts,
    )
    total_prompt_tokens = scaled_tokens["prompt"]
    total_output_tokens = scaled_tokens["output"]
    total_reasoning_tokens = scaled_tokens["reasoning"]
    total_tokens = scaled_tokens["total"]
    shared_pricing = _shared_pricing(generation_usages)
    pricing_supported = all("pricing" in usage for usage in generation_usages)

    input_cost = None
    output_cost = None
    total_cost = None
    if pricing_supported:
        # Keep the cost fields aligned with the retry-inclusive token totals.
        input_cost = sum(
            (usage["prompt_tokens"] * attempt_count / 1_000_000)
            * usage["pricing"]["input_usd_per_million_tokens"]
            for usage, attempt_count in zip(generation_usages, attempt_counts)
        )
        output_cost = sum(
            (_billable_output_token_count(usage) * attempt_count / 1_000_000)
            * usage["pricing"]["output_usd_per_million_tokens"]
            for usage, attempt_count in zip(generation_usages, attempt_counts)
        )
        total_cost = input_cost + output_cost

    summary = {
        "prompt_tokens": total_prompt_tokens,
        "output_tokens": total_output_tokens,
        "reasoning_tokens": total_reasoning_tokens,
        "total_tokens": total_tokens,
        "input_cost_usd": round_cost(
            input_cost,
            decimal_places=COST_DECIMAL_PLACES,
        ),
        "output_cost_usd": round_cost(
            output_cost,
            decimal_places=COST_DECIMAL_PLACES,
        ),
        "total_cost_usd": round_cost(
            total_cost,
            decimal_places=COST_DECIMAL_PLACES,
        ),
        "average_trajectory_cost_usd": round_cost(
            total_cost / trajectory_count
            if total_cost is not None and trajectory_count > 0
            else None,
            decimal_places=COST_DECIMAL_PLACES,
        ),
        "notes": [
            "Cost summary scales each saved trajectory by its successful attempt number.",
            "Retry-inclusive totals assume earlier failed attempts used the same token profile as the successful attempt.",
        ],
    }
    if shared_pricing is not None:
        summary["pricing"] = shared_pricing
    return summary


def _build_preflight_cost_estimate_summary(
    runtime_config: RuntimeConfig,
    task_definition: TaskDefinition,
) -> dict[str, Any]:
    historical_generation_usages = _historical_preflight_generation_usages(
        runtime_config,
        task_definition,
    )
    if historical_generation_usages:
        summary = _build_cost_estimate_summary_from_generation_usages(
            _projected_generation_usages_from_history(
                historical_generation_usages,
                num_runs=runtime_config.num_runs,
            ),
            runtime_config=runtime_config,
        )
        summary["notes"][0] = (
            "Projected total uses observed API usage from prior matching runs "
            "for this task, model, and sampling configuration."
        )
        summary["notes"][2] = (
            "Token counts come from previously saved API usage metadata for "
            "matching runs."
        )
        summary["notes"].append(
            f"Matched {len(historical_generation_usages)} prior run(s) from "
            "saved cost summaries."
        )
        return summary

    manual_estimate = _sampling_strategy_for_runtime(
        runtime_config
    ).preflight_token_estimate(
        task_definition=task_definition,
        runtime_config=runtime_config,
    )
    generation_usage = reprice_generation_usage(
        {
            "successful_attempt_number": _projected_attempt_count(runtime_config),
            "prompt_tokens": manual_estimate.prompt_tokens,
            "output_tokens": manual_estimate.output_tokens,
            "reasoning_tokens": manual_estimate.reasoning_tokens,
            "total_tokens": (
                manual_estimate.prompt_tokens
                + manual_estimate.output_tokens
                + manual_estimate.reasoning_tokens
            ),
            "usage_source": "manual_task_estimate",
            "traffic_type": _default_traffic_type_for_runtime(runtime_config),
            "observed_cost_usd": 0.0,
        },
        model=runtime_config.model,
        traffic_type=_default_traffic_type_for_runtime(runtime_config),
        round_observed_cost=False,
    )
    summary = _build_cost_estimate_summary_from_generation_usages(
        [generation_usage] * runtime_config.num_runs,
        runtime_config=runtime_config,
    )
    if _projected_attempt_count(runtime_config) > 1:
        summary["notes"][0] = (
            "Projected total uses the manual task token estimate maintained in the "
            "task definition and scales it by the configured retry budget."
        )
    else:
        summary["notes"][0] = (
            "Best case uses the manual task token estimate maintained in the "
            "task definition."
        )
    summary["notes"][2] = (
        "Token counts come from the manual task token estimate rather than "
        "observed API usage metadata."
    )
    return _append_sampling_cost_note(summary, runtime_config)
