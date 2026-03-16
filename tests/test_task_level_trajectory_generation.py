from copy import deepcopy
import json
import runpy
import sys
import threading
import unittest
from unittest import mock
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
import tempfile
import types

import data_generation.task_level.generation.cli as trajectory_generation_module
from data_generation.task_level.runtime.client import (
    GenerationResult,
    GenerationUsage,
    GoogleGenAIClient,
    TrajectoryGenerationError,
    _generation_error_status_code,
    _resolve_pricing_tier,
    load_dotenv_file,
    validate_google_auth,
)
from data_generation.task_level.sampling.base import BaseSamplingStrategy
from data_generation.task_level.sampling.verbalized import (
    VerbalizedSamplingStrategy,
    VerbalizedSamplingValidationError,
)
from data_generation.task_level.subatomic_tool_specs import build_allowed_tool_specs
from data_generation.task_level.tasks.base import (
    FiniteStateTaskValidator,
    TaskInstance,
    build_task_response_schema,
)
from data_generation.task_level.tasks import (
    HeldObjectSemanticValidationError,
    HOT_DOG_SETUP_TASK,
    HotDogSetupValidator,
    InsufficientValidUniqueTrajectoriesDuplicateError,
    InsufficientValidUniqueTrajectoriesInvalidError,
    InsufficientValidUniqueTrajectoriesMixedError,
    MissingInitialCommunicationSemanticValidationError,
    NavigationSemanticValidationError,
    ObservationSequenceSemanticValidationError,
    DuplicateTrajectoryValidationError,
    PREPARE_COFFEE_TASK,
    ResponseFormatValidationError,
    TaskSemanticValidationError,
    TaskPreconditionSemanticValidationError,
    TrajectoryStructureValidationError,
    TrajectoryValidationError,
    ToolArgumentSemanticValidationError,
    WaitDurationSemanticValidationError,
)
from data_generation.task_level.tasks.hot_dog_setup import (
    HOT_DOG_SETUP_INITIAL_STATE,
)
from data_generation.task_level.tasks.prepare_coffee import (
    PREPARE_COFFEE_ALLOWED_TOOL_SPECS,
    PREPARE_COFFEE_NON_COMMUNICATE_TOOL_NAMES,
    PREPARE_COFFEE_INITIAL_STATE,
    PrepareCoffeeValidator,
    build_prepare_coffee_prompt,
)
from data_generation.task_level.subatomic_tool_calls import discover_subatomic_tools
from data_generation.task_level.runtime.batch_generation import (
    BatchRunContext,
    RichBatchProgressDisplay,
    _batch_request_payload,
    _create_batch_progress_handles,
)
from data_generation.task_level.generation.cli import (
    main,
    parse_args,
    run_cli,
)
from data_generation.task_level.generation.config import (
    BATCH_INTERRUPTED_MESSAGE,
    DEFAULT_OUTPUT_DIR,
    INTERRUPTED_EXIT_CODE,
    INTERRUPTED_MESSAGE,
    REQUEST_DIRECTORY_NAME,
    RuntimeConfig,
)
from data_generation.task_level.generation.costs import (
    _build_cost_summary_from_generation_usages,
    _build_preflight_cost_estimate_summary,
)
from data_generation.task_level.generation.orchestrator import (
    generate_single_trajectory,
    generate_trajectories,
)
from data_generation.task_level.generation.outputs import (
    build_cost_output_payload,
    build_error_summary_output_payload,
    build_summary_output_payload,
    resolve_cost_output_path,
    resolve_dataset_output_path,
    resolve_error_output_path,
    resolve_prompt_output_dir,
    resolve_raw_output_dir,
    resolve_request_output_path,
    resolve_trajectory_output_dir,
)
from data_generation.task_level.generation.progress import (
    PROGRESS_BAR_WIDTH,
    ProgressHandles,
    RichProgressDisplay,
    RichTaskProgressAdapter,
    TQDM_BAR_FORMAT,
    _progress_status_with_accumulated_cost,
    _validation_error_progress_summary,
)
from data_generation.task_level.generation.runtime_support import (
    _build_retry_feedback_text,
    _is_non_retryable_generation_error,
    _maybe_reserve_signature,
    extract_json_candidate,
)


PREPARE_COFFEE_ACTION_SPECS = (
    ("navigate_to_fixture", {"fixture_id": "cabinet_1"}),
    ("open_hinged_part", {"target_id": "cabinet_1", "part_id": "door"}),
    ("pick_up_object", {"object_id": "mug_1", "source_id": "cabinet_1"}),
    ("navigate_to_fixture", {"fixture_id": "counter_1"}),
    ("place_on_surface", {"object_id": "mug_1", "support_id": "counter_1"}),
    ("navigate_to_fixture", {"fixture_id": "counter_1"}),
    ("pick_up_object", {"object_id": "mug_1", "source_id": "counter_1"}),
    ("navigate_to_fixture", {"fixture_id": "coffee_machine_1"}),
    (
        "place_under_dispenser",
        {"object_id": "mug_1", "dispenser_id": "coffee_machine_dispenser"},
    ),
    ("press_button", {"target_id": "coffee_machine_1", "control_id": "start_button"}),
)

HOT_DOG_SETUP_ACTION_SPECS = (
    ("navigate_to_fixture", {"fixture_id": "counter_1"}),
    ("pick_up_object", {"object_id": "hotdog_bun_1", "source_id": "counter_1"}),
    ("navigate_to_fixture", {"fixture_id": "dining_table_1"}),
    ("place_on_object", {"object_id": "hotdog_bun_1", "support_object_id": "plate_1"}),
    ("navigate_to_fixture", {"fixture_id": "cabinet_1"}),
    ("pick_up_object", {"object_id": "condiment_1", "source_id": "cabinet_1"}),
    ("navigate_to_fixture", {"fixture_id": "dining_table_1"}),
    ("place_next_to", {"object_id": "condiment_1", "reference_object_id": "plate_1"}),
    ("navigate_to_fixture", {"fixture_id": "fridge_1"}),
    ("pick_up_object", {"object_id": "sausage_1", "source_id": "fridge_1"}),
    ("navigate_to_fixture", {"fixture_id": "dining_table_1"}),
    ("place_on_object", {"object_id": "sausage_1", "support_object_id": "plate_1"}),
)


def renumber_candidate_steps(candidate):
    """Reassigns contiguous step indexes after a test mutates a candidate."""

    for index, step in enumerate(candidate["steps"]):
        step["step"] = index
    return candidate


def make_required_get_image_step(
    agent_id,
    *,
    reasoning,
    camera_view="wrist",
):
    """Builds a compact get_image step used to frame task actions."""

    return {
        "step": -1,
        "agent": agent_id,
        "tool": "get_image",
        "args": {"camera_view": camera_view},
        "reasoning": reasoning,
    }


def frame_action_steps_with_get_images(action_steps):
    """Places the required get_image calls before and after task actions."""

    if not action_steps:
        return []

    framed_steps = []
    for action_step in action_steps:
        framed_steps.append(
            make_required_get_image_step(
                action_step["agent"],
                reasoning="I should inspect before the next task action.",
            )
        )
        framed_steps.append(action_step)
    framed_steps.append(
        make_required_get_image_step(
            action_steps[-1]["agent"],
            reasoning="I should capture the completed setup.",
        )
    )
    return framed_steps


def find_step_index(candidate, tool_name, *, occurrence=0):
    """Finds the zero-based index of the requested tool occurrence."""

    matches = [
        index
        for index, step in enumerate(candidate["steps"])
        if step["tool"] == tool_name
    ]
    if occurrence >= len(matches):
        raise AssertionError(
            f"Could not find occurrence {occurrence} of tool {tool_name}."
        )
    return matches[occurrence]


def make_valid_candidate(
    *,
    communicate_messages=("I will grab the mug.", "I will be ready at the machine."),
    action_agents=("agent_0", "agent_1", "agent_1"),
    action_reasoning=(
        "The mug starts in the cabinet.",
        "The mug must reach the dispenser next.",
        "The mug is ready for brewing.",
    ),
    include_agents=True,
):
    if len(action_agents) == 3:
        # Expand the original three-phase fixture into the shared subatomic steps.
        expanded_action_agents = (
            (action_agents[0],) * 5
            + (action_agents[1],) * 4
            + (action_agents[2],)
        )
    else:
        expanded_action_agents = tuple(action_agents)

    if len(action_reasoning) == 3:
        # Reuse the original three reasoning beats across the decomposed action groups.
        expanded_action_reasoning = (
            (action_reasoning[0],) * 5
            + (action_reasoning[1],) * 4
            + (action_reasoning[2],)
        )
    else:
        expanded_action_reasoning = tuple(action_reasoning)

    action_steps = [
        {
            "step": -1,
            "agent": agent_id,
            "tool": tool_name,
            "args": dict(tool_args),
            "reasoning": reasoning,
        }
        for (tool_name, tool_args), agent_id, reasoning in zip(
            PREPARE_COFFEE_ACTION_SPECS,
            expanded_action_agents,
            expanded_action_reasoning,
        )
    ]

    candidate = {
        "steps": [
            {
                "step": 0,
                "agent": "agent_0",
                "tool": "communicate",
                "args": {
                    "to": "agent_1",
                    "message": communicate_messages[0],
                },
                "reasoning": "We need a shared plan before acting.",
            },
            {
                "step": 1,
                "agent": "agent_1",
                "tool": "communicate",
                "args": {
                    "to": "agent_0",
                    "message": communicate_messages[1],
                },
                "reasoning": "I should confirm the handoff sequence.",
            },
            *action_steps,
        ],
    }
    if include_agents:
        candidate["agents"] = [
            {"agent": "agent_0"},
            {"agent": "agent_1"},
        ]
    return renumber_candidate_steps(candidate)


def make_valid_hot_dog_setup_candidate(*, include_agents=True):
    """Builds a valid HotDogSetup candidate trajectory for validator tests."""

    action_agents = (
        ("agent_0",) * 4
        + ("agent_1",) * 8
    )
    action_reasoning = (
        ("The bun starts on the counter.",) * 4
        + ("The condiment should be moved beside the plate.",) * 4
        + ("The sausage still needs to be added to the plate.",) * 4
    )
    action_steps = [
        {
            "step": -1,
            "agent": agent_id,
            "tool": tool_name,
            "args": dict(tool_args),
            "reasoning": reasoning,
        }
        for (tool_name, tool_args), agent_id, reasoning in zip(
            HOT_DOG_SETUP_ACTION_SPECS,
            action_agents,
            action_reasoning,
        )
    ]
    candidate = {
        "steps": [
            {
                "step": 0,
                "agent": "agent_0",
                "tool": "communicate",
                "args": {
                    "to": "agent_1",
                    "message": "I will start with the bun.",
                },
                "reasoning": "We need a shared setup plan first.",
            },
            {
                "step": 1,
                "agent": "agent_1",
                "tool": "communicate",
                "args": {
                    "to": "agent_0",
                    "message": "I will handle the condiment and sausage.",
                },
                "reasoning": "I should confirm the remaining ingredients.",
            },
            *action_steps,
        ],
    }
    if include_agents:
        candidate["agents"] = [
            {"agent": "agent_0"},
            {"agent": "agent_1"},
        ]
    return renumber_candidate_steps(candidate)


def make_invalid_candidate_missing_initial_communication():
    candidate = make_valid_candidate()
    candidate["steps"] = candidate["steps"][1:]
    return renumber_candidate_steps(candidate)


def make_alternative_valid_candidate():
    """Builds a valid PrepareCoffee trace with a different legal action ordering."""

    candidate = make_valid_candidate()
    insert_index = find_step_index(candidate, "navigate_to_fixture", occurrence=0)
    candidate["steps"].insert(
        insert_index,
        {
            "step": -1,
            "agent": "agent_1",
            "tool": "navigate_to_fixture",
            "args": {
                "fixture_id": "coffee_machine_1",
            },
            "reasoning": "I can stage at the machine before the mug arrives.",
        },
    )
    return renumber_candidate_steps(candidate)


TOY_FSM_INITIAL_STATE = {
    "agents": {
        "agent_0": {
            "location": "staging_area",
            "held_object": None,
        },
        "agent_1": {
            "location": "staging_area",
            "held_object": None,
        },
    },
    "objects": {
        "apple_1": {
            "location": "table_1",
        },
        "mug_1": {
            "location": "table_1",
        },
    },
    "fixtures": {
        "table_1": {"fixture_type": "counter"},
        "shelf_1": {"fixture_type": "counter"},
        "cabinet_1": {
            "fixture_type": "cabinet",
            "parts": {
                "door": {
                    "part_type": "hinged_part",
                    "state": "closed",
                }
            },
        },
    },
    "machine_state": {},
}

TOY_FSM_ALLOWED_TOOL_SPECS = build_allowed_tool_specs(
    (
        "communicate",
        "get_image",
        "navigate_to_fixture",
        "open_hinged_part",
        "pick_up_object",
        "place_on_surface",
        "wait",
    )
)

PLACEMENT_REFERENCE_INITIAL_STATE = {
    "agents": {
        "agent_0": {
            "location": "staging_area",
            "held_object": None,
        },
        "agent_1": {
            "location": "staging_area",
            "held_object": None,
        },
    },
    "objects": {
        "apple_1": {
            "location": "table_1",
        },
        "cup_1": {
            "location": "table_1",
        },
        "mug_1": {
            "location": "shelf_1",
        },
    },
    "fixtures": {
        "table_1": {"fixture_type": "counter"},
        "shelf_1": {"fixture_type": "counter"},
        "coffee_machine_1": {"fixture_type": "coffee_machine"},
    },
    "machine_state": {
        "coffee_machine_1": {
            "dispenser_id": "coffee_machine_dispenser",
        }
    },
}

PLACEMENT_REFERENCE_ALLOWED_TOOL_SPECS = build_allowed_tool_specs(
    (
        "communicate",
        "get_image",
        "navigate_to_fixture",
        "pick_up_object",
        "place_next_to",
        "place_under",
    )
)


def make_toy_action_spec(
    tool_name,
    tool_args,
):
    """Builds a compact toy action payload for shared FSM unit tests."""

    return {
        "tool": tool_name,
        "args": dict(tool_args),
    }


def make_toy_candidate(
    action_sequence,
    *,
    action_agents=None,
):
    """Builds a toy candidate that exercises only the shared FSM rules."""

    if action_agents is None:
        action_agents = ("agent_0",) * len(action_sequence)

    steps = [
        {
            "step": 0,
            "agent": "agent_0",
            "tool": "communicate",
            "args": {
                "to": "agent_1",
                "message": "I will handle the shared FSM test actions.",
            },
            "reasoning": "We should coordinate before the first action.",
        },
        {
            "step": 1,
            "agent": "agent_1",
            "tool": "communicate",
            "args": {
                "to": "agent_0",
                "message": "I will stay clear while you execute the sequence.",
            },
            "reasoning": "I should confirm the test setup.",
        },
    ]

    action_steps = [
        {
            "step": -1,
            "agent": agent_id,
            "tool": action_spec["tool"],
            "args": dict(action_spec["args"]),
            "reasoning": f"Shared FSM test action {index + 1}.",
        }
        for index, (action_spec, agent_id) in enumerate(
            zip(action_sequence, action_agents)
        )
    ]
    steps.extend(frame_action_steps_with_get_images(action_steps))

    return renumber_candidate_steps(
        {
        "agents": [
            {"agent": "agent_0"},
            {"agent": "agent_1"},
        ],
        "steps": steps,
        }
    )


class ToyFiniteStateValidator(FiniteStateTaskValidator):
    """Wraps the shared FSM with a tiny test-only task configuration."""

    def __init__(self):
        """Injects the toy task config used by generic FSM unit tests."""

        super().__init__(
            composite_task="ToyFSMTask",
            agent_ids=("agent_0", "agent_1"),
            initial_state=TOY_FSM_INITIAL_STATE,
            allowed_tool_specs=TOY_FSM_ALLOWED_TOOL_SPECS,
        )

    def is_goal_state_satisfied(self, runtime_state):
        """Checks the toy goal with a simple boolean state predicate."""

        return runtime_state.objects["apple_1"]["location"] == "shelf_1"


class PlacementReferenceValidator(FiniteStateTaskValidator):
    """Exercises placement tools that resolve destinations through references."""

    def __init__(self):
        """Configures a small FSM task for placement reference tests."""

        super().__init__(
            composite_task="PlacementReferenceTask",
            agent_ids=("agent_0", "agent_1"),
            initial_state=PLACEMENT_REFERENCE_INITIAL_STATE,
            allowed_tool_specs=PLACEMENT_REFERENCE_ALLOWED_TOOL_SPECS,
        )

    def is_goal_state_satisfied(self, runtime_state):
        """Checks that both reference-based placements land in the right location."""

        return (
            runtime_state.objects["apple_1"]["location"] == "shelf_1"
            and runtime_state.objects["cup_1"]["location"] == "coffee_machine_dispenser"
        )


class FakeClient:
    def generate(
        self,
        *,
        model,
        prompt,
        response_schema,
        temperature,
        thinking_level=None,
    ):
        raise NotImplementedError


class SequencedFakeClient(FakeClient):
    """Returns pre-seeded responses in order for retry and progress tests."""

    def __init__(self, responses):
        self._responses = list(responses)

    def generate(
        self,
        *,
        model,
        prompt,
        response_schema,
        temperature,
        thinking_level=None,
    ):
        if not self._responses:
            raise AssertionError("No more fake responses configured.")
        return self._responses.pop(0)


class PromptCapturingSequencedFakeClient(SequencedFakeClient):
    """Records prompts so retry tests can assert on prompt contents."""

    def __init__(self, responses):
        super().__init__(responses)
        self.prompts = []

    def generate(
        self,
        *,
        model,
        prompt,
        response_schema,
        temperature,
        thinking_level=None,
    ):
        self.prompts.append(prompt)
        return super().generate(
            model=model,
            prompt=prompt,
            response_schema=response_schema,
            temperature=temperature,
            thinking_level=thinking_level,
        )


class FakeGoogleGenAIClientError(Exception):
    pass


def make_batch_job(name, *, state="JOB_STATE_SUCCEEDED", error=None):
    return types.SimpleNamespace(
        name=name,
        state=types.SimpleNamespace(value=state),
        error=error,
    )


def make_batch_usage_metadata(
    *,
    prompt_tokens=1000,
    candidate_tokens=200,
    thoughts_tokens=50,
    total_tokens=1250,
):
    return {
        "promptTokenCount": prompt_tokens,
        "candidatesTokenCount": candidate_tokens,
        "thoughtsTokenCount": thoughts_tokens,
        "totalTokenCount": total_tokens,
    }


def make_batch_output_row(
    variation_key,
    *,
    candidate=None,
    status="",
    usage_metadata=None,
):
    run_index = int(variation_key.split("-")[1])
    row = {
        "variation_key": variation_key,
        "request": {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": build_prepare_coffee_prompt(
                                variation_key,
                                task_instance=PREPARE_COFFEE_TASK.build_task_instance(run_index),
                            )
                        }
                    ],
                }
            ]
        },
        "status": status,
    }
    if candidate is not None:
        row["response"] = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": json.dumps(candidate)}],
                    }
                }
            ]
        }
        if usage_metadata is not None:
            row["response"]["usageMetadata"] = usage_metadata
    return row


def make_batch_download(blob_uri, rows):
    return [
        (
            blob_uri,
            "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        )
    ]


def make_verbalized_response(
    *candidates,
    probabilities=None,
    as_json_string=False,
):
    """Builds a verbalized multi-trajectory payload for tests."""

    if probabilities is None:
        probabilities = [0.5 for _ in candidates]
    payload = {
        "responses": [
            {
                "probability": probability,
                "trajectory": candidate,
            }
            for probability, candidate in zip(probabilities, candidates)
        ]
    }
    if as_json_string:
        return json.dumps(payload)
    return payload


def make_prepare_coffee_task_instance(run_index=0):
    """Builds the deterministic PrepareCoffee task instance used by runtime."""

    return PREPARE_COFFEE_TASK.build_task_instance(run_index)


def build_prepare_coffee_prompt_for_run(variation_key, *, run_index=0, retry_feedback=None):
    """Builds the exact runtime prompt for one PrepareCoffee run."""

    return build_prepare_coffee_prompt(
        variation_key,
        task_instance=make_prepare_coffee_task_instance(run_index),
        retry_feedback=retry_feedback,
    )


class FakeBatchService:
    def __init__(self, created_jobs, *, job_sequences=None, get_side_effect=None):
        self._created_jobs = list(created_jobs)
        self._job_sequences = dict(job_sequences or {})
        self._get_side_effect = get_side_effect
        self.create_calls = []
        self.get_calls = []
        self.cancel_calls = []

    def create_job(self, *, model, input_uri, output_prefix, display_name):
        self.create_calls.append(
            {
                "model": model,
                "input_uri": input_uri,
                "output_prefix": output_prefix,
                "display_name": display_name,
            }
        )
        if not self._created_jobs:
            raise AssertionError("No fake batch jobs configured.")
        job = self._created_jobs.pop(0)
        self._job_sequences.setdefault(job.name, [job])
        return job

    def get_job(self, *, name):
        self.get_calls.append(name)
        if self._get_side_effect is not None:
            raise self._get_side_effect
        sequence = self._job_sequences.get(name)
        if not sequence:
            raise AssertionError(f"No fake batch job sequence configured for {name}.")
        if len(sequence) > 1:
            return sequence.pop(0)
        return sequence[0]

    def cancel_job(self, *, name):
        self.cancel_calls.append(name)


class FakeBatchStorage:
    def __init__(self, downloads_by_prefix=None):
        self.downloads_by_prefix = dict(downloads_by_prefix or {})
        self.upload_calls = []
        self.download_calls = []

    def upload_text(self, *, text, gcs_uri):
        self.upload_calls.append({"gcs_uri": gcs_uri, "text": text})

    def download_texts(self, *, gcs_prefix):
        self.download_calls.append(gcs_prefix)
        return list(self.downloads_by_prefix.get(gcs_prefix, []))


def make_fake_google_genai_modules(client_cls):
    fake_google_module = types.ModuleType("google")
    fake_google_genai_module = types.ModuleType("google.genai")
    fake_google_genai_types_module = types.ModuleType("google.genai.types")

    class FakeHttpOptions:
        def __init__(self, **kwargs):
            self.api_version = kwargs.get("api_version")

    fake_google_genai_module.Client = client_cls
    fake_google_genai_types_module.HttpOptions = FakeHttpOptions
    fake_google_genai_module.types = fake_google_genai_types_module
    fake_google_module.genai = fake_google_genai_module
    return (
        fake_google_module,
        fake_google_genai_module,
        fake_google_genai_types_module,
    )


class SubatomicToolCatalogTests(unittest.TestCase):
    def test_discover_subatomic_tools_exposes_shared_catalog(self):
        tool_names = {tool.name for tool in discover_subatomic_tools()}
        self.assertIn("give_space", tool_names)
        self.assertIn("get_image", tool_names)
        self.assertIn("place_next_to", tool_names)
        self.assertIn("pick_up_object", tool_names)
        self.assertIn("place_under", tool_names)
        self.assertIn("place_under_dispenser", tool_names)
        self.assertIn("press_button", tool_names)
        self.assertIn("wait", tool_names)
        self.assertTrue(all(tool_name == tool_name.lower() for tool_name in tool_names))

    def test_prepare_coffee_prompt_contains_allowed_tools_and_rules(self):
        prompt = build_prepare_coffee_prompt("unit-test")
        self.assertIn("give_space", prompt)
        self.assertIn("pick_up_object", prompt)
        self.assertIn("place_under_dispenser", prompt)
        self.assertIn("press_button", prompt)
        self.assertIn('"wait"', prompt)
        self.assertIn("communicate", prompt)
        self.assertIn("Simple execution rules:", prompt)
        self.assertIn("variation key: unit-test", prompt)
        self.assertIn(
            "Before interacting with a fixture, surface, receptacle, or dispenser",
            prompt,
        )
        self.assertIn(
            "Only use pick_up_object when the object is still at the listed source_id",
            prompt,
        )
        self.assertIn(
            "After picking up an object, that agent should only navigate or place that same object",
            prompt,
        )
        self.assertIn(
            "Only use a placement tool for the exact object the acting agent is currently holding.",
            prompt,
        )
        self.assertIn(
            "Open cabinet_1.door before using pick_up_object on mug_1 from cabinet_1.",
            prompt,
        )
        self.assertIn(
            "Only press coffee_machine_1.start_button after mug_1 is already at coffee_machine_dispenser.",
            prompt,
        )
        self.assertIn(
            "emit wait(seconds) instead of skipping that agent",
            prompt,
        )
        self.assertIn(
            "execute give_space(fixture_id)",
            prompt,
        )
        self.assertIn(
            "communicate first about that upcoming navigation",
            prompt,
        )
        self.assertIn(
            "before agent_A arrives",
            prompt,
        )
        self.assertIn("Initial agent positions:", prompt)

    def test_prepare_coffee_prompt_appends_retry_feedback(self):
        prompt = build_prepare_coffee_prompt(
            "unit-test",
            retry_feedback=(
                "Previous attempt failed validation.\n\n"
                "Failure summary:\n"
                "- error_type: NavigationSemanticValidationError\n"
                "- failing_step: 9\n"
                "- message: agent_1 must navigate first.\n\n"
                "Repair instructions:\n"
                "- Regenerate the full trajectory from step 0."
            ),
        )

        self.assertIn("Previous attempt failed validation.", prompt)
        self.assertIn("NavigationSemanticValidationError", prompt)
        self.assertIn("Regenerate the full trajectory from step 0.", prompt)
        self.assertIn(
            "use wait with a short positive duration",
            prompt,
        )
        self.assertNotIn("get_image", prompt)
        self.assertNotIn("Full subatomic tool catalog for context:", prompt)
        self.assertNotIn("Communication tool:", prompt)
        self.assertNotIn(
            "communicate: Send a short coordination message to the other agent.",
            prompt,
        )
        self.assertIn(
            "refer to agents using exact IDs like agent_0 and agent_1",
            prompt,
        )
        self.assertIn(
            "Number steps consecutively starting at 0 with no gaps.",
            prompt,
        )
        self.assertNotIn("entity_refs", prompt)
        self.assertNotIn("Output requirements:", prompt)
        self.assertNotIn("Return an object with key: steps.", prompt)
        self.assertNotIn("top-level agents field", prompt)
        self.assertNotIn(
            "Each agent entry must contain: agent, role, initial_plan.",
            prompt,
        )

    def test_prepare_coffee_allowed_tools_include_wait(self):
        self.assertIn("give_space", PREPARE_COFFEE_ALLOWED_TOOL_SPECS)
        self.assertIn("give_space", PREPARE_COFFEE_NON_COMMUNICATE_TOOL_NAMES)
        self.assertIn("wait", PREPARE_COFFEE_ALLOWED_TOOL_SPECS)
        self.assertIn("wait", PREPARE_COFFEE_NON_COMMUNICATE_TOOL_NAMES)


class PrepareCoffeeTaskInstanceTests(unittest.TestCase):
    def test_task_instance_samples_start_positions_from_existing_fixtures(self):
        task_instance = make_prepare_coffee_task_instance(0)
        allowed_fixture_ids = set(
            PREPARE_COFFEE_ALLOWED_TOOL_SPECS["navigate_to_fixture"]["allowed_fixture_ids"]
        )

        for agent_id in ("agent_0", "agent_1"):
            self.assertIn(
                task_instance.initial_state["agents"][agent_id]["location"],
                allowed_fixture_ids,
            )

    def test_task_instance_sampling_is_stable_for_the_same_run(self):
        first_task_instance = make_prepare_coffee_task_instance(0)
        second_task_instance = make_prepare_coffee_task_instance(0)

        self.assertEqual(first_task_instance.initial_state, second_task_instance.initial_state)

    def test_runtime_prompt_uses_sampled_initial_positions(self):
        task_instance = make_prepare_coffee_task_instance(0)
        prompt = build_prepare_coffee_prompt(
            "traj-000000-attempt-00",
            task_instance=task_instance,
        )

        self.assertIn("Initial agent positions:", prompt)
        for agent_id in ("agent_0", "agent_1"):
            self.assertIn(
                f"- {agent_id}: {task_instance.initial_state['agents'][agent_id]['location']}",
                prompt,
            )
        self.assertNotIn('"location": "staging_area"', prompt)

    def test_validator_uses_sampled_initial_positions(self):
        initial_state = deepcopy(PREPARE_COFFEE_INITIAL_STATE)
        initial_state["agents"]["agent_0"]["location"] = "cabinet_1"
        initial_state["agents"]["agent_1"]["location"] = "counter_1"
        validator = PrepareCoffeeValidator(TaskInstance(initial_state=initial_state))
        candidate = make_valid_candidate()
        candidate["steps"].pop(find_step_index(candidate, "navigate_to_fixture", occurrence=0))
        renumber_candidate_steps(candidate)

        validation = validator.validate(candidate)

        self.assertTrue(validation["is_valid"])

    def test_build_allowed_tool_specs_adds_basic_give_space(self):
        allowed_tool_specs = build_allowed_tool_specs(
            ("communicate", "navigate_to_fixture"),
            overrides={
                "navigate_to_fixture": {
                    "allowed_fixture_ids": ["table_1", "shelf_1"],
                }
            },
        )

        self.assertIn("give_space", allowed_tool_specs)
        self.assertEqual(
            allowed_tool_specs["give_space"]["tool_args"],
            ["fixture_id"],
        )
        self.assertEqual(
            allowed_tool_specs["give_space"]["allowed_fixture_ids"],
            ["table_1", "shelf_1"],
        )


class HotDogSetupTaskTests(unittest.TestCase):
    def test_hot_dog_setup_task_is_registered(self):
        self.assertEqual(HOT_DOG_SETUP_TASK.composite_task, "HotDogSetup")

    def test_hot_dog_setup_validator_accepts_valid_candidate(self):
        validator = HotDogSetupValidator(
            TaskInstance(initial_state=deepcopy(HOT_DOG_SETUP_INITIAL_STATE))
        )

        validation = validator.validate(make_valid_hot_dog_setup_candidate())

        self.assertTrue(validation["is_valid"])
        self.assertEqual(
            validation["final_state"]["objects"]["hotdog_bun_1"]["location"],
            "plate_1",
        )
        self.assertEqual(
            validation["final_state"]["objects"]["sausage_1"]["location"],
            "plate_1",
        )
        self.assertTrue(
            validation["final_state"]["machine_state"]["hot_dog_setup"]["condiment_placed_next_to_plate"]
        )

    def test_hot_dog_setup_validator_requires_condiment_next_to_plate(self):
        validator = HotDogSetupValidator(
            TaskInstance(initial_state=deepcopy(HOT_DOG_SETUP_INITIAL_STATE))
        )
        candidate = make_valid_hot_dog_setup_candidate()
        place_condiment_index = find_step_index(candidate, "place_next_to")
        candidate["steps"].pop(place_condiment_index)
        renumber_candidate_steps(candidate)

        with self.assertRaises(TaskSemanticValidationError) as raised:
            validator.validate(candidate)

        self.assertIsInstance(raised.exception, TaskSemanticValidationError)


class DotenvLoadingTests(unittest.TestCase):
    def test_load_dotenv_file_populates_missing_environment_values(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dotenv_path = Path(tmpdir) / ".env"
            dotenv_path.write_text(
                "\n".join(
                    [
                        "GOOGLE_CLOUD_PROJECT=dotenv-project",
                        'GOOGLE_CLOUD_LOCATION="europe-west4"',
                        "export GOOGLE_GENAI_USE_VERTEXAI=True",
                    ]
                ),
                encoding="utf-8",
            )
            with mock.patch.dict("os.environ", {}, clear=True):
                loaded = load_dotenv_file(dotenv_path)

        self.assertEqual(loaded["GOOGLE_CLOUD_PROJECT"], "dotenv-project")
        self.assertEqual(loaded["GOOGLE_CLOUD_LOCATION"], "europe-west4")
        self.assertEqual(loaded["GOOGLE_GENAI_USE_VERTEXAI"], "True")

    def test_load_dotenv_file_does_not_override_existing_environment_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dotenv_path = Path(tmpdir) / ".env"
            dotenv_path.write_text(
                "GOOGLE_CLOUD_PROJECT=dotenv-project\n",
                encoding="utf-8",
            )
            with mock.patch.dict(
                "os.environ",
                {"GOOGLE_CLOUD_PROJECT": "shell-project"},
                clear=True,
            ):
                load_dotenv_file(dotenv_path)
                runtime_config = parse_args([])

        self.assertEqual(runtime_config.project, "shell-project")

    def test_parse_args_accepts_explicit_cost_output(self):
        runtime_config = parse_args(["--cost-output", "/tmp/custom_costs.json"])
        self.assertEqual(
            runtime_config.cost_output_path, Path("/tmp/custom_costs.json")
        )

    def test_parse_args_rejects_removed_output_flag(self):
        with self.assertRaises(SystemExit):
            parse_args(["--output", "/tmp/trajectories.json"])

    def test_parse_args_disables_validation_by_default(self):
        runtime_config = parse_args([])
        self.assertTrue(runtime_config.disable_validation)

    def test_parse_args_accepts_enable_validation(self):
        runtime_config = parse_args(["--enable-validation"])
        self.assertFalse(runtime_config.disable_validation)

    def test_parse_args_accepts_thinking_level_flag_and_alias(self):
        dashed_runtime_config = parse_args(["--thinking-level", "minimal"])
        underscored_runtime_config = parse_args(["--thinking_level", "high"])

        self.assertEqual(dashed_runtime_config.thinking_level, "minimal")
        self.assertEqual(underscored_runtime_config.thinking_level, "high")

    def test_parse_args_accepts_num_runs_flag_and_alias(self):
        dashed_runtime_config = parse_args(["--num-runs", "7"])
        underscored_runtime_config = parse_args(["--num_runs", "9"])

        self.assertEqual(dashed_runtime_config.num_runs, 7)
        self.assertEqual(underscored_runtime_config.num_runs, 9)

    def test_parse_args_accepts_sampling_flags(self):
        runtime_config = parse_args(
            ["--sampling", "verbalized", "--verbalized-k", "3"]
        )

        self.assertEqual(runtime_config.sampling, "verbalized")
        self.assertEqual(runtime_config.verbalized_k, 3)

    def test_parse_args_accepts_tasks_flag_for_single_task(self):
        runtime_config = parse_args(["--tasks", "PrepareCoffee"])

        self.assertEqual(runtime_config.composite_task, "PrepareCoffee")
        self.assertEqual(runtime_config.composite_tasks, ("PrepareCoffee",))

    def test_parse_args_accepts_multiple_tasks(self):
        runtime_config = parse_args(["--tasks", "PrepareCoffee", "HotDogSetup"])

        self.assertEqual(runtime_config.composite_task, "PrepareCoffee")
        self.assertEqual(
            runtime_config.composite_tasks,
            ("PrepareCoffee", "HotDogSetup"),
        )
        self.assertIsNone(runtime_config.summary_path)

    def test_runtime_config_for_task_preserves_single_requested_task(self):
        runtime_config = parse_args(["--tasks", "PrepareCoffee", "HotDogSetup"])

        task_runtime_config = runtime_config.for_task("PrepareCoffee")

        self.assertEqual(task_runtime_config.composite_task, "PrepareCoffee")
        self.assertEqual(
            task_runtime_config.composite_tasks,
            ("PrepareCoffee",),
        )

    def test_parse_args_rejects_removed_task_flags(self):
        with self.assertRaises(SystemExit):
            parse_args(["--task", "PrepareCoffee"])
        with self.assertRaises(SystemExit):
            parse_args(["--composite-task", "PrepareCoffee"])
        with self.assertRaises(SystemExit):
            parse_args(["--composite_task", "PrepareCoffee"])

    def test_parse_args_accepts_batch_processing_and_env_batch_prefix(self):
        with mock.patch.dict(
            "os.environ",
            {"GOOGLE_CLOUD_BATCH_GCS_PREFIX": "gs://env-bucket/batch-prefix"},
            clear=True,
        ):
            runtime_config = parse_args(["--batch-processing"])

        self.assertTrue(runtime_config.batch_processing)
        self.assertEqual(
            runtime_config.batch_gcs_prefix,
            "gs://env-bucket/batch-prefix",
        )

    def test_parse_args_accepts_batch_processing_underscore_aliases(self):
        runtime_config = parse_args(
            [
                "--batch_processing",
                "--batch_gcs_prefix",
                "gs://cli-bucket/alias-prefix",
            ]
        )

        self.assertTrue(runtime_config.batch_processing)
        self.assertEqual(
            runtime_config.batch_gcs_prefix,
            "gs://cli-bucket/alias-prefix",
        )

    def test_resolve_dataset_output_path_adds_timestamped_subdirectory_for_default_output(
        self,
    ):
        resolved = resolve_dataset_output_path(
            "PrepareCoffee",
            generated_at=datetime(2026, 3, 10, 12, 34, 56, tzinfo=timezone.utc),
        )

        self.assertEqual(
            resolved,
            DEFAULT_OUTPUT_DIR
            / "prepare_coffee"
            / "20260310T123456Z"
            / "summary.json",
        )

    def test_resolve_dataset_output_path_uses_patched_default_output_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir)
            resolved = None

            with mock.patch(
                "data_generation.task_level.generation.outputs.DEFAULT_OUTPUT_DIR",
                output_root,
            ):
                resolved = resolve_dataset_output_path(
                    "PrepareCoffee",
                    generated_at=datetime(2026, 3, 10, 12, 34, 56, tzinfo=timezone.utc),
                )

        self.assertEqual(
            resolved,
            output_root
            / "prepare_coffee"
            / "20260310T123456Z"
            / "summary.json",
        )

    def test_resolve_request_output_path_uses_patched_default_output_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir)
            resolved = None

            with mock.patch(
                "data_generation.task_level.generation.outputs.DEFAULT_OUTPUT_DIR",
                output_root,
            ):
                resolved = resolve_request_output_path(
                    generated_at=datetime(2026, 3, 10, 12, 34, 56, tzinfo=timezone.utc),
                )

        self.assertEqual(
            resolved,
            output_root
            / REQUEST_DIRECTORY_NAME
            / "20260310T123456Z"
            / "summary.json",
        )

    def test_resolve_trajectory_output_dir_uses_sibling_trajectories_directory(self):
        output_path = Path("/tmp/summary.json")

        self.assertEqual(
            resolve_trajectory_output_dir(output_path),
            Path("/tmp/trajectories"),
        )

    def test_resolve_prompt_output_dir_uses_sibling_prompts_directory(self):
        output_path = Path("/tmp/summary.json")

        self.assertEqual(
            resolve_prompt_output_dir(output_path),
            Path("/tmp/prompts"),
        )

    def test_resolve_raw_output_dir_uses_sibling_outputs_directory(self):
        output_path = Path("/tmp/summary.json")

        self.assertEqual(
            resolve_raw_output_dir(output_path),
            Path("/tmp/outputs"),
        )

    def test_resolve_error_output_path_defaults_to_error_summary_filename(self):
        self.assertEqual(
            resolve_error_output_path(Path("/tmp/summary.json")),
            Path("/tmp/summary_errors.json"),
        )

    def test_validate_google_auth_raises_clear_error_when_credentials_missing(self):
        fake_google_module = types.ModuleType("google")
        fake_google_auth_module = types.ModuleType("google.auth")
        fake_google_auth_exceptions_module = types.ModuleType("google.auth.exceptions")

        class FakeDefaultCredentialsError(Exception):
            pass

        def fake_default(*args, **kwargs):
            raise FakeDefaultCredentialsError("missing")

        fake_google_auth_module.default = fake_default
        fake_google_auth_exceptions_module.DefaultCredentialsError = (
            FakeDefaultCredentialsError
        )
        fake_google_module.auth = fake_google_auth_module

        with mock.patch.dict(
            "sys.modules",
            {
                "google": fake_google_module,
                "google.auth": fake_google_auth_module,
                "google.auth.exceptions": fake_google_auth_exceptions_module,
            },
        ):
            with self.assertRaises(TrajectoryGenerationError):
                validate_google_auth("demo-project")

    def test_google_genai_client_uses_env_driven_init_for_adc_path(self):
        class FakeGenAIClient:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        (
            fake_google_module,
            fake_google_genai_module,
            fake_google_genai_types_module,
        ) = make_fake_google_genai_modules(FakeGenAIClient)

        with mock.patch.dict("os.environ", {}, clear=True):
            with mock.patch.dict(
                "sys.modules",
                {
                    "google": fake_google_module,
                    "google.genai": fake_google_genai_module,
                    "google.genai.types": fake_google_genai_types_module,
                },
            ):
                with mock.patch(
                    "data_generation.task_level.runtime.client.validate_google_auth"
                ) as validate_auth:
                    client = GoogleGenAIClient(project="demo-project", location="global")

        validate_auth.assert_called_once_with("demo-project")
        self.assertEqual(set(client._client.kwargs), {"http_options"})
        self.assertEqual(client._client.kwargs["http_options"].api_version, "v1")

    def test_google_genai_client_raises_clear_error_for_missing_vertex_permissions(self):
        class FakeModels:
            def generate_content(self, **kwargs):
                raise FakeGoogleGenAIClientError(
                    "403 PERMISSION_DENIED. Permission "
                    "'aiplatform.endpoints.predict' denied."
                )

        class FakeGenAIClient:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.models = FakeModels()

        (
            fake_google_module,
            fake_google_genai_module,
            fake_google_genai_types_module,
        ) = make_fake_google_genai_modules(FakeGenAIClient)

        with mock.patch.dict(
            "os.environ",
            {"GOOGLE_CLOUD_PROJECT": "demo-project"},
            clear=True,
        ):
            with mock.patch.dict(
                "sys.modules",
                {
                    "google": fake_google_module,
                    "google.genai": fake_google_genai_module,
                    "google.genai.types": fake_google_genai_types_module,
                },
            ):
                with mock.patch(
                    "data_generation.task_level.runtime.client.validate_google_auth"
                ):
                    client = GoogleGenAIClient(project="demo-project", location="global")

        with self.assertRaises(TrajectoryGenerationError) as context:
            client.generate(
                model="gemini-3-flash-preview",
                prompt="Say hi",
                response_schema={},
                temperature=0.1,
            )

        self.assertIn("Vertex AI `GenerateContent`", str(context.exception))
        self.assertIn("aiplatform.endpoints.predict", str(context.exception))

    def test_google_genai_client_includes_optional_thinking_level(self):
        class FakeModels:
            def __init__(self):
                self.calls = []

            def generate_content(self, **kwargs):
                self.calls.append(kwargs)
                return types.SimpleNamespace(text='{"ok": true}', usage_metadata=None)

        class FakeGenAIClient:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.models = FakeModels()

        (
            fake_google_module,
            fake_google_genai_module,
            fake_google_genai_types_module,
        ) = make_fake_google_genai_modules(FakeGenAIClient)

        with mock.patch.dict("os.environ", {}, clear=True):
            with mock.patch.dict(
                "sys.modules",
                {
                    "google": fake_google_module,
                    "google.genai": fake_google_genai_module,
                    "google.genai.types": fake_google_genai_types_module,
                },
            ):
                with mock.patch(
                    "data_generation.task_level.runtime.client.validate_google_auth"
                ):
                    client = GoogleGenAIClient(project="demo-project", location="global")

        client.generate(
            model="gemini-3.1-flash-lite-preview",
            prompt="Say hi",
            response_schema={"type": "OBJECT"},
            temperature=0.1,
            thinking_level="minimal",
        )

        call = client._client.models.calls[0]
        self.assertEqual(call["model"], "gemini-3.1-flash-lite-preview")
        self.assertEqual(call["config"]["max_output_tokens"], 32768)
        self.assertEqual(
            call["config"]["thinking_config"],
            {"thinking_level": "minimal"},
        )

    def test_google_genai_client_raises_clear_error_for_invalid_argument(self):
        class InvalidArgumentError(Exception):
            def __init__(self):
                super().__init__(
                    "400 INVALID_ARGUMENT. {'error': {'code': 400, 'message': "
                    "'Request contains an invalid argument.', 'status': "
                    "'INVALID_ARGUMENT'}}"
                )
                self.response = types.SimpleNamespace(status_code=400)

        class FakeModels:
            def generate_content(self, **kwargs):
                raise InvalidArgumentError()

        class FakeGenAIClient:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.models = FakeModels()

        (
            fake_google_module,
            fake_google_genai_module,
            fake_google_genai_types_module,
        ) = make_fake_google_genai_modules(FakeGenAIClient)

        with mock.patch.dict("os.environ", {}, clear=True):
            with mock.patch.dict(
                "sys.modules",
                {
                    "google": fake_google_module,
                    "google.genai": fake_google_genai_module,
                    "google.genai.types": fake_google_genai_types_module,
                },
            ):
                with mock.patch(
                    "data_generation.task_level.runtime.client.validate_google_auth"
                ):
                    client = GoogleGenAIClient(project="demo-project", location="global")

        with self.assertRaises(TrajectoryGenerationError) as context:
            client.generate(
                model="gemini-3.1-flash-lite-preview",
                prompt="Say hi",
                response_schema={
                    "type": "OBJECT",
                    "properties": {
                        "responses": {
                            "type": "ARRAY",
                            "minItems": 10,
                            "maxItems": 10,
                        }
                    },
                },
                temperature=0.1,
                thinking_level="low",
            )

        self.assertIn("400 INVALID_ARGUMENT", str(context.exception))
        self.assertIn("--verbalized-k", str(context.exception))
        self.assertIn("4 or lower", str(context.exception))


class FiniteStateTaskValidatorTests(unittest.TestCase):
    def test_build_task_response_schema_uses_allowed_tool_specs(self):
        allowed_tool_specs = build_allowed_tool_specs(
            (
                "communicate",
                "get_image",
                "pick_up_object",
                "place_in_receptacle",
                "place_next_to",
                "place_on_object",
                "place_under",
                "set_rotary_control",
                "wait",
            )
        )

        response_schema = build_task_response_schema(
            agent_ids=("agent_0", "agent_1"),
            allowed_tool_specs=allowed_tool_specs,
            min_steps=4,
        )

        step_properties = response_schema["properties"]["steps"]["items"]["properties"]
        self.assertEqual(
            list(step_properties["tool"]["enum"]),
            list(allowed_tool_specs),
        )
        self.assertEqual(
            list(step_properties["args"]["properties"]),
            [
                "to",
                "message",
                "camera_view",
                "object_id",
                "source_id",
                "receptacle_id",
                "reference_object_id",
                "support_object_id",
                "reference_fixture_id",
                "target_id",
                "control_id",
                "goal",
                "seconds",
                "fixture_id",
            ],
        )
        self.assertEqual(
            step_properties["args"]["properties"]["to"]["enum"],
            ["agent_0", "agent_1"],
        )
        self.assertEqual(
            step_properties["args"]["properties"]["seconds"]["type"],
            "INTEGER",
        )
        self.assertNotIn("agents", response_schema["properties"])
        self.assertEqual(response_schema["required"], ["steps"])
        self.assertNotIn("image_path", step_properties)
        self.assertNotIn("entity_refs", step_properties)
        self.assertEqual(response_schema["properties"]["steps"]["minItems"], 4)

    def test_validator_inserts_canonical_agents_when_model_omits_them(self):
        validator = PrepareCoffeeValidator()
        candidate = make_valid_candidate(include_agents=False)

        validation = validator.validate(candidate)

        self.assertTrue(validation["is_valid"])
        self.assertEqual(
            validation["normalized_candidate"]["agents"],
            [{"agent": "agent_0"}, {"agent": "agent_1"}],
        )

    def test_validator_allows_navigation_while_holding(self):
        actions = (
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "table_1"},
            ),
            make_toy_action_spec(
                "pick_up_object",
                {"object_id": "apple_1", "source_id": "table_1"},
            ),
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "shelf_1"},
            ),
            make_toy_action_spec(
                "place_on_surface",
                {"object_id": "apple_1", "support_id": "shelf_1"},
            ),
        )

        validator = ToyFiniteStateValidator()
        validation = validator.validate(make_toy_candidate(actions))

        self.assertTrue(validation["is_valid"])
        self.assertEqual(
            validation["final_state"]["objects"]["apple_1"]["location"],
            "shelf_1",
        )

    def test_validator_allows_get_image_while_holding(self):
        actions = (
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "table_1"},
            ),
            make_toy_action_spec(
                "pick_up_object",
                {"object_id": "apple_1", "source_id": "table_1"},
            ),
            make_toy_action_spec(
                "get_image",
                {"camera_view": "wrist"},
            ),
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "shelf_1"},
            ),
            make_toy_action_spec(
                "place_on_surface",
                {"object_id": "apple_1", "support_id": "shelf_1"},
            ),
        )

        validator = ToyFiniteStateValidator()
        validation = validator.validate(make_toy_candidate(actions))

        self.assertTrue(validation["is_valid"])
        self.assertEqual(
            validation["final_state"]["objects"]["apple_1"]["location"],
            "shelf_1",
        )

    def test_validator_allows_wait_while_holding(self):
        actions = (
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "table_1"},
            ),
            make_toy_action_spec(
                "pick_up_object",
                {"object_id": "apple_1", "source_id": "table_1"},
            ),
            make_toy_action_spec(
                "wait",
                {"seconds": 3},
            ),
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "shelf_1"},
            ),
            make_toy_action_spec(
                "place_on_surface",
                {"object_id": "apple_1", "support_id": "shelf_1"},
            ),
        )

        validator = ToyFiniteStateValidator()
        validation = validator.validate(make_toy_candidate(actions))

        self.assertTrue(validation["is_valid"])
        self.assertEqual(
            validation["final_state"]["objects"]["apple_1"]["location"],
            "shelf_1",
        )

    def test_validator_allows_give_space_while_holding(self):
        actions = (
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "table_1"},
            ),
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "table_1"},
            ),
            make_toy_action_spec(
                "pick_up_object",
                {"object_id": "apple_1", "source_id": "table_1"},
            ),
            make_toy_action_spec(
                "give_space",
                {"fixture_id": "table_1"},
            ),
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "shelf_1"},
            ),
            make_toy_action_spec(
                "place_on_surface",
                {"object_id": "apple_1", "support_id": "shelf_1"},
            ),
        )

        validator = ToyFiniteStateValidator()
        validation = validator.validate(
            make_toy_candidate(
                actions,
                action_agents=(
                    "agent_0",
                    "agent_1",
                    "agent_0",
                    "agent_0",
                    "agent_0",
                    "agent_0",
                ),
            )
        )

        self.assertTrue(validation["is_valid"])

    def test_validator_allows_give_space_before_other_agent_arrives(self):
        actions = (
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "table_1"},
            ),
            make_toy_action_spec(
                "pick_up_object",
                {"object_id": "apple_1", "source_id": "table_1"},
            ),
            make_toy_action_spec(
                "give_space",
                {"fixture_id": "table_1"},
            ),
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "table_1"},
            ),
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "shelf_1"},
            ),
            make_toy_action_spec(
                "place_on_surface",
                {"object_id": "apple_1", "support_id": "shelf_1"},
            ),
        )

        validator = ToyFiniteStateValidator()
        validation = validator.validate(
            make_toy_candidate(
                actions,
                action_agents=(
                    "agent_0",
                    "agent_0",
                    "agent_0",
                    "agent_1",
                    "agent_0",
                    "agent_0",
                ),
            )
        )

        self.assertTrue(validation["is_valid"])

    def test_validator_allows_place_next_to_using_reference_object_location(self):
        actions = (
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "table_1"},
            ),
            make_toy_action_spec(
                "pick_up_object",
                {"object_id": "apple_1", "source_id": "table_1"},
            ),
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "shelf_1"},
            ),
            make_toy_action_spec(
                "place_next_to",
                {"object_id": "apple_1", "reference_object_id": "mug_1"},
            ),
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "table_1"},
            ),
            make_toy_action_spec(
                "pick_up_object",
                {"object_id": "cup_1", "source_id": "table_1"},
            ),
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "coffee_machine_1"},
            ),
            make_toy_action_spec(
                "place_under",
                {"object_id": "cup_1", "reference_fixture_id": "coffee_machine_1"},
            ),
        )

        validator = PlacementReferenceValidator()
        validation = validator.validate(make_toy_candidate(actions))

        self.assertTrue(validation["is_valid"])
        self.assertEqual(
            validation["final_state"]["objects"]["apple_1"]["location"],
            "shelf_1",
        )

    def test_validator_allows_place_under_using_fixture_dispenser_location(self):
        actions = (
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "table_1"},
            ),
            make_toy_action_spec(
                "pick_up_object",
                {"object_id": "apple_1", "source_id": "table_1"},
            ),
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "shelf_1"},
            ),
            make_toy_action_spec(
                "place_next_to",
                {"object_id": "apple_1", "reference_object_id": "mug_1"},
            ),
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "table_1"},
            ),
            make_toy_action_spec(
                "pick_up_object",
                {"object_id": "cup_1", "source_id": "table_1"},
            ),
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "coffee_machine_1"},
            ),
            make_toy_action_spec(
                "place_under",
                {"object_id": "cup_1", "reference_fixture_id": "coffee_machine_1"},
            ),
        )

        validator = PlacementReferenceValidator()
        validation = validator.validate(make_toy_candidate(actions))

        self.assertTrue(validation["is_valid"])
        self.assertEqual(
            validation["final_state"]["objects"]["cup_1"]["location"],
            "coffee_machine_dispenser",
        )

    def test_validator_rejects_non_positive_wait_duration(self):
        actions = (
            make_toy_action_spec(
                "wait",
                {"seconds": 0},
            ),
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "table_1"},
            ),
            make_toy_action_spec(
                "pick_up_object",
                {"object_id": "apple_1", "source_id": "table_1"},
            ),
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "shelf_1"},
            ),
            make_toy_action_spec(
                "place_on_surface",
                {"object_id": "apple_1", "support_id": "shelf_1"},
            ),
        )

        validator = ToyFiniteStateValidator()

        with self.assertRaises(TaskSemanticValidationError) as raised:
            validator.validate(make_toy_candidate(actions))

        self.assertIsInstance(raised.exception, WaitDurationSemanticValidationError)
        self.assertIn(
            "wait requires seconds to be a positive integer",
            str(raised.exception),
        )

    def test_validator_accepts_alternative_valid_action_order(self):
        actions = (
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "cabinet_1"},
            ),
            make_toy_action_spec(
                "open_hinged_part",
                {"target_id": "cabinet_1", "part_id": "door"},
            ),
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "table_1"},
            ),
            make_toy_action_spec(
                "pick_up_object",
                {"object_id": "apple_1", "source_id": "table_1"},
            ),
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "shelf_1"},
            ),
            make_toy_action_spec(
                "place_on_surface",
                {"object_id": "apple_1", "support_id": "shelf_1"},
            ),
        )

        validator = ToyFiniteStateValidator()
        validation = validator.validate(make_toy_candidate(actions))

        self.assertTrue(validation["is_valid"])

    def test_validator_rejects_missing_get_image_before_task_action(self):
        actions = (
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "table_1"},
            ),
            make_toy_action_spec(
                "pick_up_object",
                {"object_id": "apple_1", "source_id": "table_1"},
            ),
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "shelf_1"},
            ),
            make_toy_action_spec(
                "place_on_surface",
                {"object_id": "apple_1", "support_id": "shelf_1"},
            ),
        )
        candidate = make_toy_candidate(actions)
        candidate["steps"].pop(find_step_index(candidate, "get_image", occurrence=0))
        renumber_candidate_steps(candidate)

        validator = ToyFiniteStateValidator()

        with self.assertRaises(TaskSemanticValidationError) as raised:
            validator.validate(candidate)

        self.assertIsInstance(raised.exception, ObservationSequenceSemanticValidationError)
        self.assertIn("must be immediately preceded by get_image", str(raised.exception))

    def test_validator_rejects_missing_get_image_after_task_action(self):
        actions = (
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "table_1"},
            ),
            make_toy_action_spec(
                "pick_up_object",
                {"object_id": "apple_1", "source_id": "table_1"},
            ),
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "shelf_1"},
            ),
            make_toy_action_spec(
                "place_on_surface",
                {"object_id": "apple_1", "support_id": "shelf_1"},
            ),
        )
        candidate = make_toy_candidate(actions)
        candidate["steps"].pop(find_step_index(candidate, "get_image", occurrence=4))
        renumber_candidate_steps(candidate)

        validator = ToyFiniteStateValidator()

        with self.assertRaises(TaskSemanticValidationError) as raised:
            validator.validate(candidate)

        self.assertIsInstance(raised.exception, ObservationSequenceSemanticValidationError)
        self.assertIn("must be immediately followed by get_image", str(raised.exception))

    def test_validator_rejects_second_pickup_while_holding(self):
        actions = (
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "table_1"},
            ),
            make_toy_action_spec(
                "pick_up_object",
                {"object_id": "apple_1", "source_id": "table_1"},
            ),
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "table_1"},
            ),
            make_toy_action_spec(
                "pick_up_object",
                {"object_id": "mug_1", "source_id": "table_1"},
            ),
        )

        validator = ToyFiniteStateValidator()

        with self.assertRaises(TaskSemanticValidationError) as raised:
            validator.validate(make_toy_candidate(actions))

        self.assertIsInstance(raised.exception, HeldObjectSemanticValidationError)
        self.assertIn("cannot pick up a second object", str(raised.exception))

    def test_validator_rejects_empty_hand_only_tool_while_holding(self):
        actions = (
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "table_1"},
            ),
            make_toy_action_spec(
                "pick_up_object",
                {"object_id": "apple_1", "source_id": "table_1"},
            ),
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "cabinet_1"},
            ),
            make_toy_action_spec(
                "open_hinged_part",
                {"target_id": "cabinet_1", "part_id": "door"},
            ),
        )

        validator = ToyFiniteStateValidator()

        with self.assertRaises(TaskSemanticValidationError) as raised:
            validator.validate(make_toy_candidate(actions))

        self.assertIn(
            "must place apple_1 before using open_hinged_part",
            str(raised.exception),
        )

    def test_validator_rejects_placement_without_holding_object(self):
        actions = (
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "shelf_1"},
            ),
            make_toy_action_spec(
                "place_on_surface",
                {"object_id": "apple_1", "support_id": "shelf_1"},
            ),
        )

        validator = ToyFiniteStateValidator()

        with self.assertRaises(TaskSemanticValidationError) as raised:
            validator.validate(make_toy_candidate(actions))

        self.assertIn(
            "must be holding apple_1 before placing it",
            str(raised.exception),
        )

    def test_validator_rejects_interaction_without_navigation(self):
        actions = (
            make_toy_action_spec(
                "open_hinged_part",
                {"target_id": "cabinet_1", "part_id": "door"},
            ),
        )

        validator = ToyFiniteStateValidator()

        with self.assertRaises(TaskSemanticValidationError) as raised:
            validator.validate(make_toy_candidate(actions))

        self.assertIsInstance(raised.exception, NavigationSemanticValidationError)
        self.assertIn("must use navigate_to_fixture to reach cabinet_1", str(raised.exception))

    def test_validator_rejects_missing_initial_communication(self):
        actions = (
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "table_1"},
            ),
        )
        candidate = make_toy_candidate(actions)
        candidate["steps"] = candidate["steps"][1:]
        for index, step in enumerate(candidate["steps"]):
            step["step"] = index

        validator = ToyFiniteStateValidator()

        with self.assertRaises(TaskSemanticValidationError) as raised:
            validator.validate(candidate)

        self.assertIsInstance(raised.exception, MissingInitialCommunicationSemanticValidationError)
        self.assertIn(
            "Both agents must coordinate via communication before the first task action.",
            str(raised.exception),
        )

    def test_validator_rejects_legal_trace_that_never_reaches_goal(self):
        actions = (
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "table_1"},
            ),
            make_toy_action_spec(
                "pick_up_object",
                {"object_id": "apple_1", "source_id": "table_1"},
            ),
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "shelf_1"},
            ),
        )

        validator = ToyFiniteStateValidator()

        with self.assertRaises(TaskSemanticValidationError) as raised:
            validator.validate(make_toy_candidate(actions))

        self.assertIn(
            "Trajectory never satisfied the ToyFSMTask goal state.",
            str(raised.exception),
        )

    def test_validator_rejects_extra_steps_after_goal_state(self):
        actions = (
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "table_1"},
            ),
            make_toy_action_spec(
                "pick_up_object",
                {"object_id": "apple_1", "source_id": "table_1"},
            ),
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "shelf_1"},
            ),
            make_toy_action_spec(
                "place_on_surface",
                {"object_id": "apple_1", "support_id": "shelf_1"},
            ),
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "cabinet_1"},
            ),
        )

        validator = ToyFiniteStateValidator()

        with self.assertRaises(TaskSemanticValidationError) as raised:
            validator.validate(make_toy_candidate(actions))

        self.assertIn(
            "No steps are allowed after the ToyFSMTask goal state is satisfied.",
            str(raised.exception),
        )


class PrepareCoffeeValidatorTests(unittest.TestCase):
    def setUp(self):
        self.validator = PrepareCoffeeValidator()

    def test_validator_accepts_valid_trace(self):
        validation = self.validator.validate(make_valid_candidate())
        self.assertTrue(validation["is_valid"])
        self.assertTrue(validation["final_state"]["coffee_machine_started"])
        self.assertEqual(
            validation["final_state"]["objects"]["mug_1"]["location"],
            "coffee_machine_dispenser",
        )

    def test_validator_rejects_missing_initial_communication(self):
        candidate = make_valid_candidate()
        candidate["steps"].pop(1)
        renumber_candidate_steps(candidate)

        with self.assertRaises(TrajectoryValidationError):
            self.validator.validate(candidate)

    def test_validator_accepts_alternative_valid_order(self):
        validation = self.validator.validate(make_alternative_valid_candidate())

        self.assertTrue(validation["is_valid"])
        self.assertTrue(validation["final_state"]["coffee_machine_started"])

    def test_validator_rejects_broken_entity_continuity(self):
        candidate = make_valid_candidate()
        candidate["steps"][find_step_index(candidate, "pick_up_object", occurrence=0)][
            "args"
        ]["object_id"] = "mug_2"

        with self.assertRaises(TrajectoryValidationError):
            self.validator.validate(candidate)

    def test_validator_rejects_legal_trace_that_never_reaches_goal(self):
        candidate = make_valid_candidate()
        candidate["steps"] = candidate["steps"][:11]

        with self.assertRaises(TaskSemanticValidationError) as raised:
            self.validator.validate(candidate)

        self.assertIn(
            "Trajectory never satisfied the PrepareCoffee goal state.",
            str(raised.exception),
        )

    def test_validator_rejects_missing_navigation_before_interaction(self):
        candidate = make_valid_candidate()
        candidate["steps"].pop(find_step_index(candidate, "navigate_to_fixture", occurrence=2))
        renumber_candidate_steps(candidate)

        with self.assertRaises(TaskSemanticValidationError) as raised:
            self.validator.validate(candidate)

        self.assertIsInstance(raised.exception, NavigationSemanticValidationError)

    def test_validator_rejects_missing_cabinet_open_before_pickup(self):
        candidate = make_valid_candidate()
        candidate["steps"].pop(find_step_index(candidate, "open_hinged_part", occurrence=0))
        renumber_candidate_steps(candidate)

        with self.assertRaises(TaskSemanticValidationError) as raised:
            self.validator.validate(candidate)

        self.assertIsInstance(raised.exception, TaskPreconditionSemanticValidationError)

    def test_validator_rejects_invalid_hold_place_ordering(self):
        candidate = make_valid_candidate()
        candidate["steps"][find_step_index(candidate, "place_on_surface", occurrence=0)][
            "agent"
        ] = "agent_1"

        with self.assertRaises(TaskSemanticValidationError):
            self.validator.validate(candidate)

    def test_validator_rejects_unsupported_tool_name(self):
        candidate = make_valid_candidate()
        candidate["steps"][find_step_index(candidate, "pick_up_object", occurrence=0)][
            "tool"
        ] = "unsupported_tool"
        candidate["steps"][find_step_index(candidate, "unsupported_tool", occurrence=0)][
            "args"
        ] = {
            "object_id": "mug_1",
        }

        with self.assertRaises(TaskSemanticValidationError):
            self.validator.validate(candidate)

    def test_validator_rejects_invalid_task_specific_control_id(self):
        candidate = make_valid_candidate()
        candidate["steps"][find_step_index(candidate, "press_button", occurrence=0)][
            "args"
        ]["control_id"] = "wrong_button"

        with self.assertRaises(TaskSemanticValidationError) as raised:
            self.validator.validate(candidate)

        self.assertIsInstance(raised.exception, ToolArgumentSemanticValidationError)
        self.assertIn("press_button requires control_id", str(raised.exception))

    def test_validator_rejects_missing_reasoning(self):
        candidate = make_valid_candidate()
        candidate["steps"][find_step_index(candidate, "navigate_to_fixture", occurrence=0)][
            "reasoning"
        ] = ""

        with self.assertRaises(TrajectoryStructureValidationError) as raised:
            self.validator.validate(candidate)

        self.assertEqual(raised.exception.step, 2)

    def test_validator_rejects_missing_navigation_with_step_number(self):
        candidate = make_valid_candidate()
        candidate["steps"].pop(find_step_index(candidate, "navigate_to_fixture", occurrence=2))
        renumber_candidate_steps(candidate)

        with self.assertRaises(TaskSemanticValidationError) as raised:
            self.validator.validate(candidate)

        self.assertIsInstance(raised.exception, NavigationSemanticValidationError)
        self.assertEqual(raised.exception.step, 7)
        self.assertIn("Step 7 (pick_up_object):", str(raised.exception))
        self.assertIn("must use navigate_to_fixture to reach counter_1", str(raised.exception))
        self.assertIn("before using pick_up_object", str(raised.exception))

    def test_extract_json_candidate_uses_response_format_error_for_invalid_payload(self):
        with self.assertRaises(ResponseFormatValidationError):
            extract_json_candidate("not json")

    def test_duplicate_signature_uses_specific_validation_error(self):
        validation = self.validator.validate(make_valid_candidate())

        with self.assertRaises(DuplicateTrajectoryValidationError):
            _maybe_reserve_signature(
                validation,
                disable_validation=False,
                seen_signatures={validation["signature"]},
                seen_signatures_lock=threading.Lock(),
            )

    def test_trajectory_signature_includes_initial_state(self):
        first_validation = self.validator.validate(make_valid_candidate())
        alternate_initial_state = deepcopy(PREPARE_COFFEE_INITIAL_STATE)
        alternate_initial_state["agents"]["agent_0"]["location"] = "counter_1"
        alternate_initial_state["agents"]["agent_1"]["location"] = "cabinet_1"
        alternate_validator = PrepareCoffeeValidator(
            TaskInstance(initial_state=alternate_initial_state)
        )

        second_validation = alternate_validator.validate(make_valid_candidate())

        self.assertNotEqual(first_validation["signature"], second_validation["signature"])

    def test_base_sampling_strategy_preserves_task_prompt(self):
        strategy = BaseSamplingStrategy()

        prompt = strategy.build_prompt(
            task_definition=PREPARE_COFFEE_TASK,
            runtime_config=RuntimeConfig(
                composite_task="PrepareCoffee",
                num_runs=1,
                model="gemini-3-flash-preview",
                sdk="google-genai",
                project="demo-project",
                location="global",
                temperature=0.5,
                max_workers=1,
                max_retries=1,
            ),
            task_instance=make_prepare_coffee_task_instance(0),
            variation_key="traj-000000-attempt-00",
        )

        self.assertNotIn("Output requirements:", prompt)
        self.assertNotIn("Base sampling instructions:", prompt)
        self.assertNotIn("responses array", prompt)

    def test_verbalized_sampling_strategy_prompt_owns_multi_trajectory_wrapper(self):
        strategy = VerbalizedSamplingStrategy()

        prompt = strategy.build_prompt(
            task_definition=PREPARE_COFFEE_TASK,
            runtime_config=RuntimeConfig(
                composite_task="PrepareCoffee",
                num_runs=1,
                model="gemini-3-flash-preview",
                sdk="google-genai",
                project="demo-project",
                location="global",
                temperature=0.5,
                max_workers=1,
                max_retries=1,
                sampling="verbalized",
                verbalized_k=2,
            ),
            task_instance=make_prepare_coffee_task_instance(0),
            variation_key="traj-000000-attempt-00",
        )

        self.assertNotIn("Output requirements:", prompt)
        self.assertIn("Verbalized sampling instructions:", prompt)
        self.assertIn("Return one JSON object with key responses.", prompt)
        self.assertIn(
            "do not concentrate communication only at the beginning",
            prompt,
        )
        self.assertIn("Do not return a standalone top-level steps object.", prompt)
        self.assertNotIn("Ignore the single-trajectory output shape above", prompt)

    def test_verbalized_sampling_strategy_extracts_candidates(self):
        strategy = VerbalizedSamplingStrategy()

        sampled_candidates = strategy.extract_candidates(
            raw_response=make_verbalized_response(
                make_valid_candidate(include_agents=False),
                make_alternative_valid_candidate(),
                probabilities=[0.7, 0.3],
                as_json_string=True,
            ),
            task_definition=PREPARE_COFFEE_TASK,
            runtime_config=RuntimeConfig(
                composite_task="PrepareCoffee",
                num_runs=1,
                model="gemini-3-flash-preview",
                sdk="google-genai",
                project="demo-project",
                location="global",
                temperature=0.5,
                max_workers=1,
                max_retries=1,
                sampling="verbalized",
                verbalized_k=2,
            ),
        )

        self.assertEqual(len(sampled_candidates), 2)
        self.assertEqual(sampled_candidates[0].probability, 0.7)
        self.assertEqual(
            sampled_candidates[0].candidate["steps"][0]["tool"],
            "communicate",
        )

    def test_verbalized_sampling_strategy_rejects_duplicate_trajectories(self):
        strategy = VerbalizedSamplingStrategy()
        runtime_config = RuntimeConfig(
            composite_task="PrepareCoffee",
            num_runs=1,
            model="gemini-3-flash-preview",
            sdk="google-genai",
            project="demo-project",
            location="global",
            temperature=0.5,
            max_workers=1,
            max_retries=1,
            sampling="verbalized",
            verbalized_k=2,
        )

        with self.assertRaises(VerbalizedSamplingValidationError):
            strategy.extract_candidates(
                raw_response=make_verbalized_response(
                    make_valid_candidate(),
                    make_valid_candidate(),
                    probabilities=[0.5, 0.5],
                ),
                task_definition=PREPARE_COFFEE_TASK,
                runtime_config=runtime_config,
            )


class GenerationTests(unittest.TestCase):
    def test_resolve_pricing_tier_supports_gemini_31_flash_lite_preview(self):
        on_demand_pricing = _resolve_pricing_tier(
            "gemini-3.1-flash-lite-preview",
            "ON_DEMAND",
        )
        priority_pricing = _resolve_pricing_tier(
            "gemini-3.1-flash-lite-preview",
            "ON_DEMAND_PRIORITY",
        )
        flex_pricing = _resolve_pricing_tier(
            "gemini-3.1-flash-lite-preview",
            "ON_DEMAND_FLEX",
        )

        self.assertIsNotNone(on_demand_pricing)
        self.assertEqual(on_demand_pricing.input_usd_per_million_tokens, 0.25)
        self.assertEqual(on_demand_pricing.output_usd_per_million_tokens, 1.5)
        self.assertIsNotNone(priority_pricing)
        self.assertEqual(priority_pricing.input_usd_per_million_tokens, 0.45)
        self.assertEqual(priority_pricing.output_usd_per_million_tokens, 2.7)
        self.assertIsNotNone(flex_pricing)
        self.assertEqual(flex_pricing.input_usd_per_million_tokens, 0.13)
        self.assertEqual(flex_pricing.output_usd_per_million_tokens, 0.75)

    def test_generate_trajectories_cancels_pending_futures_on_keyboard_interrupt(self):
        runtime_config = RuntimeConfig(
            composite_task="PrepareCoffee",
            num_runs=2,
            model="gemini-3-flash-preview",
            sdk="google-genai",
            project="demo-project",
            location="global",
            temperature=0.5,
            max_workers=1,
            max_retries=1,
        )

        class InterruptingFuture:
            def __init__(self, *, raises_keyboard_interrupt=False):
                self._raises_keyboard_interrupt = raises_keyboard_interrupt
                self.cancel = mock.Mock(return_value=True)

            def result(self):
                if self._raises_keyboard_interrupt:
                    raise KeyboardInterrupt
                return mock.sentinel.unused_result

        class FakeExecutor:
            def __init__(self):
                self.shutdown = mock.Mock()
                self.submitted_futures = []

            def submit(self, *args, **kwargs):
                future = InterruptingFuture(
                    raises_keyboard_interrupt=not self.submitted_futures
                )
                self.submitted_futures.append(future)
                return future

        fake_executor = FakeExecutor()

        with mock.patch(
            "data_generation.task_level.generation.costs._build_preflight_cost_estimate_summary",
            return_value={"best_case_total_usd": None, "worst_case_total_usd": None},
        ):
            with mock.patch(
                "data_generation.task_level.generation.progress._create_progress_handles",
                return_value=ProgressHandles(
                    display=None,
                    overall_progress=None,
                    trajectory_progress_bars=[None, None],
                    log_writer=None,
                ),
            ):
                with mock.patch(
                    "data_generation.task_level.generation.progress._close_progress_handles"
                ) as close_progress_handles:
                    with mock.patch(
                        "data_generation.task_level.runtime.on_demand_generation.ThreadPoolExecutor",
                        return_value=fake_executor,
                    ):
                        with mock.patch(
                            "data_generation.task_level.runtime.on_demand_generation.as_completed",
                            side_effect=lambda futures: list(futures),
                        ):
                            with self.assertRaises(KeyboardInterrupt):
                                generate_trajectories(
                                    runtime_config,
                                    show_progress=False,
                                )

        self.assertEqual(len(fake_executor.submitted_futures), 2)
        for future in fake_executor.submitted_futures:
            future.cancel.assert_called_once_with()
        fake_executor.shutdown.assert_called_once_with(
            wait=False,
            cancel_futures=True,
        )
        close_progress_handles.assert_called_once()

    def test_batch_processing_requires_gcs_prefix(self):
        runtime_config = RuntimeConfig(
            composite_task="PrepareCoffee",
            num_runs=1,
            model="gemini-3-flash-preview",
            sdk="google-genai",
            project="demo-project",
            location="global",
            temperature=0.5,
            max_workers=1,
            max_retries=1,
            batch_processing=True,
        )

        with self.assertRaises(TrajectoryGenerationError) as raised:
            generate_trajectories(runtime_config, show_progress=False)

        self.assertIn("--batch-gcs-prefix", str(raised.exception))

    def test_batch_processing_single_round_success_preserves_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_config = RuntimeConfig(
                composite_task="PrepareCoffee",
                num_runs=2,
                model="gemini-3-flash-preview",
                sdk="google-genai",
                project="demo-project",
                location="global",
                temperature=0.5,
                max_workers=8,
                max_retries=2,
                batch_processing=True,
                batch_gcs_prefix="gs://demo-bucket/batch-prefix",
            )
            batch_run_context = BatchRunContext(
                run_id="20260311T000000Z",
                local_staging_dir=Path(tmpdir) / "batch" / "20260311T000000Z",
                gcs_run_prefix="gs://demo-bucket/batch-prefix/prepare_coffee/20260311T000000Z",
            )
            round_output_prefix = (
                "gs://demo-bucket/batch-prefix/prepare_coffee/20260311T000000Z/"
                "round-01/output"
            )
            batch_storage = FakeBatchStorage(
                {
                    round_output_prefix: make_batch_download(
                        "gs://demo-bucket/output/predictions.jsonl",
                        [
                            make_batch_output_row(
                                "traj-000001-attempt-00",
                                candidate=make_valid_candidate(
                                    communicate_messages=(
                                        "I will handle the machine.",
                                        "I will clear the cabinet route.",
                                    ),
                                    action_agents=("agent_1", "agent_0", "agent_0"),
                                ),
                            ),
                            make_batch_output_row(
                                "traj-000000-attempt-00",
                                candidate=make_valid_candidate(),
                            ),
                        ],
                    )
                }
            )
            batch_service = FakeBatchService([make_batch_job("batchJobs/round-01")])

            with mock.patch(
                "data_generation.task_level.runtime.batch_generation._build_batch_run_context",
                return_value=batch_run_context,
            ):
                with mock.patch(
                    "data_generation.task_level.runtime.batch_generation._build_batch_service_from_runtime",
                    return_value=batch_service,
                ):
                    with mock.patch(
                        "data_generation.task_level.runtime.batch_generation._build_batch_storage_from_runtime",
                        return_value=batch_storage,
                    ):
                        payload = generate_trajectories(
                            runtime_config,
                            show_progress=False,
                        )

        self.assertEqual(
            [trajectory["trajectory_id"] for trajectory in payload["trajectories"]],
            ["traj_000000", "traj_000001"],
        )
        self.assertEqual(
            payload["trajectories"][0]["generation_usage"]["successful_attempt_number"],
            1,
        )
        self.assertEqual(
            payload["trajectories"][1]["generation_usage"]["successful_attempt_number"],
            1,
        )
        self.assertEqual(len(batch_storage.upload_calls), 1)
        self.assertIn(
            "\"variation_key\": \"traj-000000-attempt-00\"",
            batch_storage.upload_calls[0]["text"],
        )
        self.assertIn(
            "\"variation_key\": \"traj-000001-attempt-00\"",
            batch_storage.upload_calls[0]["text"],
        )
        self.assertEqual(batch_service.create_calls[0]["output_prefix"], round_output_prefix)

    def test_batch_request_payload_includes_optional_thinking_level(self):
        runtime_config = RuntimeConfig(
            composite_task="PrepareCoffee",
            num_runs=1,
            model="gemini-3.1-flash-lite-preview",
            sdk="google-genai",
            project="demo-project",
            location="global",
            temperature=0.4,
            max_workers=1,
            max_retries=2,
            thinking_level="minimal",
            batch_processing=True,
            batch_gcs_prefix="gs://demo-bucket/batch-prefix",
        )

        payload = _batch_request_payload(
            prompt="test prompt",
            runtime_config=runtime_config,
            task_definition=PREPARE_COFFEE_TASK,
        )

        self.assertEqual(
            payload["generationConfig"]["thinkingConfig"],
            {"thinkingLevel": "minimal"},
        )

    def test_batch_processing_retries_only_outstanding_trajectories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_config = RuntimeConfig(
                composite_task="PrepareCoffee",
                num_runs=2,
                model="gemini-3-flash-preview",
                sdk="google-genai",
                project="demo-project",
                location="global",
                temperature=0.5,
                max_workers=3,
                max_retries=2,
                batch_processing=True,
                batch_gcs_prefix="gs://demo-bucket/batch-prefix",
            )
            batch_run_context = BatchRunContext(
                run_id="20260311T000000Z",
                local_staging_dir=Path(tmpdir) / "batch" / "20260311T000000Z",
                gcs_run_prefix="gs://demo-bucket/batch-prefix/prepare_coffee/20260311T000000Z",
            )
            round_one_output_prefix = (
                "gs://demo-bucket/batch-prefix/prepare_coffee/20260311T000000Z/"
                "round-01/output"
            )
            round_two_output_prefix = (
                "gs://demo-bucket/batch-prefix/prepare_coffee/20260311T000000Z/"
                "round-02/output"
            )
            batch_storage = FakeBatchStorage(
                {
                    round_one_output_prefix: make_batch_download(
                        "gs://demo-bucket/output/round-01.jsonl",
                        [
                            make_batch_output_row(
                                "traj-000000-attempt-00",
                                candidate=make_valid_candidate(),
                            ),
                            make_batch_output_row(
                                "traj-000001-attempt-00",
                                status="internal error",
                            ),
                        ],
                    ),
                    round_two_output_prefix: make_batch_download(
                        "gs://demo-bucket/output/round-02.jsonl",
                        [
                            make_batch_output_row(
                                "traj-000001-attempt-01",
                                candidate=make_valid_candidate(
                                    communicate_messages=(
                                        "I will take the cabinet.",
                                        "I will set up the machine.",
                                    ),
                                    action_agents=("agent_1", "agent_0", "agent_0"),
                                ),
                            )
                        ],
                    ),
                }
            )
            batch_service = FakeBatchService(
                [
                    make_batch_job("batchJobs/round-01"),
                    make_batch_job("batchJobs/round-02"),
                ]
            )

            with mock.patch(
                "data_generation.task_level.runtime.batch_generation._build_batch_run_context",
                return_value=batch_run_context,
            ):
                with mock.patch(
                    "data_generation.task_level.runtime.batch_generation._build_batch_service_from_runtime",
                    return_value=batch_service,
                ):
                    with mock.patch(
                        "data_generation.task_level.runtime.batch_generation._build_batch_storage_from_runtime",
                        return_value=batch_storage,
                    ):
                        payload = generate_trajectories(
                            runtime_config,
                            show_progress=False,
                        )

        self.assertEqual(len(batch_service.create_calls), 2)
        self.assertEqual(
            payload["trajectories"][1]["generation_usage"]["successful_attempt_number"],
            2,
        )
        self.assertIn(
            "\"variation_key\": \"traj-000001-attempt-01\"",
            batch_storage.upload_calls[1]["text"],
        )
        self.assertNotIn(
            "\"variation_key\": \"traj-000000-attempt-01\"",
            batch_storage.upload_calls[1]["text"],
        )

    def test_batch_processing_disable_validation_keeps_invalid_trajectory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_config = RuntimeConfig(
                composite_task="PrepareCoffee",
                num_runs=1,
                model="gemini-3-flash-preview",
                sdk="google-genai",
                project="demo-project",
                location="global",
                temperature=0.5,
                max_workers=2,
                max_retries=2,
                disable_validation=True,
                batch_processing=True,
                batch_gcs_prefix="gs://demo-bucket/batch-prefix",
            )
            batch_run_context = BatchRunContext(
                run_id="20260311T000000Z",
                local_staging_dir=Path(tmpdir) / "batch" / "20260311T000000Z",
                gcs_run_prefix="gs://demo-bucket/batch-prefix/prepare_coffee/20260311T000000Z",
            )
            round_output_prefix = (
                "gs://demo-bucket/batch-prefix/prepare_coffee/20260311T000000Z/"
                "round-01/output"
            )
            batch_storage = FakeBatchStorage(
                {
                    round_output_prefix: make_batch_download(
                        "gs://demo-bucket/output/round-01.jsonl",
                        [
                            make_batch_output_row(
                                "traj-000000-attempt-00",
                                candidate=make_invalid_candidate_missing_initial_communication(),
                            )
                        ],
                    )
                }
            )
            batch_service = FakeBatchService([make_batch_job("batchJobs/round-01")])

            with mock.patch(
                "data_generation.task_level.runtime.batch_generation._build_batch_run_context",
                return_value=batch_run_context,
            ):
                with mock.patch(
                    "data_generation.task_level.runtime.batch_generation._build_batch_service_from_runtime",
                    return_value=batch_service,
                ):
                    with mock.patch(
                        "data_generation.task_level.runtime.batch_generation._build_batch_storage_from_runtime",
                        return_value=batch_storage,
                    ):
                        payload = generate_trajectories(
                            runtime_config,
                            show_progress=False,
                        )

        self.assertEqual(len(batch_service.create_calls), 1)
        self.assertFalse(payload["trajectories"][0]["validation"]["is_valid"])
        self.assertTrue(payload["trajectories"][0]["validation"]["validation_disabled"])
        self.assertNotIn(
            "successful_attempt_number",
            payload["trajectories"][0]["generation_usage"],
        )

    def test_batch_processing_uses_flex_pricing_for_cost_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_config = RuntimeConfig(
                composite_task="PrepareCoffee",
                num_runs=1,
                model="gemini-3-flash-preview",
                sdk="google-genai",
                project="demo-project",
                location="global",
                temperature=0.5,
                max_workers=1,
                max_retries=1,
                batch_processing=True,
                batch_gcs_prefix="gs://demo-bucket/batch-prefix",
            )
            batch_run_context = BatchRunContext(
                run_id="20260311T000000Z",
                local_staging_dir=Path(tmpdir) / "batch" / "20260311T000000Z",
                gcs_run_prefix="gs://demo-bucket/batch-prefix/prepare_coffee/20260311T000000Z",
            )
            round_output_prefix = (
                "gs://demo-bucket/batch-prefix/prepare_coffee/20260311T000000Z/"
                "round-01/output"
            )
            batch_storage = FakeBatchStorage(
                {
                    round_output_prefix: make_batch_download(
                        "gs://demo-bucket/output/round-01.jsonl",
                        [
                            make_batch_output_row(
                                "traj-000000-attempt-00",
                                candidate=make_valid_candidate(),
                                usage_metadata=make_batch_usage_metadata(),
                            )
                        ],
                    )
                }
            )
            batch_service = FakeBatchService([make_batch_job("batchJobs/round-01")])

            with mock.patch(
                "data_generation.task_level.runtime.batch_generation._build_batch_run_context",
                return_value=batch_run_context,
            ):
                with mock.patch(
                    "data_generation.task_level.runtime.batch_generation._build_batch_service_from_runtime",
                    return_value=batch_service,
                ):
                    with mock.patch(
                        "data_generation.task_level.runtime.batch_generation._build_batch_storage_from_runtime",
                        return_value=batch_storage,
                    ):
                        payload = generate_trajectories(
                            runtime_config,
                            show_progress=False,
                        )

        self.assertEqual(
            payload["trajectories"][0]["generation_usage"]["traffic_type"],
            "ON_DEMAND_FLEX",
        )
        self.assertEqual(
            payload["cost_summary"]["pricing"],
            {
                "model": "gemini-3-flash-preview",
                "input_usd_per_million_tokens": 0.25,
                "output_usd_per_million_tokens": 1.5,
            },
        )
        self.assertAlmostEqual(
            payload["cost_summary"]["total_cost_usd"],
            0.0006,
        )

    def test_batch_processing_cancels_active_jobs_on_keyboard_interrupt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_config = RuntimeConfig(
                composite_task="PrepareCoffee",
                num_runs=1,
                model="gemini-3-flash-preview",
                sdk="google-genai",
                project="demo-project",
                location="global",
                temperature=0.5,
                max_workers=1,
                max_retries=1,
                batch_processing=True,
                batch_gcs_prefix="gs://demo-bucket/batch-prefix",
            )
            batch_run_context = BatchRunContext(
                run_id="20260311T000000Z",
                local_staging_dir=Path(tmpdir) / "batch" / "20260311T000000Z",
                gcs_run_prefix="gs://demo-bucket/batch-prefix/prepare_coffee/20260311T000000Z",
            )
            batch_storage = FakeBatchStorage()
            batch_service = FakeBatchService(
                [make_batch_job("batchJobs/interrupt", state="JOB_STATE_RUNNING")],
                get_side_effect=KeyboardInterrupt(),
            )

            with mock.patch(
                "data_generation.task_level.runtime.batch_generation._build_batch_run_context",
                return_value=batch_run_context,
            ):
                with mock.patch(
                    "data_generation.task_level.runtime.batch_generation._build_batch_service_from_runtime",
                    return_value=batch_service,
                ):
                    with mock.patch(
                        "data_generation.task_level.runtime.batch_generation._build_batch_storage_from_runtime",
                        return_value=batch_storage,
                    ):
                        with self.assertRaises(KeyboardInterrupt) as raised:
                            generate_trajectories(
                                runtime_config,
                                show_progress=False,
                            )

        self.assertEqual(str(raised.exception), BATCH_INTERRUPTED_MESSAGE)
        self.assertEqual(batch_service.cancel_calls, ["batchJobs/interrupt"])

    def test_batch_processing_job_failure_raises_clear_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_config = RuntimeConfig(
                composite_task="PrepareCoffee",
                num_runs=1,
                model="gemini-3-flash-preview",
                sdk="google-genai",
                project="demo-project",
                location="global",
                temperature=0.5,
                max_workers=1,
                max_retries=1,
                batch_processing=True,
                batch_gcs_prefix="gs://demo-bucket/batch-prefix",
            )
            batch_run_context = BatchRunContext(
                run_id="20260311T000000Z",
                local_staging_dir=Path(tmpdir) / "batch" / "20260311T000000Z",
                gcs_run_prefix="gs://demo-bucket/batch-prefix/prepare_coffee/20260311T000000Z",
            )
            batch_storage = FakeBatchStorage()
            failed_job = make_batch_job(
                "batchJobs/failed",
                state="JOB_STATE_FAILED",
                error="permission denied",
            )
            batch_service = FakeBatchService([failed_job])

            with mock.patch(
                "data_generation.task_level.runtime.batch_generation._build_batch_run_context",
                return_value=batch_run_context,
            ):
                with mock.patch(
                    "data_generation.task_level.runtime.batch_generation._build_batch_service_from_runtime",
                    return_value=batch_service,
                ):
                    with mock.patch(
                        "data_generation.task_level.runtime.batch_generation._build_batch_storage_from_runtime",
                        return_value=batch_storage,
                    ):
                        with self.assertRaises(TrajectoryGenerationError) as raised:
                            generate_trajectories(
                                runtime_config,
                                show_progress=False,
                            )

        self.assertIn("JOB_STATE_FAILED", str(raised.exception))
        self.assertIn("permission denied", str(raised.exception))

    def test_saved_valid_trajectory_uses_agent_and_communication_args(self):
        runtime_config = RuntimeConfig(
            composite_task="PrepareCoffee",
            num_runs=1,
            model="gemini-3-flash-preview",
            sdk="google-genai",
            project="demo-project",
            location="global",
            temperature=0.5,
            max_workers=1,
            max_retries=1,
        )

        payload = generate_trajectories(
            runtime_config,
            client_factory=lambda: SequencedFakeClient(
                [make_valid_candidate(include_agents=False)]
            ),
            show_progress=False,
        )

        trajectory = payload["trajectories"][0]
        self.assertEqual(
            trajectory["agents"],
            [{"agent": "agent_0"}, {"agent": "agent_1"}],
        )
        self.assertEqual(trajectory["steps"][0]["tool"], "communicate")
        self.assertEqual(
            trajectory["steps"][0]["args"],
            {
                "to": "agent_1",
                "message": "I will grab the mug.",
            },
        )
        self.assertNotIn("entity_refs", trajectory["steps"][0])
        self.assertNotIn("message", trajectory["steps"][0])

    def test_saved_generated_trajectory_omits_get_image_and_image_paths(self):
        runtime_config = RuntimeConfig(
            composite_task="PrepareCoffee",
            num_runs=1,
            model="gemini-3-flash-preview",
            sdk="google-genai",
            project="demo-project",
            location="global",
            temperature=0.5,
            max_workers=1,
            max_retries=1,
        )
        payload = generate_trajectories(
            runtime_config,
            client_factory=lambda: SequencedFakeClient(
                [make_valid_candidate(include_agents=False)]
            ),
            show_progress=False,
        )

        self.assertTrue(
            all(step["tool"] != "get_image" for step in payload["trajectories"][0]["steps"])
        )
        self.assertTrue(
            all("image_path" not in step for step in payload["trajectories"][0]["steps"])
        )

    def test_generated_payload_includes_prompt_sidecars(self):
        runtime_config = RuntimeConfig(
            composite_task="PrepareCoffee",
            num_runs=1,
            model="gemini-3-flash-preview",
            sdk="google-genai",
            project="demo-project",
            location="global",
            temperature=0.5,
            max_workers=1,
            max_retries=1,
        )

        payload = generate_trajectories(
            runtime_config,
            client_factory=lambda: SequencedFakeClient([make_valid_candidate()]),
            show_progress=False,
        )

        self.assertEqual(
            payload["trajectory_prompts"],
            [
                {
                    "trajectory_id": "traj_000000",
                    "prompt": build_prepare_coffee_prompt_for_run(
                        "traj-000000-attempt-00",
                        run_index=0,
                    ),
                }
            ],
        )
        self.assertEqual(
            payload["attempt_prompts"],
            [
                {
                    "run_id": "traj_000000",
                    "attempt_number": 1,
                    "prompt": build_prepare_coffee_prompt_for_run(
                        "traj-000000-attempt-00",
                        run_index=0,
                    ),
                }
            ],
        )
        self.assertEqual(
            payload["model_config"],
            {
                "reasoning": {"thinking_level": None},
                "sampling": {"temperature": 0.5, "strategy": "base"},
            },
        )

    def test_generated_payload_includes_retry_attempt_prompts(self):
        runtime_config = RuntimeConfig(
            composite_task="PrepareCoffee",
            num_runs=1,
            model="gemini-3-flash-preview",
            sdk="google-genai",
            project="demo-project",
            location="global",
            temperature=0.5,
            max_workers=1,
            max_retries=2,
        )

        payload = generate_trajectories(
            runtime_config,
            client_factory=lambda: SequencedFakeClient(
                [
                    make_invalid_candidate_missing_initial_communication(),
                    make_valid_candidate(),
                ]
            ),
            show_progress=False,
        )

        self.assertEqual(len(payload["attempt_prompts"]), 2)
        self.assertEqual(payload["attempt_prompts"][0]["run_id"], "traj_000000")
        self.assertEqual(payload["attempt_prompts"][0]["attempt_number"], 1)
        self.assertEqual(payload["attempt_prompts"][1]["attempt_number"], 2)
        self.assertNotIn(
            "Previous attempt failed validation.",
            payload["attempt_prompts"][0]["prompt"],
        )
        self.assertIn(
            "Previous attempt failed validation.",
            payload["attempt_prompts"][1]["prompt"],
        )

    def test_verbalized_on_demand_retries_only_failed_run_and_fills_remaining_slots(self):
        runtime_config = RuntimeConfig(
            composite_task="PrepareCoffee",
            num_runs=3,
            model="gemini-3.1-flash-lite-preview",
            sdk="google-genai",
            project="demo-project",
            location="global",
            temperature=0.5,
            max_workers=1,
            max_retries=2,
            sampling="verbalized",
            verbalized_k=3,
        )

        def labeled_valid_candidate(label):
            return make_valid_candidate(
                communicate_messages=(
                    f"{label} agent_0 plan",
                    f"{label} agent_1 plan",
                )
            )

        clients = [
            SequencedFakeClient(
                [
                    make_verbalized_response(
                        make_invalid_candidate_missing_initial_communication(),
                        labeled_valid_candidate("run0-attempt1-b"),
                        labeled_valid_candidate("run0-attempt1-c"),
                        probabilities=[0.4, 0.3, 0.3],
                    ),
                    make_verbalized_response(
                        labeled_valid_candidate("run0-attempt2-a"),
                        labeled_valid_candidate("run0-attempt2-b"),
                        labeled_valid_candidate("run0-attempt2-c"),
                        probabilities=[0.5, 0.3, 0.2],
                    ),
                ]
            ),
            SequencedFakeClient(
                [
                    make_verbalized_response(
                        labeled_valid_candidate("run1-attempt1-a"),
                        labeled_valid_candidate("run1-attempt1-b"),
                        labeled_valid_candidate("run1-attempt1-c"),
                        probabilities=[0.5, 0.3, 0.2],
                    )
                ]
            ),
            SequencedFakeClient(
                [
                    make_verbalized_response(
                        labeled_valid_candidate("run2-attempt1-a"),
                        labeled_valid_candidate("run2-attempt1-b"),
                        labeled_valid_candidate("run2-attempt1-c"),
                        probabilities=[0.5, 0.3, 0.2],
                    )
                ]
            ),
        ]

        payload = generate_trajectories(
            runtime_config,
            client_factory=lambda: clients.pop(0),
            show_progress=False,
        )

        self.assertEqual(
            [
                trajectory["generation_usage"]["successful_attempt_number"]
                for trajectory in payload["trajectories"]
            ],
            [2, 2, 2, 1, 1, 1, 1, 1, 1],
        )
        self.assertEqual(
            [
                (prompt_entry["run_id"], prompt_entry["attempt_number"])
                for prompt_entry in payload["attempt_prompts"]
            ],
            [
                ("traj_000000", 1),
                ("traj_000000", 2),
                ("traj_000003", 1),
                ("traj_000006", 1),
            ],
        )

    def test_verbalized_on_demand_keeps_valid_candidates_across_attempts(self):
        runtime_config = RuntimeConfig(
            composite_task="PrepareCoffee",
            num_runs=1,
            model="gemini-3.1-flash-lite-preview",
            sdk="google-genai",
            project="demo-project",
            location="global",
            temperature=0.5,
            max_workers=1,
            max_retries=3,
            sampling="verbalized",
            verbalized_k=2,
        )
        client = PromptCapturingSequencedFakeClient(
            [
                make_verbalized_response(
                    make_invalid_candidate_missing_initial_communication(),
                    make_valid_candidate(
                        communicate_messages=(
                            "attempt1 agent_0 plan",
                            "attempt1 agent_1 plan",
                        )
                    ),
                    probabilities=[0.4, 0.6],
                ),
                make_verbalized_response(
                    make_valid_candidate(
                        communicate_messages=(
                            "attempt2 agent_0 first",
                            "attempt2 agent_1 first",
                        )
                    ),
                    make_valid_candidate(
                        communicate_messages=(
                            "attempt2 agent_0 second",
                            "attempt2 agent_1 second",
                        )
                    ),
                    probabilities=[0.7, 0.3],
                ),
            ]
        )

        payload = generate_trajectories(
            runtime_config,
            client_factory=lambda: client,
            show_progress=False,
        )

        self.assertEqual(len(client.prompts), 2)
        self.assertEqual(len(payload["trajectories"]), 2)
        first_messages = [
            trajectory["steps"][0]["args"]["message"]
            for trajectory in payload["trajectories"]
        ]
        self.assertEqual(
            first_messages,
            [
                "attempt1 agent_0 plan",
                "attempt2 agent_0 first",
            ],
        )
        self.assertEqual(
            [entry["attempt_number"] for entry in payload["attempt_prompts"]],
            [1, 2],
        )
        self.assertIn(
            "MissingInitialCommunicationSemanticValidationError",
            client.prompts[1],
        )
        self.assertTrue(
            all(
                trajectory["generation_usage"]["successful_attempt_number"] == 2
                for trajectory in payload["trajectories"]
            )
        )
        self.assertTrue(
            all(
                trajectory["generation_usage"]["retry_costs_included"] is True
                for trajectory in payload["trajectories"]
            )
        )

    def test_generated_payload_includes_raw_output_sidecars(self):
        runtime_config = RuntimeConfig(
            composite_task="PrepareCoffee",
            num_runs=1,
            model="gemini-3-flash-preview",
            sdk="google-genai",
            project="demo-project",
            location="global",
            temperature=0.5,
            max_workers=1,
            max_retries=1,
        )
        raw_output = json.dumps(make_valid_candidate())

        payload = generate_trajectories(
            runtime_config,
            client_factory=lambda: SequencedFakeClient([raw_output]),
            show_progress=False,
        )

        self.assertEqual(
            payload["trajectory_outputs"],
            [
                {
                    "trajectory_id": "traj_000000",
                    "raw_output": raw_output,
                }
            ],
        )
        self.assertNotIn("raw_output", payload["trajectories"][0])

    def test_verbalized_sampling_saves_all_candidates_and_splits_usage(self):
        runtime_config = RuntimeConfig(
            composite_task="PrepareCoffee",
            num_runs=1,
            model="gemini-3-flash-preview",
            sdk="google-genai",
            project="demo-project",
            location="global",
            temperature=0.5,
            max_workers=1,
            max_retries=1,
            sampling="verbalized",
            verbalized_k=2,
        )
        raw_response = make_verbalized_response(
            make_valid_candidate(include_agents=False),
            make_alternative_valid_candidate(),
            probabilities=[0.65, 0.35],
        )

        payload = generate_trajectories(
            runtime_config,
            client_factory=lambda: SequencedFakeClient(
                [
                    GenerationResult(
                        payload=raw_response,
                        usage=GenerationUsage(
                            prompt_tokens=900,
                            candidates_tokens=300,
                            thoughts_tokens=0,
                            total_tokens=1200,
                        ),
                    )
                ]
            ),
            show_progress=False,
        )

        self.assertEqual(payload["num_runs"], 1)
        self.assertEqual(payload["num_trajectories"], 2)
        self.assertEqual(
            [trajectory["trajectory_id"] for trajectory in payload["trajectories"]],
            ["traj_000000", "traj_000001"],
        )
        self.assertEqual(
            [trajectory["sampling_metadata"]["probability"] for trajectory in payload["trajectories"]],
            [0.65, 0.35],
        )
        self.assertEqual(
            sum(
                trajectory["generation_usage"]["prompt_tokens"]
                for trajectory in payload["trajectories"]
            ),
            900,
        )
        self.assertEqual(
            sum(
                trajectory["generation_usage"]["output_tokens"]
                for trajectory in payload["trajectories"]
            ),
            300,
        )
        self.assertEqual(
            payload["trajectory_outputs"][0]["raw_output"]["probability"],
            0.65,
        )
        self.assertEqual(payload["cost_summary"]["prompt_tokens"], 900)
        self.assertEqual(payload["cost_summary"]["output_tokens"], 300)
        self.assertEqual(payload["cost_summary"]["total_tokens"], 1200)
        self.assertIn(
            "counted once per model response",
            payload["cost_summary"]["notes"][-1],
        )

    def test_verbalized_sampling_split_usage_recomputes_total_tokens(self):
        runtime_config = RuntimeConfig(
            composite_task="PrepareCoffee",
            num_runs=1,
            model="gemini-3-flash-preview",
            sdk="google-genai",
            project="demo-project",
            location="global",
            temperature=0.5,
            max_workers=1,
            max_retries=1,
            sampling="verbalized",
            verbalized_k=2,
        )

        payload = generate_trajectories(
            runtime_config,
            client_factory=lambda: SequencedFakeClient(
                [
                    GenerationResult(
                        payload=make_verbalized_response(
                            make_valid_candidate(include_agents=False),
                            make_alternative_valid_candidate(),
                            probabilities=[0.6, 0.4],
                        ),
                        usage=GenerationUsage(
                            prompt_tokens=1,
                            candidates_tokens=1,
                            thoughts_tokens=1,
                            total_tokens=3,
                        ),
                    )
                ]
            ),
            show_progress=False,
        )

        self.assertEqual(
            [trajectory["generation_usage"]["total_tokens"] for trajectory in payload["trajectories"]],
            [3, 0],
        )
        for trajectory in payload["trajectories"]:
            generation_usage = trajectory["generation_usage"]
            self.assertEqual(
                generation_usage["total_tokens"],
                generation_usage["prompt_tokens"]
                + generation_usage["output_tokens"]
                + generation_usage["reasoning_tokens"],
            )

    def test_batch_processing_verbalized_sampling_saves_all_candidates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_config = RuntimeConfig(
                composite_task="PrepareCoffee",
                num_runs=1,
                model="gemini-3-flash-preview",
                sdk="google-genai",
                project="demo-project",
                location="global",
                temperature=0.5,
                max_workers=1,
                max_retries=1,
                sampling="verbalized",
                verbalized_k=2,
                batch_processing=True,
                batch_gcs_prefix="gs://demo-bucket/batch-prefix",
            )
            batch_run_context = BatchRunContext(
                run_id="20260311T000000Z",
                local_staging_dir=Path(tmpdir) / "batch" / "20260311T000000Z",
                gcs_run_prefix="gs://demo-bucket/batch-prefix/prepare_coffee/20260311T000000Z",
            )
            round_output_prefix = (
                "gs://demo-bucket/batch-prefix/prepare_coffee/20260311T000000Z/"
                "round-01/output"
            )
            batch_storage = FakeBatchStorage(
                {
                    round_output_prefix: make_batch_download(
                        "gs://demo-bucket/output/predictions.jsonl",
                        [
                            make_batch_output_row(
                                "traj-000000-attempt-00",
                                candidate=make_verbalized_response(
                                    make_valid_candidate(),
                                    make_alternative_valid_candidate(),
                                    probabilities=[0.8, 0.2],
                                ),
                            )
                        ],
                    )
                }
            )
            batch_service = FakeBatchService([make_batch_job("batchJobs/round-01")])

            with mock.patch(
                "data_generation.task_level.runtime.batch_generation._build_batch_run_context",
                return_value=batch_run_context,
            ):
                with mock.patch(
                    "data_generation.task_level.runtime.batch_generation._build_batch_service_from_runtime",
                    return_value=batch_service,
                ):
                    with mock.patch(
                        "data_generation.task_level.runtime.batch_generation._build_batch_storage_from_runtime",
                        return_value=batch_storage,
                    ):
                        payload = generate_trajectories(
                            runtime_config,
                            show_progress=False,
                        )

        self.assertEqual(payload["num_trajectories"], 2)
        self.assertEqual(
            [trajectory["trajectory_id"] for trajectory in payload["trajectories"]],
            ["traj_000000", "traj_000001"],
        )
        self.assertEqual(
            [trajectory["sampling_metadata"]["probability"] for trajectory in payload["trajectories"]],
            [0.8, 0.2],
        )

    def test_preflight_cost_estimate_counts_verbalized_prompt_once_per_run(self):
        runtime_config = RuntimeConfig(
            composite_task="PrepareCoffee",
            num_runs=2,
            model="gemini-3-flash-preview",
            sdk="google-genai",
            project="demo-project",
            location="global",
            temperature=0.5,
            max_workers=1,
            max_retries=3,
            sampling="verbalized",
            verbalized_k=3,
        )

        summary = _build_preflight_cost_estimate_summary(
            runtime_config,
            PREPARE_COFFEE_TASK,
        )

        self.assertEqual(
            summary["best_case_tokens"]["prompt"],
            PREPARE_COFFEE_TASK.preflight_token_estimate.prompt_tokens * 2 * 3,
        )
        self.assertEqual(
            summary["best_case_tokens"]["output"],
            PREPARE_COFFEE_TASK.preflight_token_estimate.output_tokens * 3 * 2 * 3,
        )
        self.assertIn(
            "counted once per model response",
            summary["notes"][-1],
        )

    def test_preflight_cost_estimate_uses_matching_history_when_available(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            history_dir = output_root / "prepare_coffee" / "20260314T010000Z"
            history_dir.mkdir(parents=True, exist_ok=True)
            (history_dir / "cost_summary.json").write_text(
                json.dumps(
                    {
                        "composite_task": "PrepareCoffee",
                        "sdk": "google-genai",
                        "model": "gemini-3.1-flash-lite-preview",
                        "num_runs": 2,
                        "num_trajectories": 6,
                        "generated_at": "2026-03-14T01:00:00+00:00",
                        "model_config": {
                            "reasoning": {"thinking_level": "low"},
                            "sampling": {
                                "temperature": 0.6,
                                "strategy": "verbalized",
                                "verbalized_k": 3,
                            },
                        },
                        "trajectory_costs": [
                            {
                                "trajectory_id": "traj_000000",
                                "generation_usage": {
                                    "successful_attempt_number": 1,
                                    "prompt_tokens": 100,
                                    "output_tokens": 200,
                                    "reasoning_tokens": 10,
                                    "total_tokens": 310,
                                    "usage_source": "api_usage_metadata",
                                    "traffic_type": "ON_DEMAND",
                                    "observed_cost_usd": 0.0004,
                                },
                            },
                            {
                                "trajectory_id": "traj_000001",
                                "generation_usage": {
                                    "successful_attempt_number": 1,
                                    "prompt_tokens": 100,
                                    "output_tokens": 200,
                                    "reasoning_tokens": 10,
                                    "total_tokens": 310,
                                    "usage_source": "api_usage_metadata",
                                    "traffic_type": "ON_DEMAND",
                                    "observed_cost_usd": 0.0004,
                                },
                            },
                            {
                                "trajectory_id": "traj_000002",
                                "generation_usage": {
                                    "successful_attempt_number": 1,
                                    "prompt_tokens": 100,
                                    "output_tokens": 200,
                                    "reasoning_tokens": 10,
                                    "total_tokens": 310,
                                    "usage_source": "api_usage_metadata",
                                    "traffic_type": "ON_DEMAND",
                                    "observed_cost_usd": 0.0004,
                                },
                            },
                            {
                                "trajectory_id": "traj_000003",
                                "generation_usage": {
                                    "successful_attempt_number": 2,
                                    "prompt_tokens": 120,
                                    "output_tokens": 240,
                                    "reasoning_tokens": 20,
                                    "total_tokens": 380,
                                    "usage_source": "api_usage_metadata",
                                    "traffic_type": "ON_DEMAND",
                                    "observed_cost_usd": 0.0005,
                                },
                            },
                            {
                                "trajectory_id": "traj_000004",
                                "generation_usage": {
                                    "successful_attempt_number": 2,
                                    "prompt_tokens": 120,
                                    "output_tokens": 240,
                                    "reasoning_tokens": 20,
                                    "total_tokens": 380,
                                    "usage_source": "api_usage_metadata",
                                    "traffic_type": "ON_DEMAND",
                                    "observed_cost_usd": 0.0005,
                                },
                            },
                            {
                                "trajectory_id": "traj_000005",
                                "generation_usage": {
                                    "successful_attempt_number": 2,
                                    "prompt_tokens": 120,
                                    "output_tokens": 240,
                                    "reasoning_tokens": 20,
                                    "total_tokens": 380,
                                    "usage_source": "api_usage_metadata",
                                    "traffic_type": "ON_DEMAND",
                                    "observed_cost_usd": 0.0005,
                                },
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            runtime_config = RuntimeConfig(
                composite_task="PrepareCoffee",
                num_runs=2,
                model="gemini-3.1-flash-lite-preview",
                sdk="google-genai",
                project="demo-project",
                location="global",
                temperature=0.6,
                max_workers=1,
                max_retries=5,
                sampling="verbalized",
                verbalized_k=3,
                thinking_level="low",
                summary_path=output_root / "prepare_coffee" / "20260315T010000Z" / "summary.json",
            )

            with mock.patch(
                "data_generation.task_level.generation.costs.DEFAULT_OUTPUT_DIR",
                output_root,
            ):
                summary = _build_preflight_cost_estimate_summary(
                    runtime_config,
                    PREPARE_COFFEE_TASK,
                )

        self.assertEqual(
            summary["best_case_tokens"],
            {"prompt": 1020, "output": 2040, "reasoning": 150, "total": 3210},
        )
        self.assertAlmostEqual(summary["best_case_total_usd"], 0.0035)
        self.assertIn("observed API usage", summary["notes"][0])
        self.assertIn("saved API usage metadata", summary["notes"][2])
        self.assertTrue(
            any("Matched 2 prior run(s)" in note for note in summary["notes"])
        )
        self.assertTrue(
            any("counted once per model response" in note for note in summary["notes"])
        )

    def test_permission_denied_errors_are_treated_as_non_retryable(self):
        self.assertTrue(
            _is_non_retryable_generation_error(
                Exception(
                    "403 PERMISSION_DENIED. Permission 'aiplatform.endpoints.predict' denied."
                )
            )
        )

    def test_invalid_argument_errors_with_response_status_are_non_retryable(self):
        exc = Exception("400 INVALID_ARGUMENT. Request contains an invalid argument.")
        exc.response = types.SimpleNamespace(status_code=400)

        self.assertEqual(_generation_error_status_code(exc), 400)
        self.assertTrue(_is_non_retryable_generation_error(exc))

    def test_parallel_generation_retries_duplicate_and_preserves_order(self):
        runtime_config = RuntimeConfig(
            composite_task="PrepareCoffee",
            num_runs=2,
            model="gemini-3-flash-preview",
            sdk="google-genai",
            project="demo-project",
            location="global",
            temperature=0.5,
            max_workers=2,
            max_retries=3,
        )
        shared_responses = [
            make_valid_candidate(),
            make_valid_candidate(),
            make_valid_candidate(
                communicate_messages=(
                    "I will take cabinet duty.",
                    "I will handle the machine setup.",
                ),
                action_agents=("agent_1", "agent_0", "agent_0"),
                action_reasoning=(
                    "I can retrieve the mug first.",
                    "I can finish setup at the dispenser.",
                    "I should start brewing immediately.",
                ),
            ),
        ]

        def client_factory():
            return SequencedFakeClient(shared_responses)

        payload = generate_trajectories(
            runtime_config,
            client_factory=client_factory,
            show_progress=False,
        )
        self.assertEqual(len(payload["trajectories"]), 2)
        self.assertEqual(
            [trajectory["trajectory_id"] for trajectory in payload["trajectories"]],
            ["traj_000000", "traj_000001"],
        )
        self.assertNotEqual(
            payload["trajectories"][0]["validation"]["signature"],
            payload["trajectories"][1]["validation"]["signature"],
        )

    def test_cost_summary_uses_usage_metadata_when_available(self):
        runtime_config = RuntimeConfig(
            composite_task="PrepareCoffee",
            num_runs=1,
            model="gemini-3-flash-preview",
            sdk="google-genai",
            project="demo-project",
            location="global",
            temperature=0.5,
            max_workers=1,
            max_retries=3,
        )
        payload = generate_trajectories(
            runtime_config,
            client_factory=lambda: SequencedFakeClient(
                [
                    GenerationResult(
                        payload=make_valid_candidate(),
                        usage=GenerationUsage(
                            prompt_tokens=1000,
                            candidates_tokens=200,
                            thoughts_tokens=50,
                            total_tokens=1250,
                            traffic_type="ON_DEMAND",
                        ),
                    )
                ]
            ),
            show_progress=False,
        )

        self.assertEqual(
            payload["trajectories"][0]["generation_usage"]["usage_source"],
            "api_usage_metadata",
        )
        self.assertEqual(
            payload["trajectories"][0]["generation_usage"]["output_tokens"],
            200,
        )
        self.assertEqual(
            payload["trajectories"][0]["generation_usage"]["reasoning_tokens"],
            50,
        )
        self.assertAlmostEqual(
            payload["trajectories"][0]["generation_usage"]["observed_cost_usd"],
            0.0013,
        )
        self.assertEqual(
            payload["cost_summary"]["prompt_tokens"],
            1000,
        )
        self.assertEqual(
            payload["cost_summary"]["output_tokens"],
            200,
        )
        self.assertEqual(
            payload["cost_summary"]["reasoning_tokens"],
            50,
        )
        self.assertEqual(
            payload["cost_summary"]["total_tokens"],
            1250,
        )
        self.assertAlmostEqual(
            payload["cost_summary"]["input_cost_usd"],
            0.0005,
        )
        self.assertAlmostEqual(
            payload["cost_summary"]["output_cost_usd"],
            0.0008,
        )
        self.assertAlmostEqual(
            payload["cost_summary"]["total_cost_usd"],
            0.0013,
        )
        self.assertAlmostEqual(
            payload["cost_summary"]["average_trajectory_cost_usd"],
            0.0013,
        )
        self.assertEqual(
            payload["cost_summary"]["pricing"],
            {
                "model": "gemini-3-flash-preview",
                "input_usd_per_million_tokens": 0.5,
                "output_usd_per_million_tokens": 3.0,
            },
        )
        self.assertNotIn("usage_sources", payload["cost_summary"])
        self.assertNotIn(
            "all_trajectories_used_api_usage_metadata",
            payload["cost_summary"],
        )
        self.assertNotIn("pricing_supported", payload["cost_summary"])
        self.assertNotIn("pricing_reference", payload["cost_summary"])
        self.assertNotIn("currency", payload["cost_summary"])
        self.assertNotIn("cost_estimate", payload)

    def test_generate_trajectories_logs_cost_when_progress_enabled(self):
        runtime_config = RuntimeConfig(
            composite_task="PrepareCoffee",
            num_runs=1,
            model="gemini-3-flash-preview",
            sdk="google-genai",
            project="demo-project",
            location="global",
            temperature=0.5,
            max_workers=1,
            max_retries=3,
        )

        with mock.patch(
            "data_generation.task_level.generation.progress._log_runtime_message"
        ) as log_runtime_message:
            generate_trajectories(
                runtime_config,
                client_factory=lambda: SequencedFakeClient(
                    [
                        GenerationResult(
                            payload=make_valid_candidate(),
                            usage=GenerationUsage(
                                prompt_tokens=1000,
                                candidates_tokens=200,
                                thoughts_tokens=50,
                                total_tokens=1250,
                                traffic_type="ON_DEMAND",
                            ),
                        )
                    ]
                ),
                show_progress=True,
            )

        self.assertEqual(log_runtime_message.call_count, 1)
        projected_log = log_runtime_message.call_args_list[0]

        self.assertIn("Initial projected cost", projected_log.args[0])
        self.assertTrue(projected_log.args[0].startswith("Initial projected cost: $"))
        self.assertNotIn("best case", projected_log.args[0])
        self.assertNotIn("worst case", projected_log.args[0])
        self.assertEqual(projected_log.kwargs["enabled"], True)

    def test_progress_status_with_accumulated_cost_appends_suffix(self):
        self.assertEqual(
            _progress_status_with_accumulated_cost(
                "running",
                accumulated_cost_text="accumulated=$0.0013 projected=$0.0100",
            ),
            "running accumulated=$0.0013 projected=$0.0100",
        )
        self.assertEqual(
            _progress_status_with_accumulated_cost(
                "",
                accumulated_cost_text="accumulated=$0.0013 projected=$0.0100",
            ),
            "accumulated=$0.0013 projected=$0.0100",
        )

    def test_generate_trajectories_updates_overall_progress_with_accumulated_cost(self):
        runtime_config = RuntimeConfig(
            composite_task="PrepareCoffee",
            num_runs=2,
            model="gemini-3-flash-preview",
            sdk="google-genai",
            project="demo-project",
            location="global",
            temperature=0.5,
            max_workers=1,
            max_retries=1,
        )
        overall_progress = mock.Mock()
        trajectory_progress_bars = [mock.Mock(), mock.Mock()]
        progress_handles = ProgressHandles(
            display=None,
            overall_progress=overall_progress,
            trajectory_progress_bars=trajectory_progress_bars,
            log_writer=None,
        )
        client_responses = [
            SequencedFakeClient(
                [
                    GenerationResult(
                        payload=make_valid_candidate(),
                        usage=GenerationUsage(
                            prompt_tokens=1000,
                            candidates_tokens=200,
                            thoughts_tokens=50,
                            total_tokens=1250,
                            traffic_type="ON_DEMAND",
                        ),
                    )
                ]
            ),
            SequencedFakeClient(
                [
                    GenerationResult(
                        payload=make_alternative_valid_candidate(),
                        usage=GenerationUsage(
                            prompt_tokens=1000,
                            candidates_tokens=200,
                            thoughts_tokens=50,
                            total_tokens=1250,
                            traffic_type="ON_DEMAND",
                        ),
                    )
                ]
            ),
        ]

        with mock.patch(
            "data_generation.task_level.generation.progress._create_progress_handles",
            return_value=progress_handles,
        ):
            payload = generate_trajectories(
                runtime_config,
                client_factory=lambda: client_responses.pop(0),
                show_progress=False,
            )

        self.assertEqual(payload["num_trajectories"], 2)
        self.assertEqual(overall_progress.update.call_count, 2)
        self.assertEqual(
            overall_progress.set_postfix_str.call_args_list[0].args[0],
            "running accumulated=$0.0000 projected=NaN",
        )
        self.assertEqual(
            overall_progress.set_postfix_str.call_args_list[-1].args[0],
            "running accumulated=$0.0026 projected=$0.0026",
        )

    def test_generate_single_trajectory_updates_progress_status_with_cost(self):
        self.assertEqual(PROGRESS_BAR_WIDTH, 30)
        self.assertIn("{bar:30}", TQDM_BAR_FORMAT)
        runtime_config = RuntimeConfig(
            composite_task="PrepareCoffee",
            num_runs=1,
            model="gemini-3-flash-preview",
            sdk="google-genai",
            project="demo-project",
            location="global",
            temperature=0.5,
            max_workers=1,
            max_retries=3,
        )
        trajectory_progress = mock.Mock()

        generate_single_trajectory(
            trajectory_index=0,
            runtime_config=runtime_config,
            task_definition=PREPARE_COFFEE_TASK,
            client_factory=lambda: SequencedFakeClient(
                [
                    GenerationResult(
                        payload=make_valid_candidate(),
                        usage=GenerationUsage(
                            prompt_tokens=1000,
                            candidates_tokens=200,
                            thoughts_tokens=50,
                            total_tokens=1250,
                            traffic_type="ON_DEMAND",
                        ),
                    )
                ]
            ),
            overall_progress=mock.Mock(),
            trajectory_progress=trajectory_progress,
        )

        trajectory_progress.start.assert_called_once()
        trajectory_progress.complete.assert_called_once_with(1)
        self.assertEqual(
            trajectory_progress.set_postfix_str.call_args_list[0].args[0],
            "attempt 1/3 generating",
        )
        self.assertEqual(
            trajectory_progress.set_postfix_str.call_args_list[-1].args[0],
            "done attempts=1/3 total=$0.0013 avg=$0.0013 calls=12",
        )

    def test_generate_single_trajectory_retry_status_includes_tool_call_count(self):
        runtime_config = RuntimeConfig(
            composite_task="PrepareCoffee",
            num_runs=1,
            model="gemini-3-flash-preview",
            sdk="google-genai",
            project="demo-project",
            location="global",
            temperature=0.5,
            max_workers=1,
            max_retries=2,
        )
        trajectory_progress = mock.Mock()

        generate_single_trajectory(
            trajectory_index=0,
            runtime_config=runtime_config,
            task_definition=PREPARE_COFFEE_TASK,
            client_factory=lambda: SequencedFakeClient(
                [
                    make_invalid_candidate_missing_initial_communication(),
                    make_valid_candidate(),
                ]
            ),
            overall_progress=mock.Mock(),
            trajectory_progress=trajectory_progress,
        )

        status_updates = [
            call.args[0] for call in trajectory_progress.set_postfix_str.call_args_list
        ]
        trajectory_progress.start.assert_called_once()
        trajectory_progress.complete.assert_called_once_with(1)
        retry_status = next(
            status
            for status in status_updates
            if status.startswith("attempt 1/2 retry calls=11")
        )
        self.assertIn(
            "MissingInitialCommunicationSemanticValidationError step=1",
            retry_status,
        )
        self.assertIn(
            "Both agents must coordinate via communication before the first task action.",
            retry_status,
        )
        resumed_generation_status = next(
            status
            for status in status_updates
            if status.startswith("attempt 2/2 generating after invalid ")
        )
        self.assertIn(
            "MissingInitialCommunicationSemanticValidationError step=1",
            resumed_generation_status,
        )
        self.assertTrue(status_updates[-1].startswith("done attempts=2/2 total=$"))
        self.assertTrue(status_updates[-1].endswith(" calls=12"))

    def test_generate_single_trajectory_feeds_validation_feedback_into_retry_prompt(self):
        runtime_config = RuntimeConfig(
            composite_task="PrepareCoffee",
            num_runs=1,
            model="gemini-3-flash-preview",
            sdk="google-genai",
            project="demo-project",
            location="global",
            temperature=0.5,
            max_workers=1,
            max_retries=2,
        )
        client = PromptCapturingSequencedFakeClient(
            [
                make_invalid_candidate_missing_initial_communication(),
                make_valid_candidate(),
            ]
        )

        generate_single_trajectory(
            trajectory_index=0,
            runtime_config=runtime_config,
            task_definition=PREPARE_COFFEE_TASK,
            client_factory=lambda: client,
            overall_progress=mock.Mock(),
            trajectory_progress=mock.Mock(),
        )

        self.assertEqual(len(client.prompts), 2)
        self.assertNotIn("Previous attempt failed validation.", client.prompts[0])
        self.assertIn("Previous attempt failed validation.", client.prompts[1])
        self.assertIn(
            "MissingInitialCommunicationSemanticValidationError",
            client.prompts[1],
        )
        self.assertIn("failing_step: 1", client.prompts[1])
        self.assertIn("Local bad example:", client.prompts[1])
        self.assertIn("Regenerate the full trajectory from step 0.", client.prompts[1])

    def test_generate_single_trajectory_keeps_attempt_counts_in_done_status_when_validation_disabled(self):
        runtime_config = RuntimeConfig(
            composite_task="PrepareCoffee",
            num_runs=1,
            model="gemini-3-flash-preview",
            sdk="google-genai",
            project="demo-project",
            location="global",
            temperature=0.5,
            max_workers=1,
            max_retries=2,
            disable_validation=True,
        )
        trajectory_progress = mock.Mock()

        trajectory = generate_single_trajectory(
            trajectory_index=0,
            runtime_config=runtime_config,
            task_definition=PREPARE_COFFEE_TASK,
            client_factory=lambda: SequencedFakeClient(
                [
                    "not valid json",
                    make_invalid_candidate_missing_initial_communication(),
                ]
            ),
            overall_progress=mock.Mock(),
            trajectory_progress=trajectory_progress,
        )

        self.assertFalse(trajectory["validation"]["is_valid"])
        status_updates = [
            call.args[0] for call in trajectory_progress.set_postfix_str.call_args_list
        ]
        trajectory_progress.start.assert_called_once()
        trajectory_progress.complete.assert_called_once_with(1)
        self.assertEqual(status_updates[0], "generating")
        retry_status = next(
            status for status in status_updates if status.startswith("retrying")
        )
        self.assertIn("ResponseFormatValidationError", retry_status)
        self.assertIn("Model response did not contain JSON.", retry_status)
        self.assertIn(
            (
                "generating after invalid ResponseFormatValidationError: "
                "Model response did not contain JSON."
            ),
            status_updates,
        )
        self.assertNotIn("valid", status_updates)
        self.assertTrue(status_updates[-1].startswith("done attempts=2/2 total=$"))
        self.assertIn("calls=11", status_updates[-1])
        self.assertTrue(
            status_updates[-1].endswith(
                " invalid MissingInitialCommunicationSemanticValidationError step=1"
            )
        )
        self.assertEqual(status_updates[0], "generating")

    def test_validation_error_progress_summary_keeps_progress_message_short(self):
        validation = {
            "is_valid": False,
            "error_type": "NavigationSemanticValidationError",
            "error_base_type": "TaskSemanticValidationError",
            "error": (
                "Step 9 (pick_up_object): agent_1 is at coffee_machine_1 and must use "
                "navigate_to_fixture to reach counter_1 before using pick_up_object."
            ),
            "step": 9,
        }

        self.assertEqual(
            _validation_error_progress_summary(validation),
            "NavigationSemanticValidationError step=9",
        )

    def test_build_retry_feedback_text_includes_local_bad_example(self):
        feedback = _build_retry_feedback_text(
            NavigationSemanticValidationError(
                "agent_1 must navigate_to_fixture(counter_1) before pick_up_object.",
                step=9,
                details={"agent": "agent_1", "expected_location": "counter_1"},
            ),
            candidate=make_valid_candidate(),
        )

        self.assertIn("Previous attempt failed validation.", feedback)
        self.assertIn("error_type: NavigationSemanticValidationError", feedback)
        self.assertIn("failing_step: 9", feedback)
        self.assertIn("Local bad example:", feedback)
        self.assertIn('"step": 8', feedback)
        self.assertIn('"step": 9', feedback)
        self.assertIn("Regenerate the full trajectory from step 0.", feedback)

    def test_generate_single_trajectory_reports_total_and_average_cost_for_verbalized_sampling(self):
        runtime_config = RuntimeConfig(
            composite_task="PrepareCoffee",
            num_runs=1,
            model="gemini-3-flash-preview",
            sdk="google-genai",
            project="demo-project",
            location="global",
            temperature=0.5,
            max_workers=1,
            max_retries=3,
            sampling="verbalized",
            verbalized_k=2,
        )
        trajectory_progress = mock.Mock()

        generate_single_trajectory(
            trajectory_index=0,
            runtime_config=runtime_config,
            task_definition=PREPARE_COFFEE_TASK,
            client_factory=lambda: SequencedFakeClient(
                [
                    GenerationResult(
                        payload=make_verbalized_response(
                            make_valid_candidate(include_agents=False),
                            make_alternative_valid_candidate(),
                            probabilities=[0.6, 0.4],
                        ),
                        usage=GenerationUsage(
                            prompt_tokens=900,
                            candidates_tokens=300,
                            thoughts_tokens=0,
                            total_tokens=1200,
                            traffic_type="ON_DEMAND",
                        ),
                    )
                ]
            ),
            overall_progress=mock.Mock(),
            trajectory_progress=trajectory_progress,
        )

        trajectory_progress.complete.assert_called_once_with(2)
        self.assertEqual(
            trajectory_progress.set_postfix_str.call_args_list[-1].args[0],
            "done attempts=1/3 total=$0.0014 avg=$0.0007 success=2/2 avg_calls=12.5",
        )

    def test_generate_single_trajectory_retry_status_includes_verbalized_invalid_error(self):
        runtime_config = RuntimeConfig(
            composite_task="PrepareCoffee",
            num_runs=1,
            model="gemini-3-flash-preview",
            sdk="google-genai",
            project="demo-project",
            location="global",
            temperature=0.5,
            max_workers=1,
            max_retries=2,
            sampling="verbalized",
            verbalized_k=2,
        )
        trajectory_progress = mock.Mock()

        generate_single_trajectory(
            trajectory_index=0,
            runtime_config=runtime_config,
            task_definition=PREPARE_COFFEE_TASK,
            client_factory=lambda: SequencedFakeClient(
                [
                    GenerationResult(
                        payload=make_verbalized_response(
                            make_valid_candidate(include_agents=False),
                            make_invalid_candidate_missing_initial_communication(),
                            probabilities=[0.6, 0.4],
                        ),
                        usage=GenerationUsage(
                            prompt_tokens=900,
                            candidates_tokens=300,
                            thoughts_tokens=0,
                            total_tokens=1200,
                            traffic_type="ON_DEMAND",
                        ),
                    ),
                    GenerationResult(
                        payload=make_verbalized_response(
                            make_alternative_valid_candidate(),
                            make_invalid_candidate_missing_initial_communication(),
                            probabilities=[0.7, 0.3],
                        ),
                        usage=GenerationUsage(
                            prompt_tokens=900,
                            candidates_tokens=300,
                            thoughts_tokens=0,
                            total_tokens=1200,
                            traffic_type="ON_DEMAND",
                        ),
                    ),
                ]
            ),
            overall_progress=mock.Mock(),
            trajectory_progress=trajectory_progress,
        )

        status_updates = [
            call.args[0] for call in trajectory_progress.set_postfix_str.call_args_list
        ]
        retry_status = next(
            status
            for status in status_updates
            if status.startswith("attempt 1/2 retry calls=")
        )
        self.assertIn(
            "MissingInitialCommunicationSemanticValidationError step=1",
            retry_status,
        )
        resumed_generation_status = next(
            status
            for status in status_updates
            if status.startswith("attempt 2/2 generating after invalid ")
        )
        self.assertIn(
            "MissingInitialCommunicationSemanticValidationError step=1",
            resumed_generation_status,
        )

    def test_generate_single_trajectory_raises_specific_error_for_insufficient_verbalized_results(self):
        runtime_config = RuntimeConfig(
            composite_task="PrepareCoffee",
            num_runs=1,
            model="gemini-3-flash-preview",
            sdk="google-genai",
            project="demo-project",
            location="global",
            temperature=0.5,
            max_workers=1,
            max_retries=1,
            sampling="verbalized",
            verbalized_k=2,
        )

        with self.assertRaises(TrajectoryGenerationError) as raised:
            generate_single_trajectory(
                trajectory_index=0,
                runtime_config=runtime_config,
                task_definition=PREPARE_COFFEE_TASK,
                client_factory=lambda: SequencedFakeClient(
                    [
                        GenerationResult(
                            payload=make_verbalized_response(
                                make_valid_candidate(include_agents=False),
                                make_invalid_candidate_missing_initial_communication(),
                                probabilities=[0.6, 0.4],
                            ),
                            usage=GenerationUsage(
                                prompt_tokens=900,
                                candidates_tokens=300,
                                thoughts_tokens=0,
                                total_tokens=1200,
                                traffic_type="ON_DEMAND",
                            ),
                        )
                    ]
                ),
                overall_progress=mock.Mock(),
                trajectory_progress=mock.Mock(),
            )

        self.assertIn(
            InsufficientValidUniqueTrajectoriesInvalidError.__name__,
            str(raised.exception),
        )
        self.assertIn(
            "Verbalized run did not produce enough valid unique trajectories because some candidates failed validation.",
            str(raised.exception),
        )

    def test_generate_single_trajectory_raises_duplicate_specific_error_for_insufficient_verbalized_results(self):
        runtime_config = RuntimeConfig(
            composite_task="PrepareCoffee",
            num_runs=1,
            model="gemini-3-flash-preview",
            sdk="google-genai",
            project="demo-project",
            location="global",
            temperature=0.5,
            max_workers=1,
            max_retries=1,
            sampling="verbalized",
            verbalized_k=2,
        )
        first_candidate = make_valid_candidate(include_agents=False)
        second_candidate = make_alternative_valid_candidate()
        task_instance = make_prepare_coffee_task_instance(0)
        validator = PREPARE_COFFEE_TASK.validator_factory(task_instance)
        seen_signatures = {
            validator.validate(first_candidate)["signature"],
            validator.validate(second_candidate)["signature"],
        }

        with self.assertRaises(TrajectoryGenerationError) as raised:
            generate_single_trajectory(
                trajectory_index=0,
                runtime_config=runtime_config,
                task_definition=PREPARE_COFFEE_TASK,
                client_factory=lambda: SequencedFakeClient(
                    [
                        GenerationResult(
                            payload=make_verbalized_response(
                                first_candidate,
                                second_candidate,
                                probabilities=[0.6, 0.4],
                            ),
                            usage=GenerationUsage(
                                prompt_tokens=900,
                                candidates_tokens=300,
                                thoughts_tokens=0,
                                total_tokens=1200,
                                traffic_type="ON_DEMAND",
                            ),
                        )
                    ]
                ),
                overall_progress=mock.Mock(),
                trajectory_progress=mock.Mock(),
                seen_signatures=seen_signatures,
                seen_signatures_lock=threading.Lock(),
            )

        self.assertIn(
            InsufficientValidUniqueTrajectoriesDuplicateError.__name__,
            str(raised.exception),
        )
        self.assertIn(
            "Verbalized run did not produce enough valid unique trajectories because some candidates duplicated existing trajectories.",
            str(raised.exception),
        )

    def test_generate_single_trajectory_raises_mixed_specific_error_for_insufficient_verbalized_results(self):
        runtime_config = RuntimeConfig(
            composite_task="PrepareCoffee",
            num_runs=1,
            model="gemini-3-flash-preview",
            sdk="google-genai",
            project="demo-project",
            location="global",
            temperature=0.5,
            max_workers=1,
            max_retries=1,
            sampling="verbalized",
            verbalized_k=2,
        )
        duplicate_candidate = make_valid_candidate(include_agents=False)
        task_instance = make_prepare_coffee_task_instance(0)
        validator = PREPARE_COFFEE_TASK.validator_factory(task_instance)
        seen_signatures = {validator.validate(duplicate_candidate)["signature"]}

        with self.assertRaises(TrajectoryGenerationError) as raised:
            generate_single_trajectory(
                trajectory_index=0,
                runtime_config=runtime_config,
                task_definition=PREPARE_COFFEE_TASK,
                client_factory=lambda: SequencedFakeClient(
                    [
                        GenerationResult(
                            payload=make_verbalized_response(
                                duplicate_candidate,
                                make_invalid_candidate_missing_initial_communication(),
                                probabilities=[0.7, 0.3],
                            ),
                            usage=GenerationUsage(
                                prompt_tokens=900,
                                candidates_tokens=300,
                                thoughts_tokens=0,
                                total_tokens=1200,
                                traffic_type="ON_DEMAND",
                            ),
                        )
                    ]
                ),
                overall_progress=mock.Mock(),
                trajectory_progress=mock.Mock(),
                seen_signatures=seen_signatures,
                seen_signatures_lock=threading.Lock(),
            )

        self.assertIn(
            InsufficientValidUniqueTrajectoriesMixedError.__name__,
            str(raised.exception),
        )
        self.assertIn(
            "Verbalized run did not produce enough valid unique trajectories because some candidates failed validation and others duplicated existing trajectories.",
            str(raised.exception),
        )

    def test_generate_single_trajectory_returns_all_records_for_verbalized_sampling(self):
        runtime_config = RuntimeConfig(
            composite_task="PrepareCoffee",
            num_runs=1,
            model="gemini-3-flash-preview",
            sdk="google-genai",
            project="demo-project",
            location="global",
            temperature=0.5,
            max_workers=1,
            max_retries=1,
            sampling="verbalized",
            verbalized_k=2,
        )

        trajectories = generate_single_trajectory(
            trajectory_index=0,
            runtime_config=runtime_config,
            task_definition=PREPARE_COFFEE_TASK,
            client_factory=lambda: SequencedFakeClient(
                [
                    GenerationResult(
                        payload=make_verbalized_response(
                            make_valid_candidate(include_agents=False),
                            make_alternative_valid_candidate(),
                            probabilities=[0.6, 0.4],
                        ),
                        usage=GenerationUsage(
                            prompt_tokens=900,
                            candidates_tokens=300,
                            thoughts_tokens=0,
                            total_tokens=1200,
                            traffic_type="ON_DEMAND",
                        ),
                    )
                ]
            ),
            overall_progress=mock.Mock(),
            trajectory_progress=mock.Mock(),
        )

        self.assertIsInstance(trajectories, list)
        self.assertEqual(
            [trajectory["trajectory_id"] for trajectory in trajectories],
            ["traj_000000", "traj_000001"],
        )
        self.assertEqual(
            [trajectory["sampling_metadata"]["probability"] for trajectory in trajectories],
            [0.6, 0.4],
        )

    def test_generate_single_trajectory_reports_success_fraction_for_mixed_verbalized_run(self):
        runtime_config = RuntimeConfig(
            composite_task="PrepareCoffee",
            num_runs=1,
            model="gemini-3-flash-preview",
            sdk="google-genai",
            project="demo-project",
            location="global",
            temperature=0.5,
            max_workers=1,
            max_retries=1,
            sampling="verbalized",
            verbalized_k=2,
            disable_validation=True,
        )
        trajectory_progress = mock.Mock()

        generate_single_trajectory(
            trajectory_index=0,
            runtime_config=runtime_config,
            task_definition=PREPARE_COFFEE_TASK,
            client_factory=lambda: SequencedFakeClient(
                [
                    GenerationResult(
                        payload=make_verbalized_response(
                            make_valid_candidate(include_agents=False),
                            make_invalid_candidate_missing_initial_communication(),
                            probabilities=[0.6, 0.4],
                        ),
                        usage=GenerationUsage(
                            prompt_tokens=900,
                            candidates_tokens=300,
                            thoughts_tokens=0,
                            total_tokens=1200,
                            traffic_type="ON_DEMAND",
                        ),
                    )
                ]
            ),
            overall_progress=mock.Mock(),
            trajectory_progress=trajectory_progress,
        )

        self.assertIn(
            "success=1/2",
            trajectory_progress.set_postfix_str.call_args_list[-1].args[0],
        )
        self.assertIn(
            "invalid MissingInitialCommunicationSemanticValidationError step=1",
            trajectory_progress.set_postfix_str.call_args_list[-1].args[0],
        )

    def test_rich_progress_display_uses_non_expanding_layout(self):
        runtime_config = RuntimeConfig(
            composite_task="PrepareCoffee",
            num_runs=2,
            model="gemini-3-flash-preview",
            sdk="google-genai",
            project="demo-project",
            location="global",
            temperature=0.5,
            max_workers=2,
            max_retries=3,
        )
        fake_progress = mock.Mock()
        fake_progress.add_task.side_effect = [101, 102, 103]

        with mock.patch(
            "data_generation.task_level.generation.progress.Console",
            return_value=mock.sentinel.console,
        ) as console_cls:
            with mock.patch(
                "data_generation.task_level.generation.progress.TextColumn",
                side_effect=[
                    mock.sentinel.description_column,
                    mock.sentinel.status_column,
                ],
            ):
                with mock.patch(
                    "data_generation.task_level.generation.progress.BarColumn",
                    return_value=mock.sentinel.bar_column,
                ):
                    with mock.patch(
                        "data_generation.task_level.generation.progress.TaskProgressColumn",
                        return_value=mock.sentinel.task_progress_column,
                    ):
                        with mock.patch(
                            "data_generation.task_level.generation.progress.MofNCompleteColumn",
                            return_value=mock.sentinel.mofn_column,
                        ):
                            with mock.patch(
                                "data_generation.task_level.generation.progress.StaticQueuedTimeElapsedColumn",
                                return_value=mock.sentinel.elapsed_column,
                            ):
                                with mock.patch(
                                    "data_generation.task_level.generation.progress.RichProgress",
                                    return_value=fake_progress,
                                ) as rich_progress:
                                    display = RichProgressDisplay(runtime_config)

        console_cls.assert_called_once_with(stderr=True)
        self.assertEqual(rich_progress.call_args.kwargs["console"], mock.sentinel.console)
        self.assertFalse(rich_progress.call_args.kwargs["expand"])
        fake_progress.start.assert_called_once()
        self.assertEqual(fake_progress.add_task.call_count, 3)
        self.assertIn("run 000000", fake_progress.add_task.call_args_list[1].args[0])
        self.assertIn("run 000001", fake_progress.add_task.call_args_list[2].args[0])
        self.assertEqual(fake_progress.add_task.call_args_list[0].kwargs.get("start"), None)
        self.assertEqual(fake_progress.add_task.call_args_list[1].kwargs["start"], True)
        self.assertEqual(fake_progress.add_task.call_args_list[2].kwargs["start"], True)
        display.close()
        fake_progress.stop.assert_called_once()

    def test_rich_progress_display_constructs_static_queued_elapsed_column(self):
        """Builds the real Rich progress display to catch column init regressions."""

        from rich.console import Console as RichConsole

        runtime_config = RuntimeConfig(
            composite_task="PrepareCoffee",
            num_runs=2,
            model="gemini-3-flash-preview",
            sdk="google-genai",
            project="demo-project",
            location="global",
            temperature=0.5,
            max_workers=1,
            max_retries=3,
            sampling="verbalized",
            verbalized_k=2,
        )
        buffer = StringIO()

        with mock.patch(
            "data_generation.task_level.generation.progress.Console",
            return_value=RichConsole(file=buffer, force_terminal=False, width=120),
        ):
            display = RichProgressDisplay(runtime_config)

        queued_task = display._progress.tasks[1]
        elapsed_column = display._progress.columns[4]
        self.assertEqual(elapsed_column.render(queued_task).plain, "0:00:00")
        display.close()

    def test_rich_task_progress_adapter_preserves_status_when_starting(self):
        """Keeps the Rich status field populated after a queued task starts."""

        progress = mock.Mock()
        adapter = RichTaskProgressAdapter(
            progress,
            task_id=7,
            total=2,
            started=False,
            status="queued",
        )

        adapter.start()

        progress.reset.assert_called_once_with(
            7,
            start=True,
            total=2,
            completed=0,
            queued=False,
            status="queued",
        )

    def test_rich_progress_display_preserves_status_when_queued_task_starts(self):
        """Exercises the real Rich display to guard against status-field regressions."""

        from rich.console import Console as RichConsole

        runtime_config = RuntimeConfig(
            composite_task="PrepareCoffee",
            num_runs=1,
            model="gemini-3-flash-preview",
            sdk="google-genai",
            project="demo-project",
            location="global",
            temperature=0.5,
            max_workers=1,
            max_retries=3,
        )
        buffer = StringIO()

        with mock.patch(
            "data_generation.task_level.generation.progress.Console",
            return_value=RichConsole(file=buffer, force_terminal=False, width=120),
        ):
            display = RichProgressDisplay(runtime_config)

        display.trajectory_progress_bars[0].start()

        started_task = display._progress.tasks[1]
        self.assertEqual(started_task.fields["status"], "queued")
        display.close()

    def test_batch_rich_progress_display_uses_non_expanding_layout(self):
        runtime_config = RuntimeConfig(
            composite_task="PrepareCoffee",
            num_runs=2,
            model="gemini-3-flash-preview",
            sdk="google-genai",
            project="demo-project",
            location="global",
            temperature=0.5,
            max_workers=2,
            max_retries=3,
            batch_processing=True,
            batch_gcs_prefix="gs://demo-bucket/batch-prefix",
        )
        fake_progress = mock.Mock()
        fake_progress.add_task.return_value = 201

        with mock.patch(
            "data_generation.task_level.generation.progress.Console",
            return_value=mock.sentinel.console,
        ) as console_cls:
            with mock.patch(
                "data_generation.task_level.generation.progress.TextColumn",
                side_effect=[
                    mock.sentinel.description_column,
                    mock.sentinel.status_column,
                ],
            ):
                with mock.patch(
                    "data_generation.task_level.generation.progress.BarColumn",
                    return_value=mock.sentinel.bar_column,
                ):
                    with mock.patch(
                        "data_generation.task_level.generation.progress.TaskProgressColumn",
                        return_value=mock.sentinel.task_progress_column,
                    ):
                        with mock.patch(
                            "data_generation.task_level.generation.progress.MofNCompleteColumn",
                            return_value=mock.sentinel.mofn_column,
                        ):
                            with mock.patch(
                                "data_generation.task_level.generation.progress.StaticQueuedTimeElapsedColumn",
                                return_value=mock.sentinel.elapsed_column,
                            ):
                                with mock.patch(
                                    "data_generation.task_level.generation.progress.RichProgress",
                                    return_value=fake_progress,
                                ) as rich_progress:
                                    display = RichBatchProgressDisplay(runtime_config)

        console_cls.assert_called_once_with(stderr=True)
        self.assertEqual(rich_progress.call_args.kwargs["console"], mock.sentinel.console)
        self.assertFalse(rich_progress.call_args.kwargs["expand"])
        fake_progress.start.assert_called_once()
        fake_progress.add_task.assert_called_once_with(
            "[cyan]runs[/cyan]",
            total=runtime_config.num_runs,
            status="waiting for batch results",
        )
        display.close()
        fake_progress.stop.assert_called_once()

    def test_create_batch_progress_handles_prefers_rich_display(self):
        runtime_config = RuntimeConfig(
            composite_task="PrepareCoffee",
            num_runs=2,
            model="gemini-3-flash-preview",
            sdk="google-genai",
            project="demo-project",
            location="global",
            temperature=0.5,
            max_workers=2,
            max_retries=3,
            batch_processing=True,
            batch_gcs_prefix="gs://demo-bucket/batch-prefix",
        )
        fake_display = mock.Mock()
        fake_display.overall_progress = mock.sentinel.overall_progress
        fake_display.console.print = mock.sentinel.log_writer

        with mock.patch(
            "data_generation.task_level.runtime.batch_generation.RichBatchProgressDisplay",
            return_value=fake_display,
        ):
            handles = _create_batch_progress_handles(
                runtime_config,
                disable_progress=False,
            )

        self.assertEqual(handles.display, fake_display)
        self.assertEqual(handles.overall_progress, mock.sentinel.overall_progress)
        self.assertEqual(handles.trajectory_progress_bars, [])
        self.assertEqual(handles.log_writer, mock.sentinel.log_writer)

    def test_cost_summary_falls_back_to_heuristic_without_usage_metadata(self):
        runtime_config = RuntimeConfig(
            composite_task="PrepareCoffee",
            num_runs=1,
            model="gemini-3-flash-preview",
            sdk="google-genai",
            project="demo-project",
            location="global",
            temperature=0.5,
            max_workers=1,
            max_retries=2,
        )
        payload = generate_trajectories(
            runtime_config,
            client_factory=lambda: SequencedFakeClient([make_valid_candidate()]),
            show_progress=False,
        )

        self.assertEqual(
            payload["trajectories"][0]["generation_usage"]["usage_source"],
            "heuristic_4_chars_per_token",
        )
        self.assertEqual(
            payload["trajectories"][0]["generation_usage"]["reasoning_tokens"],
            0,
        )
        self.assertNotIn("usage_sources", payload["cost_summary"])
        self.assertEqual(payload["cost_summary"]["reasoning_tokens"], 0)
        self.assertNotIn("pricing_supported", payload["cost_summary"])
        self.assertNotIn("pricing_reference", payload["cost_summary"])
        self.assertNotIn("currency", payload["cost_summary"])
        self.assertEqual(
            payload["cost_summary"]["pricing"],
            {
                "model": "gemini-3-flash-preview",
                "input_usd_per_million_tokens": 0.5,
                "output_usd_per_million_tokens": 3.0,
            },
        )
        self.assertNotIn("cost_estimate", payload)

    def test_post_run_cost_summary_includes_average_trajectory_cost(self):
        runtime_config = RuntimeConfig(
            composite_task="PrepareCoffee",
            num_runs=2,
            model="gemini-3-flash-preview",
            sdk="google-genai",
            project="demo-project",
            location="global",
            temperature=0.5,
            max_workers=1,
            max_retries=2,
        )
        payload = generate_trajectories(
            runtime_config,
            client_factory=lambda: SequencedFakeClient(
                [
                    GenerationResult(
                        payload=make_valid_candidate(),
                        usage=GenerationUsage(
                            prompt_tokens=1000,
                            candidates_tokens=200,
                            thoughts_tokens=50,
                            total_tokens=1250,
                            traffic_type="ON_DEMAND",
                        ),
                    ),
                    GenerationResult(
                        payload=make_alternative_valid_candidate(),
                        usage=GenerationUsage(
                            prompt_tokens=2000,
                            candidates_tokens=100,
                            thoughts_tokens=0,
                            total_tokens=2100,
                            traffic_type="ON_DEMAND",
                        ),
                    ),
                ]
            ),
            show_progress=False,
        )

        self.assertAlmostEqual(
            payload["cost_summary"]["total_cost_usd"],
            0.0025,
        )
        self.assertAlmostEqual(
            payload["cost_summary"]["average_trajectory_cost_usd"],
            0.0013,
        )

    def test_preflight_cost_estimate_uses_manual_task_token_estimate(self):
        runtime_config = RuntimeConfig(
            composite_task="PrepareCoffee",
            num_runs=2,
            model="gemini-3-flash-preview",
            sdk="google-genai",
            project="demo-project",
            location="global",
            temperature=0.5,
            max_workers=1,
            max_retries=2,
        )

        summary = _build_preflight_cost_estimate_summary(
            runtime_config,
            PREPARE_COFFEE_TASK,
        )

        self.assertEqual(
            summary["best_case_tokens"],
            {"prompt": 12440, "output": 14400, "reasoning": 0, "total": 26840},
        )
        self.assertAlmostEqual(summary["best_case_total_usd"], 0.0494)
        self.assertAlmostEqual(summary["worst_case_total_usd"], 0.0494)
        self.assertEqual(
            summary["pricing"],
            {
                "model": "gemini-3-flash-preview",
                "input_usd_per_million_tokens": 0.5,
                "output_usd_per_million_tokens": 3.0,
            },
        )
        self.assertIn("manual task token estimate", summary["notes"][0])
        self.assertIn("manual task token estimate", summary["notes"][2])

    def test_post_run_cost_summary_counts_retry_attempts(self):
        runtime_config = RuntimeConfig(
            composite_task="PrepareCoffee",
            num_runs=1,
            model="gemini-3-flash-preview",
            sdk="google-genai",
            project="demo-project",
            location="global",
            temperature=0.5,
            max_workers=1,
            max_retries=2,
        )
        payload = generate_trajectories(
            runtime_config,
            client_factory=lambda: SequencedFakeClient(
                [
                    GenerationResult(
                        payload=make_invalid_candidate_missing_initial_communication(),
                        usage=GenerationUsage(
                            prompt_tokens=1000,
                            candidates_tokens=200,
                            thoughts_tokens=50,
                            total_tokens=1250,
                            traffic_type="ON_DEMAND",
                        ),
                    ),
                    GenerationResult(
                        payload=make_valid_candidate(),
                        usage=GenerationUsage(
                            prompt_tokens=1000,
                            candidates_tokens=200,
                            thoughts_tokens=50,
                            total_tokens=1250,
                            traffic_type="ON_DEMAND",
                        ),
                    ),
                ]
            ),
            show_progress=False,
        )

        self.assertEqual(
            payload["trajectories"][0]["generation_usage"]["successful_attempt_number"],
            2,
        )
        self.assertEqual(
            payload["cost_summary"]["total_tokens"],
            2500,
        )
        self.assertAlmostEqual(
            payload["cost_summary"]["total_cost_usd"],
            0.0025,
        )
        self.assertEqual(
            payload["cost_summary"]["notes"][1],
            "Retry-inclusive totals assume earlier failed attempts used the same token profile as the successful attempt.",
        )
        self.assertNotIn("cost_estimate", payload)

    def test_cost_summary_scales_totals_by_saved_attempt_counts(self):
        summary = _build_cost_summary_from_generation_usages(
            [
                {
                    "successful_attempt_number": 1,
                    "prompt_tokens": 100,
                    "output_tokens": 20,
                    "reasoning_tokens": 5,
                    "total_tokens": 125,
                    "pricing": {
                        "model": "gemini-3-flash-preview",
                        "input_usd_per_million_tokens": 0.5,
                        "output_usd_per_million_tokens": 3.0,
                    },
                },
                {
                    "successful_attempt_number": 3,
                    "prompt_tokens": 200,
                    "output_tokens": 40,
                    "reasoning_tokens": 10,
                    "total_tokens": 250,
                    "pricing": {
                        "model": "gemini-3-flash-preview",
                        "input_usd_per_million_tokens": 0.5,
                        "output_usd_per_million_tokens": 3.0,
                    },
                },
            ]
        )

        self.assertEqual(summary["prompt_tokens"], 700)
        self.assertEqual(summary["output_tokens"], 140)
        self.assertEqual(summary["reasoning_tokens"], 35)
        self.assertEqual(summary["total_tokens"], 875)
        self.assertAlmostEqual(summary["input_cost_usd"], 0.0003)
        self.assertAlmostEqual(summary["output_cost_usd"], 0.0005)
        self.assertAlmostEqual(summary["total_cost_usd"], 0.0009)
        self.assertAlmostEqual(summary["average_trajectory_cost_usd"], 0.0004)

    def test_disable_validation_keeps_invalid_trajectory_and_records_error(self):
        runtime_config = RuntimeConfig(
            composite_task="PrepareCoffee",
            num_runs=1,
            model="gemini-3-flash-preview",
            sdk="google-genai",
            project="demo-project",
            location="global",
            temperature=0.5,
            max_workers=1,
            max_retries=1,
            disable_validation=True,
        )
        invalid_candidate = make_invalid_candidate_missing_initial_communication()
        invalid_candidate.pop("agents")
        payload = generate_trajectories(
            runtime_config,
            client_factory=lambda: SequencedFakeClient([invalid_candidate]),
            show_progress=False,
        )

        trajectory = payload["trajectories"][0]
        self.assertEqual(
            trajectory["agents"],
            [{"agent": "agent_0"}, {"agent": "agent_1"}],
        )
        self.assertFalse(trajectory["validation"]["is_valid"])
        self.assertTrue(trajectory["validation"]["validation_disabled"])
        self.assertEqual(
            trajectory["validation"]["error_type"],
            "MissingInitialCommunicationSemanticValidationError",
        )
        self.assertEqual(
            trajectory["validation"]["error_base_type"],
            "TaskSemanticValidationError",
        )
        self.assertIn(
            "Both agents must coordinate via communication before the first task action.",
            trajectory["validation"]["error"],
        )
        self.assertEqual(
            trajectory["validation"]["error_details"]["agent"],
            "agent_0",
        )
        self.assertEqual(trajectory["validation"]["step"], 1)
        self.assertEqual(len(trajectory["steps"]), 11)
        self.assertNotIn(
            "successful_attempt_number",
            trajectory["generation_usage"],
        )
        self.assertIn("observed_cost_usd", trajectory["generation_usage"])
        self.assertEqual(
            payload["cost_summary"]["total_cost_usd"],
            trajectory["generation_usage"]["observed_cost_usd"],
        )
        for field in (
            "prompt_tokens",
            "output_tokens",
            "reasoning_tokens",
            "total_tokens",
            "input_cost_usd",
            "output_cost_usd",
            "total_cost_usd",
        ):
            self.assertIn(field, payload["cost_summary"])

    def test_disable_validation_skips_duplicate_rejection(self):
        runtime_config = RuntimeConfig(
            composite_task="PrepareCoffee",
            num_runs=2,
            model="gemini-3-flash-preview",
            sdk="google-genai",
            project="demo-project",
            location="global",
            temperature=0.5,
            max_workers=1,
            max_retries=1,
            disable_validation=True,
        )
        payload = generate_trajectories(
            runtime_config,
            client_factory=lambda: SequencedFakeClient(
                [
                    make_invalid_candidate_missing_initial_communication(),
                    make_invalid_candidate_missing_initial_communication(),
                ]
            ),
            show_progress=False,
        )

        self.assertEqual(len(payload["trajectories"]), 2)
        self.assertEqual(
            payload["trajectories"][0]["validation"]["error"],
            payload["trajectories"][1]["validation"]["error"],
        )

    def test_disable_validation_logs_single_projected_cost_without_attempt_details(self):
        runtime_config = RuntimeConfig(
            composite_task="PrepareCoffee",
            num_runs=1,
            model="gemini-3-flash-preview",
            sdk="google-genai",
            project="demo-project",
            location="global",
            temperature=0.5,
            max_workers=1,
            max_retries=2,
            disable_validation=True,
        )

        with mock.patch(
            "data_generation.task_level.generation.progress._log_runtime_message"
        ) as log_runtime_message:
            generate_trajectories(
                runtime_config,
                client_factory=lambda: SequencedFakeClient(
                    [make_invalid_candidate_missing_initial_communication()]
                ),
                show_progress=True,
            )

        logged_messages = [call.args[0] for call in log_runtime_message.call_args_list]
        self.assertEqual(len(logged_messages), 1)
        self.assertTrue(logged_messages[0].startswith("Initial projected cost: $"))
        self.assertNotIn("best case", logged_messages[0])
        self.assertNotIn("worst case", logged_messages[0])
        self.assertNotIn("attempt ", logged_messages[0])

    def test_disable_validation_output_payloads_omit_attempt_number(self):
        runtime_config = RuntimeConfig(
            composite_task="PrepareCoffee",
            num_runs=1,
            model="gemini-3-flash-preview",
            sdk="google-genai",
            project="demo-project",
            location="global",
            temperature=0.5,
            max_workers=1,
            max_retries=1,
            disable_validation=True,
        )
        payload = generate_trajectories(
            runtime_config,
            client_factory=lambda: SequencedFakeClient(
                [make_invalid_candidate_missing_initial_communication()]
            ),
            show_progress=False,
        )
        summary_path = Path("/tmp/summary.json")

        summary_payload = build_summary_output_payload(payload)
        cost_payload = build_cost_output_payload(
            payload,
            trajectory_output_path=summary_path,
        )

        self.assertEqual(summary_payload["cost_summary"], payload["cost_summary"])
        self.assertNotIn("cost_estimate", summary_payload)
        self.assertNotIn("project", summary_payload)
        self.assertNotIn("location", summary_payload)
        self.assertNotIn("usage_sources", summary_payload["cost_summary"])
        self.assertNotIn(
            "all_trajectories_used_api_usage_metadata",
            summary_payload["cost_summary"],
        )
        self.assertNotIn("pricing_supported", summary_payload["cost_summary"])
        self.assertNotIn("pricing_reference", summary_payload["cost_summary"])
        self.assertNotIn("currency", summary_payload["cost_summary"])
        self.assertNotIn(
            "successful_attempt_number",
            cost_payload["trajectory_costs"][0]["generation_usage"],
        )
        self.assertEqual(
            payload["trajectories"][0]["validation"]["error_type"],
            "MissingInitialCommunicationSemanticValidationError",
        )
        self.assertEqual(
            payload["trajectories"][0]["validation"]["error_base_type"],
            "TaskSemanticValidationError",
        )
        self.assertEqual(
            cost_payload["trajectory_costs"][0]["generation_usage"]["observed_cost_usd"],
            payload["trajectories"][0]["generation_usage"]["observed_cost_usd"],
        )

    def test_error_summary_output_payload_aggregates_retry_and_saved_errors(self):
        runtime_config = RuntimeConfig(
            composite_task="PrepareCoffee",
            num_runs=1,
            model="gemini-3-flash-preview",
            sdk="google-genai",
            project="demo-project",
            location="global",
            temperature=0.5,
            max_workers=1,
            max_retries=2,
            disable_validation=True,
        )
        invalid_candidate = make_invalid_candidate_missing_initial_communication()
        invalid_candidate.pop("agents")
        payload = generate_trajectories(
            runtime_config,
            client_factory=lambda: SequencedFakeClient(
                [
                    "not valid json",
                    invalid_candidate,
                ]
            ),
            show_progress=False,
        )

        error_payload = build_error_summary_output_payload(
            payload,
            trajectory_output_path=Path("/tmp/summary.json"),
        )

        self.assertEqual(error_payload["trajectory_output_path"], "/tmp/summary.json")
        self.assertEqual(error_payload["trajectory_directory"], "/tmp/trajectories")
        self.assertEqual(error_payload["total_errors"], 2)
        self.assertEqual(
            error_payload["possible_errors"],
            [
                "MissingInitialCommunicationSemanticValidationError",
                "ResponseFormatValidationError",
            ],
        )
        self.assertEqual(
            error_payload["error_counts_by_type"],
            [
                {
                    "error_type": "MissingInitialCommunicationSemanticValidationError",
                    "count": 1,
                },
                {"error_type": "ResponseFormatValidationError", "count": 1},
            ],
        )
        self.assertEqual(
            error_payload["distinct_errors"],
            [
                {
                    "error_type": "MissingInitialCommunicationSemanticValidationError",
                    "message": (
                        "Both agents must coordinate via communication before "
                        "the first task action."
                    ),
                    "summary": (
                        "MissingInitialCommunicationSemanticValidationError: "
                        "Both agents must coordinate via communication before "
                        "the first task action."
                    ),
                    "count": 1,
                },
                {
                    "error_type": "ResponseFormatValidationError",
                    "message": "Model response did not contain JSON.",
                    "summary": (
                        "ResponseFormatValidationError: "
                        "Model response did not contain JSON."
                    ),
                    "count": 1,
                },
            ],
        )
        self.assertEqual(
            [event["error_type"] for event in error_payload["error_events"]],
            [
                "ResponseFormatValidationError",
                "MissingInitialCommunicationSemanticValidationError",
            ],
        )
        self.assertEqual(
            error_payload["error_events"][1]["error_base_type"],
            "TaskSemanticValidationError",
        )
        self.assertFalse(error_payload["error_events"][0]["saved_in_output"])
        self.assertTrue(error_payload["error_events"][1]["saved_in_output"])

    def test_resolve_cost_output_path_defaults_to_cost_summary_filename(self):
        self.assertEqual(
            resolve_cost_output_path(Path("/tmp/summary.json")),
            Path("/tmp/cost_summary.json"),
        )

    def test_build_cost_output_payload_extracts_only_cost_data(self):
        runtime_config = RuntimeConfig(
            composite_task="PrepareCoffee",
            num_runs=1,
            model="gemini-3-flash-preview",
            sdk="google-genai",
            project="demo-project",
            location="global",
            temperature=0.5,
            max_workers=1,
            max_retries=2,
        )
        payload = generate_trajectories(
            runtime_config,
            client_factory=lambda: SequencedFakeClient([make_valid_candidate()]),
            show_progress=False,
        )
        summary_path = Path("/tmp/summary.json")

        cost_payload = build_cost_output_payload(
            payload,
            trajectory_output_path=summary_path,
        )

        self.assertEqual(
            cost_payload["trajectory_output_path"],
            str(summary_path),
        )
        self.assertEqual(
            cost_payload["trajectory_directory"],
            str(summary_path.parent / "trajectories"),
        )
        self.assertEqual(len(cost_payload["trajectory_costs"]), 1)
        self.assertEqual(
            cost_payload["trajectory_costs"][0]["trajectory_id"],
            payload["trajectories"][0]["trajectory_id"],
        )
        self.assertEqual(
            cost_payload["trajectory_costs"][0]["generation_usage"],
            payload["trajectories"][0]["generation_usage"],
        )
        self.assertEqual(cost_payload["cost_summary"], payload["cost_summary"])
        self.assertEqual(cost_payload["model_config"], payload["model_config"])
        self.assertNotIn("cost_estimate", cost_payload)
        self.assertNotIn("project", cost_payload)
        self.assertNotIn("location", cost_payload)
        self.assertNotIn("usage_sources", cost_payload["cost_summary"])
        self.assertNotIn(
            "all_trajectories_used_api_usage_metadata",
            cost_payload["cost_summary"],
        )
        self.assertNotIn("pricing_supported", cost_payload["cost_summary"])
        self.assertNotIn("pricing_reference", cost_payload["cost_summary"])
        self.assertNotIn("currency", cost_payload["cost_summary"])

    def test_build_summary_output_payload_extracts_summary_and_manifest(self):
        runtime_config = RuntimeConfig(
            composite_task="PrepareCoffee",
            num_runs=1,
            model="gemini-3-flash-preview",
            sdk="google-genai",
            project="demo-project",
            location="global",
            temperature=0.5,
            max_workers=1,
            max_retries=2,
        )
        payload = generate_trajectories(
            runtime_config,
            client_factory=lambda: SequencedFakeClient([make_valid_candidate()]),
            show_progress=False,
        )

        summary_payload = build_summary_output_payload(payload)

        self.assertEqual(summary_payload["composite_task"], payload["composite_task"])
        self.assertEqual(summary_payload["cost_summary"], payload["cost_summary"])
        self.assertEqual(summary_payload["model_config"], payload["model_config"])
        self.assertNotIn("cost_estimate", summary_payload)
        self.assertNotIn("project", summary_payload)
        self.assertNotIn("location", summary_payload)
        self.assertNotIn("usage_sources", summary_payload["cost_summary"])
        self.assertNotIn(
            "all_trajectories_used_api_usage_metadata",
            summary_payload["cost_summary"],
        )
        self.assertNotIn("pricing_supported", summary_payload["cost_summary"])
        self.assertNotIn("pricing_reference", summary_payload["cost_summary"])
        self.assertNotIn("currency", summary_payload["cost_summary"])
        self.assertEqual(summary_payload["completed_trajectories"], 1)
        self.assertEqual(summary_payload["invalid_trajectories"], 0)
        self.assertEqual(summary_payload["successful_trajectory_fraction"], 1.0)
        self.assertEqual(summary_payload["trajectory_directory"], "trajectories")
        self.assertEqual(
            summary_payload["trajectory_files"],
            [
                {
                    "trajectory_id": payload["trajectories"][0]["trajectory_id"],
                    "path": "trajectories/traj_000000.json",
                }
            ],
        )
        self.assertNotIn("trajectories", summary_payload)

    def test_summary_output_payload_counts_invalid_and_successful_trajectories(self):
        invalid_candidate = make_invalid_candidate_missing_initial_communication()
        invalid_candidate.pop("agents")
        client = SequencedFakeClient(
            [
                make_valid_candidate(),
                invalid_candidate,
            ]
        )
        runtime_config = RuntimeConfig(
            composite_task="PrepareCoffee",
            num_runs=2,
            model="gemini-3-flash-preview",
            sdk="google-genai",
            project="demo-project",
            location="global",
            temperature=0.5,
            max_workers=1,
            max_retries=1,
            disable_validation=True,
        )
        payload = generate_trajectories(
            runtime_config,
            client_factory=lambda: client,
            show_progress=False,
        )

        summary_payload = build_summary_output_payload(payload)

        self.assertEqual(summary_payload["completed_trajectories"], 2)
        self.assertEqual(summary_payload["invalid_trajectories"], 1)
        self.assertEqual(summary_payload["successful_trajectory_fraction"], 0.5)

    def test_main_writes_costs_to_separate_file(self):
        fixed_payload = {
            "composite_task": "PrepareCoffee",
            "sdk": "google-genai",
            "model": "gemini-3-flash-preview",
            "model_config": {
                "reasoning": {"thinking_level": "minimal"},
                "sampling": {"temperature": 0.2},
            },
            "project": "demo-project",
            "location": "global",
            "num_trajectories": 1,
            "generated_at": "2026-03-10T00:00:00+00:00",
            "cost_summary": {
                "prompt_tokens": 1000,
                "output_tokens": 250,
                "total_tokens": 1250,
                "total_cost_usd": 0.001,
            },
            "trajectory_prompts": [
                {
                    "trajectory_id": "traj_000000",
                    "prompt": "Prompt for traj_000000",
                }
            ],
            "attempt_prompts": [
                {
                    "run_id": "traj_000000",
                    "attempt_number": 1,
                    "prompt": "Prompt for traj_000000 attempt 1",
                }
            ],
            "trajectory_outputs": [
                {
                    "trajectory_id": "traj_000000",
                    "raw_output": '{"steps":[{"reasoning":"raw model reasoning"}]}',
                }
            ],
            "trajectories": [
                {
                    "trajectory_id": "traj_000000",
                    "generation_usage": {
                        "successful_attempt_number": 1,
                        "observed_cost_usd": 0.001,
                    },
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.json"
            cost_output_path = Path(tmpdir) / "costs.json"
            error_output_path = Path(tmpdir) / "summary_errors.json"

            with mock.patch(
                "data_generation.task_level.generation.cli.generate_trajectories",
                return_value=fixed_payload,
            ):
                with mock.patch(
                    "data_generation.task_level.generation.cli.resolve_dataset_output_path",
                    return_value=summary_path,
                ):
                    with mock.patch("builtins.print") as mocked_print:
                        exit_code = main(
                            [
                                "--cost-output",
                                str(cost_output_path),
                            ]
                        )

            self.assertEqual(exit_code, 0)
            self.assertTrue(summary_path.exists())
            self.assertTrue(cost_output_path.exists())
            self.assertTrue(error_output_path.exists())

            summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
            cost_payload = json.loads(cost_output_path.read_text(encoding="utf-8"))
            error_payload = json.loads(error_output_path.read_text(encoding="utf-8"))
            trajectory_output_dir = summary_path.parent / "trajectories"
            prompt_output_dir = summary_path.parent / "prompts"
            raw_output_dir = summary_path.parent / "outputs"
            trajectory_path = trajectory_output_dir / "traj_000000.json"
            prompt_path = prompt_output_dir / "traj_000000.md"
            attempt_prompt_path = prompt_output_dir / "traj_000000_1.md"
            raw_output_path = raw_output_dir / "traj_000000.txt"

            self.assertTrue(trajectory_path.exists())
            self.assertTrue(prompt_path.exists())
            self.assertFalse(attempt_prompt_path.exists())
            self.assertTrue(raw_output_path.exists())
            self.assertEqual(
                json.loads(trajectory_path.read_text(encoding="utf-8")),
                fixed_payload["trajectories"][0],
            )
            self.assertEqual(
                prompt_path.read_text(encoding="utf-8"),
                fixed_payload["trajectory_prompts"][0]["prompt"],
            )
            self.assertEqual(
                raw_output_path.read_text(encoding="utf-8"),
                fixed_payload["trajectory_outputs"][0]["raw_output"],
            )
            self.assertEqual(
                summary_payload,
                {
                    "composite_task": "PrepareCoffee",
                    "sdk": "google-genai",
                    "model": "gemini-3-flash-preview",
                    "num_runs": 1,
                    "model_config": {
                        "reasoning": {"thinking_level": "minimal"},
                        "sampling": {"temperature": 0.2},
                    },
                    "num_trajectories": 1,
                    "generated_at": "2026-03-10T00:00:00+00:00",
                    "cost_summary": {
                        "prompt_tokens": 1000,
                        "output_tokens": 250,
                        "total_tokens": 1250,
                        "total_cost_usd": 0.001,
                    },
                    "completed_trajectories": 1,
                    "invalid_trajectories": 0,
                    "successful_trajectory_fraction": 1.0,
                    "trajectory_directory": "trajectories",
                    "trajectory_files": [
                        {
                            "trajectory_id": "traj_000000",
                            "path": "trajectories/traj_000000.json",
                        }
                    ],
                },
            )
            self.assertNotIn("project", summary_payload)
            self.assertNotIn("location", summary_payload)
            self.assertEqual(
                cost_payload["trajectory_output_path"],
                str(summary_path),
            )
            self.assertEqual(
                cost_payload["trajectory_directory"],
                str(trajectory_output_dir),
            )
            self.assertNotIn("project", cost_payload)
            self.assertNotIn("location", cost_payload)
            self.assertEqual(
                cost_payload["cost_summary"],
                fixed_payload["cost_summary"],
            )
            self.assertEqual(
                cost_payload["model_config"],
                fixed_payload["model_config"],
            )
            self.assertNotIn("cost_estimate", cost_payload)
            self.assertEqual(
                cost_payload["trajectory_costs"],
                [
                    {
                        "trajectory_id": "traj_000000",
                        "generation_usage": {
                            "successful_attempt_number": 1,
                            "observed_cost_usd": 0.001,
                        },
                    }
                ],
            )
            self.assertEqual(
                error_payload,
                {
                    "composite_task": "PrepareCoffee",
                    "sdk": "google-genai",
                    "model": "gemini-3-flash-preview",
                    "num_runs": 1,
                    "model_config": {
                        "reasoning": {"thinking_level": "minimal"},
                        "sampling": {"temperature": 0.2},
                    },
                    "num_trajectories": 1,
                    "generated_at": "2026-03-10T00:00:00+00:00",
                    "trajectory_output_path": str(summary_path),
                    "trajectory_directory": str(trajectory_output_dir),
                    "total_errors": 0,
                    "possible_errors": [],
                    "error_counts_by_type": [],
                    "distinct_errors": [],
                    "error_events": [],
                },
            )
            printed_messages = [call.args[0] for call in mocked_print.call_args_list]
            self.assertTrue(
                any(
                    message.startswith("Wrote trajectory summary to ")
                    for message in printed_messages
                )
            )
            self.assertTrue(
                any(
                    message.startswith("Wrote 1 trajectory files to ")
                    for message in printed_messages
                )
            )
            self.assertTrue(
                any(
                    message.startswith("Wrote 1 prompt files to ")
                    for message in printed_messages
                )
            )
            self.assertTrue(
                any(
                    message.startswith("Wrote 1 raw output files to ")
                    for message in printed_messages
                )
            )
            self.assertTrue(
                any(
                    message.startswith("Wrote error summary to ")
                    for message in printed_messages
                )
            )
            self.assertFalse(
                any(message.startswith("Estimated cost ") for message in printed_messages)
            )

    def test_main_writes_combined_summary_for_multiple_tasks(self):
        prepare_coffee_payload = {
            "composite_task": "PrepareCoffee",
            "sdk": "google-genai",
            "model": "gemini-3-flash-preview",
            "model_config": {
                "reasoning": {"thinking_level": None},
                "sampling": {"temperature": 0.2, "strategy": "base"},
            },
            "num_runs": 2,
            "num_trajectories": 1,
            "generated_at": "2026-03-10T00:00:00+00:00",
            "cost_summary": {
                "prompt_tokens": 100,
                "output_tokens": 40,
                "reasoning_tokens": 0,
                "total_tokens": 140,
                "input_cost_usd": 0.001,
                "output_cost_usd": 0.002,
                "total_cost_usd": 0.003,
                "average_trajectory_cost_usd": 0.003,
                "notes": ["prepare"],
            },
            "trajectory_prompts": [],
            "attempt_prompts": [],
            "trajectory_outputs": [],
            "error_events": [],
            "trajectories": [
                {
                    "trajectory_id": "traj_000000",
                    "generation_usage": {
                        "successful_attempt_number": 1,
                        "observed_cost_usd": 0.003,
                    },
                    "validation": {"is_valid": True},
                }
            ],
        }
        hot_dog_payload = {
            "composite_task": "HotDogSetup",
            "sdk": "google-genai",
            "model": "gemini-3-flash-preview",
            "model_config": {
                "reasoning": {"thinking_level": None},
                "sampling": {"temperature": 0.2, "strategy": "base"},
            },
            "num_runs": 2,
            "num_trajectories": 1,
            "generated_at": "2026-03-10T00:00:01+00:00",
            "cost_summary": {
                "prompt_tokens": 120,
                "output_tokens": 50,
                "reasoning_tokens": 0,
                "total_tokens": 170,
                "input_cost_usd": 0.002,
                "output_cost_usd": 0.003,
                "total_cost_usd": 0.005,
                "average_trajectory_cost_usd": 0.005,
                "notes": ["hotdog"],
            },
            "trajectory_prompts": [],
            "attempt_prompts": [],
            "trajectory_outputs": [],
            "error_events": [],
            "trajectories": [
                {
                    "trajectory_id": "traj_000000",
                    "generation_usage": {
                        "successful_attempt_number": 1,
                        "observed_cost_usd": 0.005,
                    },
                    "validation": {"is_valid": True},
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            request_summary_path = Path(tmpdir) / "requests" / "summary.json"
            with mock.patch(
                "data_generation.task_level.generation.cli.generate_trajectories",
                side_effect=[prepare_coffee_payload, hot_dog_payload],
            ):
                with mock.patch(
                    "data_generation.task_level.generation.cli.resolve_request_output_path",
                    return_value=request_summary_path,
                ):
                    exit_code = main(
                        [
                            "--tasks",
                            "PrepareCoffee",
                            "HotDogSetup",
                            "--num-runs",
                            "2",
                        ]
                    )
            self.assertEqual(exit_code, 0)
            self.assertTrue(request_summary_path.exists())
            combined_summary = json.loads(request_summary_path.read_text(encoding="utf-8"))
            combined_cost = json.loads(
                resolve_cost_output_path(request_summary_path).read_text(encoding="utf-8")
            )
            combined_errors = json.loads(
                resolve_error_output_path(request_summary_path).read_text(encoding="utf-8")
            )
            self.assertEqual(
                combined_summary["composite_tasks"],
                ["PrepareCoffee", "HotDogSetup"],
            )
            self.assertEqual(combined_summary["num_runs_per_task"], 2)
            self.assertEqual(combined_summary["total_requested_runs"], 4)
            self.assertEqual(combined_summary["num_trajectories"], 2)
            self.assertEqual(combined_summary["cost_summary"]["total_cost_usd"], 0.008)
            self.assertEqual(len(combined_summary["task_summaries"]), 2)
            self.assertEqual(combined_cost["cost_summary"]["total_cost_usd"], 0.008)
            self.assertEqual(combined_errors["total_errors"], 0)
            self.assertTrue(
                (request_summary_path.parent / "prepare_coffee" / "summary.json").exists()
            )
            self.assertTrue(
                (request_summary_path.parent / "hot_dog_setup" / "summary.json").exists()
            )

    def test_run_cli_exits_immediately_on_keyboard_interrupt(self):
        with mock.patch(
            "data_generation.task_level.generation.cli.main",
            side_effect=KeyboardInterrupt,
        ):
            with mock.patch(
                "data_generation.task_level.generation.cli.os._exit",
                side_effect=SystemExit(INTERRUPTED_EXIT_CODE),
            ) as mocked_exit:
                with mock.patch("builtins.print") as mocked_print:
                    with self.assertRaises(SystemExit) as raised:
                        run_cli([])

        self.assertEqual(raised.exception.code, INTERRUPTED_EXIT_CODE)
        mocked_exit.assert_called_once_with(INTERRUPTED_EXIT_CODE)
        mocked_print.assert_called_once()
        self.assertEqual(mocked_print.call_args.args[0], INTERRUPTED_MESSAGE)
        self.assertEqual(mocked_print.call_args.kwargs["flush"], True)

    def test_run_cli_uses_custom_keyboard_interrupt_message(self):
        with mock.patch(
            "data_generation.task_level.generation.cli.main",
            side_effect=KeyboardInterrupt(BATCH_INTERRUPTED_MESSAGE),
        ):
            with mock.patch(
                "data_generation.task_level.generation.cli.os._exit",
                side_effect=SystemExit(INTERRUPTED_EXIT_CODE),
            ):
                with mock.patch("builtins.print") as mocked_print:
                    with self.assertRaises(SystemExit):
                        run_cli([])

        mocked_print.assert_called_once()
        self.assertEqual(mocked_print.call_args.args[0], BATCH_INTERRUPTED_MESSAGE)

    def test_main_does_not_write_outputs_when_generation_is_interrupted(self):
        with mock.patch(
            "data_generation.task_level.generation.cli.generate_trajectories",
            side_effect=KeyboardInterrupt(BATCH_INTERRUPTED_MESSAGE),
        ):
            with mock.patch(
                "data_generation.task_level.generation.cli._write_generation_outputs"
            ) as write_outputs:
                with self.assertRaises(KeyboardInterrupt):
                    main(
                        [
                            "--batch-processing",
                            "--batch-gcs-prefix",
                            "gs://demo-bucket/batch-prefix",
                        ]
                    )

        write_outputs.assert_not_called()

    def test_run_cli_prints_single_line_for_generation_errors(self):
        with mock.patch(
            "data_generation.task_level.generation.cli.main",
            side_effect=TrajectoryGenerationError(
                "TaskSemanticValidationError: invalid trajectory"
            ),
        ):
            with mock.patch("builtins.print") as mocked_print:
                exit_code = run_cli([])

        self.assertEqual(exit_code, 1)
        mocked_print.assert_called_once()
        self.assertEqual(
            mocked_print.call_args.args[0],
            "TrajectoryGenerationError: TaskSemanticValidationError: invalid trajectory",
        )
        self.assertIs(mocked_print.call_args.kwargs["file"], sys.stderr)
        self.assertEqual(mocked_print.call_args.kwargs["flush"], True)

    def test_module_entrypoint_runs_cli(self):
        with mock.patch.object(
            sys,
            "argv",
            ["data_generation.task_level.generation.cli", "--help"],
        ):
            with self.assertRaises(SystemExit) as raised:
                runpy.run_path(
                    trajectory_generation_module.__file__,
                    run_name="__main__",
                )

        self.assertEqual(raised.exception.code, 0)

    def test_unsupported_task_raises(self):
        runtime_config = RuntimeConfig(
            composite_task="PlaceFoodInBowls",
            num_runs=1,
            model="gemini-3-flash-preview",
            sdk="google-genai",
            project="demo-project",
            location="global",
            temperature=0.5,
            max_workers=1,
            max_retries=1,
        )

        with self.assertRaises(TrajectoryGenerationError):
            generate_trajectories(
                runtime_config,
                client_factory=lambda: SequencedFakeClient([make_valid_candidate()]),
                show_progress=False,
            )


if __name__ == "__main__":
    unittest.main()
