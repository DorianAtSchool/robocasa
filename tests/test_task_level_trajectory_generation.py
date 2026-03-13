import json
import sys
import threading
import unittest
from unittest import mock
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import types

from data_generation.task_level.runtime.client import (
    GenerationResult,
    GenerationUsage,
    GoogleGenAIClient,
    TrajectoryGenerationError,
    load_dotenv_file,
    validate_google_auth,
)
from data_generation.task_level.subatomic_tool_specs import build_allowed_tool_specs
from data_generation.task_level.tasks.base import (
    FiniteStateTaskValidator,
    build_task_response_schema,
)
from data_generation.task_level.tasks import (
    DuplicateTrajectoryValidationError,
    PREPARE_COFFEE_TASK,
    ResponseFormatValidationError,
    TaskSemanticValidationError,
    TrajectoryStructureValidationError,
    TrajectoryValidationError,
)
from data_generation.task_level.tasks.prepare_coffee import (
    PrepareCoffeeValidator,
    build_prepare_coffee_prompt,
)
from data_generation.task_level.subatomic_tool_calls import discover_subatomic_tools
from data_generation.task_level.runtime.batch_generation import BatchRunContext
from data_generation.task_level.trajectory_generation import (
    BATCH_INTERRUPTED_MESSAGE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_OUTPUT_PATH,
    INTERRUPTED_EXIT_CODE,
    INTERRUPTED_MESSAGE,
    PROGRESS_BAR_WIDTH,
    RichProgressDisplay,
    TQDM_BAR_FORMAT,
    ProgressHandles,
    RuntimeConfig,
    _build_preflight_cost_estimate_summary,
    _is_non_retryable_generation_error,
    _maybe_reserve_signature,
    build_cost_output_payload,
    build_summary_output_payload,
    extract_json_candidate,
    generate_trajectories,
    generate_single_trajectory,
    main,
    parse_args,
    resolve_dataset_output_path,
    resolve_cost_output_path,
    resolve_trajectory_output_dir,
    run_cli,
)


def make_valid_candidate(
    *,
    communicate_messages=("I will grab the mug.", "I will be ready at the machine."),
    action_agents=("agent_0", "agent_1", "agent_1"),
    action_reasoning=(
        "The mug starts in the cabinet.",
        "The mug must reach the dispenser next.",
        "The mug is ready for brewing.",
    ),
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

    return {
        "agents": [
            {"agent_id": "agent_0"},
            {"agent_id": "agent_1"},
        ],
        "steps": [
            {
                "step_index": 0,
                "agent_id": "agent_0",
                "tool_name": "communicate",
                "tool_args": {
                    "to_agent_id": "agent_1",
                    "message": communicate_messages[0],
                },
                "entity_refs": {
                    "from_agent_id": "agent_0",
                    "to_agent_id": "agent_1",
                },
                "reasoning": "We need a shared plan before acting.",
            },
            {
                "step_index": 1,
                "agent_id": "agent_1",
                "tool_name": "communicate",
                "tool_args": {
                    "to_agent_id": "agent_0",
                    "message": communicate_messages[1],
                },
                "entity_refs": {
                    "from_agent_id": "agent_1",
                    "to_agent_id": "agent_0",
                },
                "reasoning": "I should confirm the handoff sequence.",
            },
            {
                "step_index": 2,
                "agent_id": expanded_action_agents[0],
                "tool_name": "navigate_to_fixture",
                "tool_args": {
                    "fixture_id": "cabinet_1",
                },
                "entity_refs": {
                    "fixture_id": "cabinet_1",
                },
                "reasoning": expanded_action_reasoning[0],
            },
            {
                "step_index": 3,
                "agent_id": expanded_action_agents[1],
                "tool_name": "open_hinged_part",
                "tool_args": {
                    "target_id": "cabinet_1",
                    "part_id": "door",
                },
                "entity_refs": {
                    "target_id": "cabinet_1",
                    "part_id": "door",
                },
                "reasoning": expanded_action_reasoning[1],
            },
            {
                "step_index": 4,
                "agent_id": expanded_action_agents[2],
                "tool_name": "pick_up_object",
                "tool_args": {
                    "object_id": "mug_1",
                    "source_id": "cabinet_1",
                },
                "entity_refs": {
                    "object_id": "mug_1",
                    "source_id": "cabinet_1",
                },
                "reasoning": expanded_action_reasoning[2],
            },
            {
                "step_index": 5,
                "agent_id": expanded_action_agents[3],
                "tool_name": "navigate_to_fixture",
                "tool_args": {
                    "fixture_id": "counter_1",
                },
                "entity_refs": {
                    "fixture_id": "counter_1",
                },
                "reasoning": expanded_action_reasoning[3],
            },
            {
                "step_index": 6,
                "agent_id": expanded_action_agents[4],
                "tool_name": "place_on_surface",
                "tool_args": {
                    "object_id": "mug_1",
                    "support_id": "counter_1",
                },
                "entity_refs": {
                    "object_id": "mug_1",
                    "support_id": "counter_1",
                },
                "reasoning": expanded_action_reasoning[4],
            },
            {
                "step_index": 7,
                "agent_id": expanded_action_agents[5],
                "tool_name": "navigate_to_fixture",
                "tool_args": {
                    "fixture_id": "counter_1",
                },
                "entity_refs": {
                    "fixture_id": "counter_1",
                },
                "reasoning": expanded_action_reasoning[5],
            },
            {
                "step_index": 8,
                "agent_id": expanded_action_agents[6],
                "tool_name": "pick_up_object",
                "tool_args": {
                    "object_id": "mug_1",
                    "source_id": "counter_1",
                },
                "entity_refs": {
                    "object_id": "mug_1",
                    "source_id": "counter_1",
                },
                "reasoning": expanded_action_reasoning[6],
            },
            {
                "step_index": 9,
                "agent_id": expanded_action_agents[7],
                "tool_name": "navigate_to_fixture",
                "tool_args": {
                    "fixture_id": "coffee_machine_1",
                },
                "entity_refs": {
                    "fixture_id": "coffee_machine_1",
                },
                "reasoning": expanded_action_reasoning[7],
            },
            {
                "step_index": 10,
                "agent_id": expanded_action_agents[8],
                "tool_name": "place_under_dispenser",
                "tool_args": {
                    "object_id": "mug_1",
                    "dispenser_id": "coffee_machine_dispenser",
                },
                "entity_refs": {
                    "object_id": "mug_1",
                    "dispenser_id": "coffee_machine_dispenser",
                },
                "reasoning": expanded_action_reasoning[8],
            },
            {
                "step_index": 11,
                "agent_id": expanded_action_agents[9],
                "tool_name": "press_button",
                "tool_args": {
                    "target_id": "coffee_machine_1",
                    "control_id": "start_button",
                },
                "entity_refs": {
                    "target_id": "coffee_machine_1",
                    "control_id": "start_button",
                },
                "reasoning": expanded_action_reasoning[9],
            },
        ],
    }


def make_invalid_candidate_missing_initial_communication():
    candidate = make_valid_candidate()
    candidate["steps"] = candidate["steps"][1:]
    for index, step in enumerate(candidate["steps"]):
        step["step_index"] = index
    return candidate


def make_alternative_valid_candidate():
    """Builds a valid PrepareCoffee trace with a different legal action ordering."""

    candidate = make_valid_candidate()
    candidate["steps"].insert(
        2,
        {
            "step_index": 2,
            "agent_id": "agent_1",
            "tool_name": "navigate_to_fixture",
            "tool_args": {
                "fixture_id": "coffee_machine_1",
            },
            "entity_refs": {
                "fixture_id": "coffee_machine_1",
            },
            "reasoning": "I can stage at the machine before the mug arrives.",
        },
    )
    for index, step in enumerate(candidate["steps"]):
        step["step_index"] = index
    return candidate


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
        "navigate_to_fixture",
        "open_hinged_part",
        "pick_up_object",
        "place_on_surface",
    )
)


def make_toy_action_spec(
    tool_name,
    tool_args,
    entity_refs,
):
    """Builds a compact toy action payload for shared FSM unit tests."""

    return {
        "tool_name": tool_name,
        "tool_args": dict(tool_args),
        "entity_refs": dict(entity_refs),
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
            "step_index": 0,
            "agent_id": "agent_0",
            "tool_name": "communicate",
            "tool_args": {
                "to_agent_id": "agent_1",
                "message": "I will handle the shared FSM test actions.",
            },
            "entity_refs": {
                "from_agent_id": "agent_0",
                "to_agent_id": "agent_1",
            },
            "reasoning": "We should coordinate before the first action.",
        },
        {
            "step_index": 1,
            "agent_id": "agent_1",
            "tool_name": "communicate",
            "tool_args": {
                "to_agent_id": "agent_0",
                "message": "I will stay clear while you execute the sequence.",
            },
            "entity_refs": {
                "from_agent_id": "agent_1",
                "to_agent_id": "agent_0",
            },
            "reasoning": "I should confirm the test setup.",
        },
    ]

    for index, (action_spec, agent_id) in enumerate(
        zip(action_sequence, action_agents),
        start=2,
    ):
        steps.append(
            {
                "step_index": index,
                "agent_id": agent_id,
                "tool_name": action_spec["tool_name"],
                "tool_args": dict(action_spec["tool_args"]),
                "entity_refs": dict(action_spec["entity_refs"]),
                "reasoning": f"Shared FSM test action {index - 1}.",
            }
        )

    return {
        "agents": [
            {"agent_id": "agent_0"},
            {"agent_id": "agent_1"},
        ],
        "steps": steps,
    }


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


class FakeClient:
    def generate(self, *, model, prompt, response_schema, temperature):
        raise NotImplementedError


class SequencedFakeClient(FakeClient):
    def __init__(self, responses):
        self._responses = list(responses)

    def generate(self, *, model, prompt, response_schema, temperature):
        if not self._responses:
            raise AssertionError("No more fake responses configured.")
        return self._responses.pop(0)


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
    row = {
        "request": {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": build_prepare_coffee_prompt(variation_key)}],
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
        self.assertIn("pick_up_object", tool_names)
        self.assertIn("place_under_dispenser", tool_names)
        self.assertIn("press_button", tool_names)
        self.assertTrue(all(tool_name == tool_name.lower() for tool_name in tool_names))

    def test_prepare_coffee_prompt_contains_subatomic_catalog_and_rules(self):
        prompt = build_prepare_coffee_prompt("unit-test")
        self.assertIn("pick_up_object", prompt)
        self.assertIn("place_under_dispenser", prompt)
        self.assertIn("press_button", prompt)
        self.assertIn("communicate", prompt)
        self.assertIn("variation key: unit-test", prompt)
        self.assertIn("Each agent entry must contain only: agent_id.", prompt)
        self.assertNotIn(
            "Each agent entry must contain: agent_id, role, initial_plan.",
            prompt,
        )


class DotenvLoadingTests(unittest.TestCase):
    def test_legacy_client_module_reexports_runtime_helpers(self):
        from data_generation.task_level import client as legacy_client
        from data_generation.task_level.runtime import client as runtime_client

        self.assertEqual(legacy_client.DEFAULT_LOCATION, runtime_client.DEFAULT_LOCATION)
        self.assertIs(legacy_client.load_dotenv_file, runtime_client.load_dotenv_file)

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
        runtime_config = parse_args(
            [
                "--output",
                "/tmp/trajectories.json",
                "--cost-output",
                "/tmp/custom_costs.json",
            ]
        )

        self.assertEqual(runtime_config.output_path, Path("/tmp/trajectories.json"))
        self.assertEqual(
            runtime_config.cost_output_path, Path("/tmp/custom_costs.json")
        )

    def test_parse_args_disables_validation_by_default(self):
        runtime_config = parse_args([])
        self.assertTrue(runtime_config.disable_validation)

    def test_parse_args_accepts_enable_validation(self):
        runtime_config = parse_args(["--enable-validation"])
        self.assertFalse(runtime_config.disable_validation)

    def test_parse_args_accepts_num_trajectories_flag_and_alias(self):
        dashed_runtime_config = parse_args(["--num-trajectories", "7"])
        underscored_runtime_config = parse_args(["--num_trajectories", "9"])

        self.assertEqual(dashed_runtime_config.num_trajectories, 7)
        self.assertEqual(underscored_runtime_config.num_trajectories, 9)

    def test_parse_args_accepts_task_flag_and_legacy_aliases(self):
        task_runtime_config = parse_args(["--task", "PrepareCoffee"])
        dashed_legacy_runtime_config = parse_args(
            ["--composite-task", "PrepareCoffee"]
        )
        underscored_legacy_runtime_config = parse_args(
            ["--composite_task", "PrepareCoffee"]
        )

        self.assertEqual(task_runtime_config.composite_task, "PrepareCoffee")
        self.assertEqual(
            dashed_legacy_runtime_config.composite_task,
            "PrepareCoffee",
        )
        self.assertEqual(
            underscored_legacy_runtime_config.composite_task,
            "PrepareCoffee",
        )

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
            DEFAULT_OUTPUT_PATH,
            "PrepareCoffee",
            generated_at=datetime(2026, 3, 10, 12, 34, 56, tzinfo=timezone.utc),
        )

        self.assertEqual(
            resolved,
            DEFAULT_OUTPUT_DIR
            / "prepare_coffee"
            / "20260310T123456Z"
            / "prepare_coffee_trajectories.json",
        )

    def test_resolve_dataset_output_path_keeps_explicit_output(self):
        explicit_output = Path("/tmp/trajectories.json")

        self.assertEqual(
            resolve_dataset_output_path(
                explicit_output,
                "PrepareCoffee",
                generated_at=datetime(2026, 3, 10, 12, 34, 56, tzinfo=timezone.utc),
            ),
            explicit_output,
        )

    def test_resolve_trajectory_output_dir_uses_sibling_trajectories_directory(self):
        output_path = Path("/tmp/prepare_coffee_trajectories.json")

        self.assertEqual(
            resolve_trajectory_output_dir(output_path),
            Path("/tmp/trajectories"),
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


class FiniteStateTaskValidatorTests(unittest.TestCase):
    def test_build_task_response_schema_uses_allowed_tool_specs(self):
        allowed_tool_specs = build_allowed_tool_specs(
            (
                "communicate",
                "pick_up_object",
                "place_in_receptacle",
                "place_on_object",
                "set_rotary_control",
            )
        )

        response_schema = build_task_response_schema(
            agent_ids=("agent_0", "agent_1"),
            allowed_tool_specs=allowed_tool_specs,
            min_steps=4,
        )

        step_properties = response_schema["properties"]["steps"]["items"]["properties"]
        self.assertEqual(
            list(step_properties["tool_name"]["enum"]),
            list(allowed_tool_specs),
        )
        self.assertEqual(
            list(step_properties["tool_args"]["properties"]),
            [
                "to_agent_id",
                "message",
                "object_id",
                "source_id",
                "receptacle_id",
                "support_object_id",
                "target_id",
                "control_id",
                "goal",
            ],
        )
        self.assertEqual(
            list(step_properties["entity_refs"]["properties"]),
            [
                "from_agent_id",
                "to_agent_id",
                "object_id",
                "source_id",
                "receptacle_id",
                "support_object_id",
                "target_id",
                "control_id",
                "goal",
            ],
        )
        self.assertEqual(
            step_properties["tool_args"]["properties"]["to_agent_id"]["enum"],
            ["agent_0", "agent_1"],
        )
        self.assertEqual(
            step_properties["entity_refs"]["properties"]["from_agent_id"]["enum"],
            ["agent_0", "agent_1"],
        )
        self.assertEqual(response_schema["properties"]["steps"]["minItems"], 4)

    def test_validator_allows_navigation_while_holding(self):
        actions = (
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "table_1"},
                {"fixture_id": "table_1"},
            ),
            make_toy_action_spec(
                "pick_up_object",
                {"object_id": "apple_1", "source_id": "table_1"},
                {"object_id": "apple_1", "source_id": "table_1"},
            ),
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "shelf_1"},
                {"fixture_id": "shelf_1"},
            ),
            make_toy_action_spec(
                "place_on_surface",
                {"object_id": "apple_1", "support_id": "shelf_1"},
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

    def test_validator_accepts_alternative_valid_action_order(self):
        actions = (
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "cabinet_1"},
                {"fixture_id": "cabinet_1"},
            ),
            make_toy_action_spec(
                "open_hinged_part",
                {"target_id": "cabinet_1", "part_id": "door"},
                {"target_id": "cabinet_1", "part_id": "door"},
            ),
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "table_1"},
                {"fixture_id": "table_1"},
            ),
            make_toy_action_spec(
                "pick_up_object",
                {"object_id": "apple_1", "source_id": "table_1"},
                {"object_id": "apple_1", "source_id": "table_1"},
            ),
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "shelf_1"},
                {"fixture_id": "shelf_1"},
            ),
            make_toy_action_spec(
                "place_on_surface",
                {"object_id": "apple_1", "support_id": "shelf_1"},
                {"object_id": "apple_1", "support_id": "shelf_1"},
            ),
        )

        validator = ToyFiniteStateValidator()
        validation = validator.validate(make_toy_candidate(actions))

        self.assertTrue(validation["is_valid"])

    def test_validator_rejects_second_pickup_while_holding(self):
        actions = (
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "table_1"},
                {"fixture_id": "table_1"},
            ),
            make_toy_action_spec(
                "pick_up_object",
                {"object_id": "apple_1", "source_id": "table_1"},
                {"object_id": "apple_1", "source_id": "table_1"},
            ),
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "table_1"},
                {"fixture_id": "table_1"},
            ),
            make_toy_action_spec(
                "pick_up_object",
                {"object_id": "mug_1", "source_id": "table_1"},
                {"object_id": "mug_1", "source_id": "table_1"},
            ),
        )

        validator = ToyFiniteStateValidator()

        with self.assertRaises(TaskSemanticValidationError) as raised:
            validator.validate(make_toy_candidate(actions))

        self.assertIn("cannot pick up a second object", str(raised.exception))

    def test_validator_rejects_empty_hand_only_tool_while_holding(self):
        actions = (
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "table_1"},
                {"fixture_id": "table_1"},
            ),
            make_toy_action_spec(
                "pick_up_object",
                {"object_id": "apple_1", "source_id": "table_1"},
                {"object_id": "apple_1", "source_id": "table_1"},
            ),
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "cabinet_1"},
                {"fixture_id": "cabinet_1"},
            ),
            make_toy_action_spec(
                "open_hinged_part",
                {"target_id": "cabinet_1", "part_id": "door"},
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
                {"fixture_id": "shelf_1"},
            ),
            make_toy_action_spec(
                "place_on_surface",
                {"object_id": "apple_1", "support_id": "shelf_1"},
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
                {"target_id": "cabinet_1", "part_id": "door"},
            ),
        )

        validator = ToyFiniteStateValidator()

        with self.assertRaises(TaskSemanticValidationError) as raised:
            validator.validate(make_toy_candidate(actions))

        self.assertIn("must navigate to cabinet_1", str(raised.exception))

    def test_validator_rejects_missing_initial_communication(self):
        actions = (
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "table_1"},
                {"fixture_id": "table_1"},
            ),
        )
        candidate = make_toy_candidate(actions)
        candidate["steps"] = candidate["steps"][1:]
        for index, step in enumerate(candidate["steps"]):
            step["step_index"] = index

        validator = ToyFiniteStateValidator()

        with self.assertRaises(TaskSemanticValidationError) as raised:
            validator.validate(candidate)

        self.assertIn(
            "Both agents must coordinate via communication before the first task action.",
            str(raised.exception),
        )

    def test_validator_rejects_legal_trace_that_never_reaches_goal(self):
        actions = (
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "table_1"},
                {"fixture_id": "table_1"},
            ),
            make_toy_action_spec(
                "pick_up_object",
                {"object_id": "apple_1", "source_id": "table_1"},
                {"object_id": "apple_1", "source_id": "table_1"},
            ),
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "shelf_1"},
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
                {"fixture_id": "table_1"},
            ),
            make_toy_action_spec(
                "pick_up_object",
                {"object_id": "apple_1", "source_id": "table_1"},
                {"object_id": "apple_1", "source_id": "table_1"},
            ),
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "shelf_1"},
                {"fixture_id": "shelf_1"},
            ),
            make_toy_action_spec(
                "place_on_surface",
                {"object_id": "apple_1", "support_id": "shelf_1"},
                {"object_id": "apple_1", "support_id": "shelf_1"},
            ),
            make_toy_action_spec(
                "navigate_to_fixture",
                {"fixture_id": "cabinet_1"},
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
        for index, step in enumerate(candidate["steps"]):
            step["step_index"] = index

        with self.assertRaises(TrajectoryValidationError):
            self.validator.validate(candidate)

    def test_validator_accepts_alternative_valid_order(self):
        validation = self.validator.validate(make_alternative_valid_candidate())

        self.assertTrue(validation["is_valid"])
        self.assertTrue(validation["final_state"]["coffee_machine_started"])

    def test_validator_rejects_broken_entity_continuity(self):
        candidate = make_valid_candidate()
        candidate["steps"][4]["tool_args"]["object_id"] = "mug_2"

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
        candidate["steps"].pop(7)
        for index, step in enumerate(candidate["steps"]):
            step["step_index"] = index

        with self.assertRaises(TaskSemanticValidationError):
            self.validator.validate(candidate)

    def test_validator_rejects_missing_cabinet_open_before_pickup(self):
        candidate = make_valid_candidate()
        candidate["steps"].pop(3)
        for index, step in enumerate(candidate["steps"]):
            step["step_index"] = index

        with self.assertRaises(TaskSemanticValidationError):
            self.validator.validate(candidate)

    def test_validator_rejects_invalid_hold_place_ordering(self):
        candidate = make_valid_candidate()
        candidate["steps"][6]["agent_id"] = "agent_1"

        with self.assertRaises(TaskSemanticValidationError):
            self.validator.validate(candidate)

    def test_validator_rejects_unsupported_tool_name(self):
        candidate = make_valid_candidate()
        candidate["steps"][4]["tool_name"] = "unsupported_tool"
        candidate["steps"][4]["tool_args"] = {
            "object_id": "mug_1",
        }
        candidate["steps"][4]["entity_refs"] = {
            "object_id": "mug_1",
        }

        with self.assertRaises(TaskSemanticValidationError):
            self.validator.validate(candidate)

    def test_validator_rejects_invalid_task_specific_control_id(self):
        candidate = make_valid_candidate()
        candidate["steps"][11]["tool_args"]["control_id"] = "wrong_button"
        candidate["steps"][11]["entity_refs"]["control_id"] = "wrong_button"

        with self.assertRaises(TaskSemanticValidationError) as raised:
            self.validator.validate(candidate)

        self.assertIn("press_button requires control_id", str(raised.exception))

    def test_validator_rejects_missing_reasoning(self):
        candidate = make_valid_candidate()
        candidate["steps"][2]["reasoning"] = ""

        with self.assertRaises(TrajectoryStructureValidationError):
            self.validator.validate(candidate)

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


class GenerationTests(unittest.TestCase):
    def test_generate_trajectories_cancels_pending_futures_on_keyboard_interrupt(self):
        runtime_config = RuntimeConfig(
            composite_task="PrepareCoffee",
            num_trajectories=2,
            output_path=mock.sentinel.output_path,
            model="gemini-3-flash-preview",
            sdk="google-genai",
            project="demo-project",
            location="global",
            temperature=0.5,
            max_workers=2,
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
            "data_generation.task_level.trajectory_generation._build_preflight_cost_estimate_summary",
            return_value={"best_case_total_usd": None, "worst_case_total_usd": None},
        ):
            with mock.patch(
                "data_generation.task_level.trajectory_generation._create_progress_handles",
                return_value=ProgressHandles(
                    display=None,
                    overall_progress=None,
                    trajectory_progress_bars=[None, None],
                    log_writer=None,
                ),
            ):
                with mock.patch(
                    "data_generation.task_level.trajectory_generation._close_progress_handles"
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
            num_trajectories=1,
            output_path=Path("/tmp/prepare_coffee_trajectories.json"),
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
                num_trajectories=2,
                output_path=Path(tmpdir) / "prepare_coffee_trajectories.json",
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
                                "traj-001-attempt-00",
                                candidate=make_valid_candidate(
                                    communicate_messages=(
                                        "I will handle the machine.",
                                        "I will clear the cabinet route.",
                                    ),
                                    action_agents=("agent_1", "agent_0", "agent_0"),
                                ),
                            ),
                            make_batch_output_row(
                                "traj-000-attempt-00",
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
            ["traj_000", "traj_001"],
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
        self.assertIn("traj-000-attempt-00", batch_storage.upload_calls[0]["text"])
        self.assertIn("traj-001-attempt-00", batch_storage.upload_calls[0]["text"])
        self.assertEqual(batch_service.create_calls[0]["output_prefix"], round_output_prefix)

    def test_batch_processing_retries_only_outstanding_trajectories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_config = RuntimeConfig(
                composite_task="PrepareCoffee",
                num_trajectories=2,
                output_path=Path(tmpdir) / "prepare_coffee_trajectories.json",
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
                                "traj-000-attempt-00",
                                candidate=make_valid_candidate(),
                            ),
                            make_batch_output_row(
                                "traj-001-attempt-00",
                                status="internal error",
                            ),
                        ],
                    ),
                    round_two_output_prefix: make_batch_download(
                        "gs://demo-bucket/output/round-02.jsonl",
                        [
                            make_batch_output_row(
                                "traj-001-attempt-01",
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
        self.assertIn("traj-001-attempt-01", batch_storage.upload_calls[1]["text"])
        self.assertNotIn("traj-000-attempt-01", batch_storage.upload_calls[1]["text"])

    def test_batch_processing_disable_validation_keeps_invalid_trajectory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_config = RuntimeConfig(
                composite_task="PrepareCoffee",
                num_trajectories=1,
                output_path=Path(tmpdir) / "prepare_coffee_trajectories.json",
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
                                "traj-000-attempt-00",
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
                num_trajectories=1,
                output_path=Path(tmpdir) / "prepare_coffee_trajectories.json",
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
                                "traj-000-attempt-00",
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
                num_trajectories=1,
                output_path=Path(tmpdir) / "prepare_coffee_trajectories.json",
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
                num_trajectories=1,
                output_path=Path(tmpdir) / "prepare_coffee_trajectories.json",
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

    def test_saved_valid_trajectory_uses_agent_ids_and_communication_tool_args(self):
        runtime_config = RuntimeConfig(
            composite_task="PrepareCoffee",
            num_trajectories=1,
            output_path=mock.sentinel.output_path,
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

        trajectory = payload["trajectories"][0]
        self.assertEqual(
            trajectory["agents"],
            [{"agent_id": "agent_0"}, {"agent_id": "agent_1"}],
        )
        self.assertEqual(trajectory["steps"][0]["tool_name"], "communicate")
        self.assertEqual(
            trajectory["steps"][0]["tool_args"],
            {
                "to_agent_id": "agent_1",
                "message": "I will grab the mug.",
            },
        )
        self.assertNotIn("message", trajectory["steps"][0])

    def test_permission_denied_errors_are_treated_as_non_retryable(self):
        self.assertTrue(
            _is_non_retryable_generation_error(
                Exception(
                    "403 PERMISSION_DENIED. Permission 'aiplatform.endpoints.predict' denied."
                )
            )
        )

    def test_parallel_generation_retries_duplicate_and_preserves_order(self):
        runtime_config = RuntimeConfig(
            composite_task="PrepareCoffee",
            num_trajectories=2,
            output_path=mock.sentinel.output_path,
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
            ["traj_000", "traj_001"],
        )
        self.assertNotEqual(
            payload["trajectories"][0]["validation"]["signature"],
            payload["trajectories"][1]["validation"]["signature"],
        )

    def test_cost_summary_uses_usage_metadata_when_available(self):
        runtime_config = RuntimeConfig(
            composite_task="PrepareCoffee",
            num_trajectories=1,
            output_path=mock.sentinel.output_path,
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
            num_trajectories=1,
            output_path=mock.sentinel.output_path,
            model="gemini-3-flash-preview",
            sdk="google-genai",
            project="demo-project",
            location="global",
            temperature=0.5,
            max_workers=1,
            max_retries=3,
        )

        with mock.patch(
            "data_generation.task_level.trajectory_generation._log_runtime_message"
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

        self.assertEqual(log_runtime_message.call_count, 2)
        projected_log = log_runtime_message.call_args_list[0]
        trajectory_log = log_runtime_message.call_args_list[1]

        self.assertIn("Projected cost", projected_log.args[0])
        self.assertIn("best case 1 try / worst case 3 tries", projected_log.args[0])
        self.assertEqual(projected_log.kwargs["enabled"], True)

        self.assertIn("traj_000", trajectory_log.args[0])
        self.assertIn("attempt 1/3", trajectory_log.args[0])
        self.assertNotIn("$0.0013", trajectory_log.args[0])
        self.assertEqual(trajectory_log.kwargs["enabled"], True)

    def test_generate_single_trajectory_updates_progress_status_with_cost(self):
        self.assertEqual(PROGRESS_BAR_WIDTH, 30)
        self.assertIn("{bar:30}", TQDM_BAR_FORMAT)
        runtime_config = RuntimeConfig(
            composite_task="PrepareCoffee",
            num_trajectories=1,
            output_path=mock.sentinel.output_path,
            model="gemini-3-flash-preview",
            sdk="google-genai",
            project="demo-project",
            location="global",
            temperature=0.5,
            max_workers=1,
            max_retries=3,
        )
        trajectory_progress = mock.Mock()
        trajectory_progress.total = runtime_config.max_retries

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

        self.assertEqual(trajectory_progress.total, 1)
        self.assertEqual(
            trajectory_progress.set_postfix_str.call_args_list[0].args[0],
            "attempt 1/3 generating",
        )
        self.assertEqual(
            trajectory_progress.set_postfix_str.call_args_list[-1].args[0],
            "done $0.0013 calls=12",
        )
        trajectory_progress.refresh.assert_called_once()

    def test_generate_single_trajectory_retry_status_includes_tool_call_count(self):
        runtime_config = RuntimeConfig(
            composite_task="PrepareCoffee",
            num_trajectories=1,
            output_path=mock.sentinel.output_path,
            model="gemini-3-flash-preview",
            sdk="google-genai",
            project="demo-project",
            location="global",
            temperature=0.5,
            max_workers=1,
            max_retries=2,
        )
        trajectory_progress = mock.Mock()
        trajectory_progress.total = runtime_config.max_retries

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
        self.assertIn("attempt 1/2 retry calls=11", status_updates)
        self.assertTrue(status_updates[-1].startswith("done $"))
        self.assertTrue(status_updates[-1].endswith(" calls=12"))

    def test_generate_single_trajectory_hides_attempt_counts_when_validation_disabled(self):
        runtime_config = RuntimeConfig(
            composite_task="PrepareCoffee",
            num_trajectories=1,
            output_path=mock.sentinel.output_path,
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
        trajectory_progress.total = runtime_config.max_retries

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
        self.assertEqual(status_updates[0], "generating")
        self.assertIn("retrying", status_updates)
        self.assertNotIn("valid", status_updates)
        self.assertTrue(status_updates[-1].startswith("done $"))
        self.assertIn("calls=11", status_updates[-1])
        self.assertTrue(
            status_updates[-1].endswith(" invalid TaskSemanticValidationError")
        )
        self.assertTrue(all("attempt " not in status for status in status_updates))

    def test_rich_progress_display_uses_non_expanding_layout(self):
        runtime_config = RuntimeConfig(
            composite_task="PrepareCoffee",
            num_trajectories=2,
            output_path=mock.sentinel.output_path,
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
            "data_generation.task_level.trajectory_generation.Console",
            return_value=mock.sentinel.console,
        ) as console_cls:
            with mock.patch(
                "data_generation.task_level.trajectory_generation.SpinnerColumn",
                return_value=mock.sentinel.spinner_column,
            ):
                with mock.patch(
                    "data_generation.task_level.trajectory_generation.TextColumn",
                    side_effect=[
                        mock.sentinel.description_column,
                        mock.sentinel.status_column,
                    ],
                ):
                    with mock.patch(
                        "data_generation.task_level.trajectory_generation.BarColumn",
                        return_value=mock.sentinel.bar_column,
                    ):
                        with mock.patch(
                            "data_generation.task_level.trajectory_generation.TaskProgressColumn",
                            return_value=mock.sentinel.task_progress_column,
                        ):
                            with mock.patch(
                                "data_generation.task_level.trajectory_generation.MofNCompleteColumn",
                                return_value=mock.sentinel.mofn_column,
                            ):
                                with mock.patch(
                                    "data_generation.task_level.trajectory_generation.TimeElapsedColumn",
                                    return_value=mock.sentinel.elapsed_column,
                                ):
                                    with mock.patch(
                                        "data_generation.task_level.trajectory_generation.RichProgress",
                                        return_value=fake_progress,
                                    ) as rich_progress:
                                        display = RichProgressDisplay(runtime_config)

        console_cls.assert_called_once_with(stderr=True)
        self.assertEqual(rich_progress.call_args.kwargs["console"], mock.sentinel.console)
        self.assertFalse(rich_progress.call_args.kwargs["expand"])
        fake_progress.start.assert_called_once()
        self.assertEqual(fake_progress.add_task.call_count, 3)
        display.close()
        fake_progress.stop.assert_called_once()

    def test_cost_summary_falls_back_to_heuristic_without_usage_metadata(self):
        runtime_config = RuntimeConfig(
            composite_task="PrepareCoffee",
            num_trajectories=1,
            output_path=mock.sentinel.output_path,
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

    def test_preflight_cost_estimate_uses_manual_task_token_estimate(self):
        runtime_config = RuntimeConfig(
            composite_task="PrepareCoffee",
            num_trajectories=2,
            output_path=mock.sentinel.output_path,
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
            {"prompt": 6220, "output": 7200, "reasoning": 0, "total": 13420},
        )
        self.assertAlmostEqual(summary["best_case_total_usd"], 0.0247)
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

    def test_post_run_cost_summary_excludes_failed_retries(self):
        runtime_config = RuntimeConfig(
            composite_task="PrepareCoffee",
            num_trajectories=1,
            output_path=mock.sentinel.output_path,
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
            1250,
        )
        self.assertAlmostEqual(
            payload["cost_summary"]["total_cost_usd"],
            0.0013,
        )
        self.assertEqual(
            payload["cost_summary"]["notes"][1],
            "Retry attempts that did not produce a saved trajectory are not included.",
        )
        self.assertNotIn("cost_estimate", payload)

    def test_disable_validation_keeps_invalid_trajectory_and_records_error(self):
        runtime_config = RuntimeConfig(
            composite_task="PrepareCoffee",
            num_trajectories=1,
            output_path=mock.sentinel.output_path,
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

        trajectory = payload["trajectories"][0]
        self.assertFalse(trajectory["validation"]["is_valid"])
        self.assertTrue(trajectory["validation"]["validation_disabled"])
        self.assertEqual(
            trajectory["validation"]["error_type"],
            "TaskSemanticValidationError",
        )
        self.assertIn(
            "Both agents must coordinate via communication before the first task action.",
            trajectory["validation"]["error"],
        )
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
            num_trajectories=2,
            output_path=mock.sentinel.output_path,
            model="gemini-3-flash-preview",
            sdk="google-genai",
            project="demo-project",
            location="global",
            temperature=0.5,
            max_workers=2,
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
            num_trajectories=1,
            output_path=mock.sentinel.output_path,
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
            "data_generation.task_level.trajectory_generation._log_runtime_message"
        ) as log_runtime_message:
            generate_trajectories(
                runtime_config,
                client_factory=lambda: SequencedFakeClient(
                    [make_invalid_candidate_missing_initial_communication()]
                ),
                show_progress=True,
            )

        logged_messages = [call.args[0] for call in log_runtime_message.call_args_list]
        self.assertEqual(len(logged_messages), 2)
        self.assertTrue(logged_messages[0].startswith("Projected cost: $"))
        self.assertNotIn("best case", logged_messages[0])
        self.assertNotIn("worst case", logged_messages[0])
        self.assertNotIn("attempt ", logged_messages[0])
        self.assertEqual(
            logged_messages[1],
            "Generated traj_000 invalid "
            "TaskSemanticValidationError: Both agents must coordinate via "
            "communication before the first task action.",
        )
        self.assertNotIn("attempt ", logged_messages[1])

    def test_disable_validation_output_payloads_omit_attempt_number(self):
        runtime_config = RuntimeConfig(
            composite_task="PrepareCoffee",
            num_trajectories=1,
            output_path=Path("/tmp/prepare_coffee_trajectories.json"),
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

        summary_payload = build_summary_output_payload(payload)
        cost_payload = build_cost_output_payload(
            payload,
            trajectory_output_path=runtime_config.output_path,
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
            "TaskSemanticValidationError",
        )
        self.assertEqual(
            cost_payload["trajectory_costs"][0]["generation_usage"]["observed_cost_usd"],
            payload["trajectories"][0]["generation_usage"]["observed_cost_usd"],
        )

    def test_resolve_cost_output_path_defaults_to_output_stem_with_costs_suffix(self):
        self.assertEqual(
            resolve_cost_output_path(Path("/tmp/prepare_coffee_trajectories.json")),
            Path("/tmp/prepare_coffee_trajectories_costs.json"),
        )

    def test_build_cost_output_payload_extracts_only_cost_data(self):
        runtime_config = RuntimeConfig(
            composite_task="PrepareCoffee",
            num_trajectories=1,
            output_path=Path("/tmp/prepare_coffee_trajectories.json"),
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

        cost_payload = build_cost_output_payload(
            payload,
            trajectory_output_path=runtime_config.output_path,
        )

        self.assertEqual(
            cost_payload["trajectory_output_path"],
            str(runtime_config.output_path),
        )
        self.assertEqual(
            cost_payload["trajectory_directory"],
            str(runtime_config.output_path.parent / "trajectories"),
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
            num_trajectories=1,
            output_path=Path("/tmp/prepare_coffee_trajectories.json"),
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
        self.assertEqual(summary_payload["trajectory_directory"], "trajectories")
        self.assertEqual(
            summary_payload["trajectory_files"],
            [
                {
                    "trajectory_id": payload["trajectories"][0]["trajectory_id"],
                    "path": "trajectories/traj_000.json",
                }
            ],
        )
        self.assertNotIn("trajectories", summary_payload)

    def test_main_writes_costs_to_separate_file(self):
        fixed_payload = {
            "composite_task": "PrepareCoffee",
            "sdk": "google-genai",
            "model": "gemini-3-flash-preview",
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
            "trajectories": [
                {
                    "trajectory_id": "traj_000",
                    "generation_usage": {
                        "successful_attempt_number": 1,
                        "observed_cost_usd": 0.001,
                    },
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "trajectories.json"
            cost_output_path = Path(tmpdir) / "costs.json"

            with mock.patch(
                "data_generation.task_level.trajectory_generation.generate_trajectories",
                return_value=fixed_payload,
            ):
                with mock.patch("builtins.print") as mocked_print:
                    exit_code = main(
                        [
                            "--output",
                            str(output_path),
                            "--cost-output",
                            str(cost_output_path),
                        ]
                    )

            self.assertEqual(exit_code, 0)
            self.assertTrue(output_path.exists())
            self.assertTrue(cost_output_path.exists())

            summary_payload = json.loads(output_path.read_text(encoding="utf-8"))
            cost_payload = json.loads(cost_output_path.read_text(encoding="utf-8"))
            trajectory_output_dir = output_path.parent / "trajectories"
            trajectory_path = trajectory_output_dir / "traj_000.json"

            self.assertTrue(trajectory_path.exists())
            self.assertEqual(
                json.loads(trajectory_path.read_text(encoding="utf-8")),
                fixed_payload["trajectories"][0],
            )
            self.assertEqual(
                summary_payload,
                {
                    "composite_task": "PrepareCoffee",
                    "sdk": "google-genai",
                    "model": "gemini-3-flash-preview",
                    "num_trajectories": 1,
                    "generated_at": "2026-03-10T00:00:00+00:00",
                    "cost_summary": {
                        "prompt_tokens": 1000,
                        "output_tokens": 250,
                        "total_tokens": 1250,
                        "total_cost_usd": 0.001,
                    },
                    "trajectory_directory": "trajectories",
                    "trajectory_files": [
                        {
                            "trajectory_id": "traj_000",
                            "path": "trajectories/traj_000.json",
                        }
                    ],
                },
            )
            self.assertNotIn("project", summary_payload)
            self.assertNotIn("location", summary_payload)
            self.assertEqual(
                cost_payload["trajectory_output_path"],
                str(output_path),
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
            self.assertNotIn("cost_estimate", cost_payload)
            self.assertEqual(
                cost_payload["trajectory_costs"],
                [
                    {
                        "trajectory_id": "traj_000",
                        "generation_usage": {
                            "successful_attempt_number": 1,
                            "observed_cost_usd": 0.001,
                        },
                    }
                ],
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
            self.assertFalse(
                any(message.startswith("Estimated cost ") for message in printed_messages)
            )

    def test_run_cli_exits_immediately_on_keyboard_interrupt(self):
        with mock.patch(
            "data_generation.task_level.trajectory_generation.main",
            side_effect=KeyboardInterrupt,
        ):
            with mock.patch(
                "data_generation.task_level.trajectory_generation.os._exit",
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
            "data_generation.task_level.trajectory_generation.main",
            side_effect=KeyboardInterrupt(BATCH_INTERRUPTED_MESSAGE),
        ):
            with mock.patch(
                "data_generation.task_level.trajectory_generation.os._exit",
                side_effect=SystemExit(INTERRUPTED_EXIT_CODE),
            ):
                with mock.patch("builtins.print") as mocked_print:
                    with self.assertRaises(SystemExit):
                        run_cli([])

        mocked_print.assert_called_once()
        self.assertEqual(mocked_print.call_args.args[0], BATCH_INTERRUPTED_MESSAGE)

    def test_main_does_not_write_outputs_when_generation_is_interrupted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "trajectories.json"

            with mock.patch(
                "data_generation.task_level.trajectory_generation.generate_trajectories",
                side_effect=KeyboardInterrupt(BATCH_INTERRUPTED_MESSAGE),
            ):
                with mock.patch(
                    "data_generation.task_level.trajectory_generation._write_generation_outputs"
                ) as write_outputs:
                    with self.assertRaises(KeyboardInterrupt):
                        main(
                            [
                                "--output",
                                str(output_path),
                                "--batch-processing",
                                "--batch-gcs-prefix",
                                "gs://demo-bucket/batch-prefix",
                            ]
                        )

        write_outputs.assert_not_called()

    def test_run_cli_prints_single_line_for_generation_errors(self):
        with mock.patch(
            "data_generation.task_level.trajectory_generation.main",
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

    def test_unsupported_task_raises(self):
        runtime_config = RuntimeConfig(
            composite_task="PlaceFoodInBowls",
            num_trajectories=1,
            output_path=mock.sentinel.output_path,
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
