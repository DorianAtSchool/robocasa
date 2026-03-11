import json
import unittest
from unittest import mock
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import types

from data_generation.task_level.client import (
    GenerationResult,
    GenerationUsage,
    GoogleGenAIClient,
    TrajectoryGenerationError,
    load_dotenv_file,
    validate_google_auth,
)
from data_generation.task_level.tasks import (
    PREPARE_COFFEE_TASK,
    TrajectoryValidationError,
)
from data_generation.task_level.tasks.prepare_coffee import (
    PrepareCoffeeValidator,
    build_prepare_coffee_prompt,
)
from data_generation.task_level.tool_calls import discover_atomic_tools
from data_generation.task_level.trajectory_generation import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_OUTPUT_PATH,
    INTERRUPTED_EXIT_CODE,
    INTERRUPTED_MESSAGE,
    ProgressHandles,
    RuntimeConfig,
    _build_preflight_cost_estimate_summary,
    _is_non_retryable_generation_error,
    build_cost_output_payload,
    build_summary_output_payload,
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
                "agent_id": action_agents[0],
                "tool_name": "PickPlaceCabinetToCounter",
                "tool_args": {
                    "object_id": "mug_1",
                    "source_fixture_id": "cabinet_1",
                    "target_fixture_id": "counter_1",
                },
                "entity_refs": {
                    "object_id": "mug_1",
                    "source_fixture_id": "cabinet_1",
                    "target_fixture_id": "counter_1",
                },
                "reasoning": action_reasoning[0],
            },
            {
                "step_index": 3,
                "agent_id": action_agents[1],
                "tool_name": "CoffeeSetupMug",
                "tool_args": {
                    "object_id": "mug_1",
                    "source_fixture_id": "counter_1",
                    "target_fixture_id": "coffee_machine_1",
                },
                "entity_refs": {
                    "object_id": "mug_1",
                    "source_fixture_id": "counter_1",
                    "target_fixture_id": "coffee_machine_1",
                },
                "reasoning": action_reasoning[1],
            },
            {
                "step_index": 4,
                "agent_id": action_agents[2],
                "tool_name": "StartCoffeeMachine",
                "tool_args": {
                    "fixture_id": "coffee_machine_1",
                    "object_id": "mug_1",
                },
                "entity_refs": {
                    "fixture_id": "coffee_machine_1",
                    "object_id": "mug_1",
                },
                "reasoning": action_reasoning[2],
            },
        ],
    }


def make_invalid_candidate_missing_initial_communication():
    candidate = make_valid_candidate()
    candidate["steps"] = candidate["steps"][1:]
    for index, step in enumerate(candidate["steps"]):
        step["step_index"] = index
    return candidate


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


class AtomicToolCatalogTests(unittest.TestCase):
    def test_discover_atomic_tools_filters_helper_classes(self):
        tool_names = {tool.name for tool in discover_atomic_tools()}
        self.assertIn("PickPlaceCabinetToCounter", tool_names)
        self.assertIn("CoffeeSetupMug", tool_names)
        self.assertIn("StartCoffeeMachine", tool_names)
        self.assertNotIn("PickPlace", tool_names)
        self.assertNotIn("PickPlaceCoffee", tool_names)
        self.assertNotIn("ManipulateDoor", tool_names)
        self.assertNotIn("OpenDoor", tool_names)

    def test_prepare_coffee_prompt_contains_catalog_and_rules(self):
        prompt = build_prepare_coffee_prompt("unit-test")
        self.assertIn("PickPlaceCabinetToCounter", prompt)
        self.assertIn("CoffeeSetupMug", prompt)
        self.assertIn("StartCoffeeMachine", prompt)
        self.assertIn("communicate", prompt)
        self.assertIn("variation key: unit-test", prompt)
        self.assertIn("Each agent entry must contain only: agent_id.", prompt)
        self.assertNotIn(
            "Each agent entry must contain: agent_id, role, initial_plan.",
            prompt,
        )


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

    def test_parse_args_accepts_disable_validation(self):
        runtime_config = parse_args(["--disable-validation"])
        self.assertTrue(runtime_config.disable_validation)

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
                    "data_generation.task_level.client.validate_google_auth"
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
                    "data_generation.task_level.client.validate_google_auth"
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


class PrepareCoffeeValidatorTests(unittest.TestCase):
    def setUp(self):
        self.validator = PrepareCoffeeValidator()

    def test_validator_accepts_valid_trace(self):
        validation = self.validator.validate(make_valid_candidate())
        self.assertTrue(validation["is_valid"])
        self.assertTrue(validation["final_state"]["coffee_machine_started"])

    def test_validator_rejects_missing_initial_communication(self):
        candidate = make_valid_candidate()
        candidate["steps"].pop(1)
        candidate["steps"][1]["step_index"] = 1
        candidate["steps"][2]["step_index"] = 2
        candidate["steps"][3]["step_index"] = 3

        with self.assertRaises(TrajectoryValidationError):
            self.validator.validate(candidate)

    def test_validator_rejects_wrong_tool_order(self):
        candidate = make_valid_candidate()
        candidate["steps"][2]["tool_name"] = "CoffeeSetupMug"
        candidate["steps"][2]["tool_args"] = {
            "object_id": "mug_1",
            "source_fixture_id": "counter_1",
            "target_fixture_id": "coffee_machine_1",
        }
        candidate["steps"][2]["entity_refs"] = {
            "object_id": "mug_1",
            "source_fixture_id": "counter_1",
            "target_fixture_id": "coffee_machine_1",
        }

        with self.assertRaises(TrajectoryValidationError):
            self.validator.validate(candidate)

    def test_validator_rejects_broken_entity_continuity(self):
        candidate = make_valid_candidate()
        candidate["steps"][3]["tool_args"]["object_id"] = "mug_2"

        with self.assertRaises(TrajectoryValidationError):
            self.validator.validate(candidate)

    def test_validator_rejects_premature_coffee_machine_start(self):
        candidate = make_valid_candidate()
        candidate["steps"] = candidate["steps"][:3] + [candidate["steps"][4]]
        candidate["steps"][3]["step_index"] = 3

        with self.assertRaises(TrajectoryValidationError):
            self.validator.validate(candidate)

    def test_validator_rejects_missing_reasoning(self):
        candidate = make_valid_candidate()
        candidate["steps"][2]["reasoning"] = ""

        with self.assertRaises(TrajectoryValidationError):
            self.validator.validate(candidate)


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
                        "data_generation.task_level.trajectory_generation.ThreadPoolExecutor",
                        return_value=fake_executor,
                    ):
                        with mock.patch(
                            "data_generation.task_level.trajectory_generation.as_completed",
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
            250,
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
            payload["cost_summary"]["usage_sources"],
            ["api_usage_metadata"],
        )
        self.assertTrue(
            payload["cost_summary"][
                "all_trajectories_used_api_usage_metadata"
            ]
        )
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
            "done $0.0013",
        )
        trajectory_progress.refresh.assert_called_once()

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
            payload["cost_summary"]["usage_sources"],
            ["heuristic_4_chars_per_token"],
        )
        self.assertTrue(payload["cost_summary"]["pricing_supported"])
        self.assertNotIn("cost_estimate", payload)

    def test_preflight_cost_estimate_uses_historical_usage_mean_when_available(self):
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

        with tempfile.TemporaryDirectory() as tmpdir:
            task_dir = Path(tmpdir) / "prepare_coffee" / "20260310T000000Z"
            task_dir.mkdir(parents=True)
            historical_cost_path = (
                task_dir / "prepare_coffee_trajectories_costs.json"
            )
            historical_cost_path.write_text(
                json.dumps(
                    {
                        "composite_task": "PrepareCoffee",
                        "model": "gemini-3-flash-preview",
                        "trajectory_costs": [
                            {
                                "trajectory_id": "traj_000",
                                "generation_usage": {
                                    "successful_attempt_number": 1,
                                    "prompt_tokens": 3000,
                                    "output_tokens": 5000,
                                    "total_tokens": 8000,
                                    "usage_source": "api_usage_metadata",
                                    "traffic_type": "ON_DEMAND",
                                    "observed_cost_usd": 0.0165,
                                },
                            },
                            {
                                "trajectory_id": "traj_001",
                                "generation_usage": {
                                    "successful_attempt_number": 1,
                                    "prompt_tokens": 3200,
                                    "output_tokens": 7000,
                                    "total_tokens": 10200,
                                    "usage_source": "api_usage_metadata",
                                    "traffic_type": "ON_DEMAND",
                                    "observed_cost_usd": 0.0226,
                                },
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with mock.patch(
                "data_generation.task_level.trajectory_generation.DEFAULT_OUTPUT_DIR",
                Path(tmpdir),
            ):
                summary = _build_preflight_cost_estimate_summary(
                    runtime_config,
                    PREPARE_COFFEE_TASK,
                )

        self.assertEqual(
            summary["best_case_tokens"],
            {"prompt": 6200, "output": 12000, "total": 18200},
        )
        self.assertAlmostEqual(summary["best_case_total_usd"], 0.0391)
        self.assertAlmostEqual(summary["worst_case_total_usd"], 0.0782)
        self.assertIn("historical saved trajectories", summary["notes"][0])

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
        self.assertIn(
            "Both agents must coordinate via communication before the first task action.",
            trajectory["validation"]["error"],
        )
        self.assertEqual(len(trajectory["steps"]), 4)

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
        self.assertEqual(
            cost_payload["cost_summary"],
            payload["cost_summary"],
        )
        self.assertNotIn("cost_estimate", cost_payload)

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
        self.assertEqual(
            summary_payload["cost_summary"],
            payload["cost_summary"],
        )
        self.assertNotIn("cost_estimate", summary_payload)
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
                    "trajectory_directory": "trajectories",
                    "trajectory_files": [
                        {
                            "trajectory_id": "traj_000",
                            "path": "trajectories/traj_000.json",
                        }
                    ],
                },
            )
            self.assertEqual(
                cost_payload["trajectory_output_path"],
                str(output_path),
            )
            self.assertEqual(
                cost_payload["trajectory_directory"],
                str(trajectory_output_dir),
            )
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
