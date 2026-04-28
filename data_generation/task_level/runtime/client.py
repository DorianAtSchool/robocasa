"""Provide generation clients, usage accounting, and pricing utilities."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data_generation.utils import (
    coerce_int,
    estimate_text_tokens,
    normalize_traffic_type,
    parse_dotenv_value,
    round_cost,
    usage_field,
)

DEFAULT_MODEL = "gemini-3-flash-preview"
DEFAULT_LOCATION = "global"
DEFAULT_SDK = "google-genai"
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DOTENV_PATH = REPO_ROOT / ".env"
COST_DECIMAL_PLACES = 4
DEFAULT_TRAFFIC_TYPE = "ON_DEMAND"
BATCH_TRAFFIC_TYPE = "ON_DEMAND_FLEX"
HEURISTIC_CHARS_PER_TOKEN = 4
CACHED_INPUT_TOKEN_DISCOUNT = 0.1
# Vertex AI text pricing is keyed by normalized traffic tier for cost estimation.
MODEL_TEXT_PRICING_USD_PER_MILLION = {
    "gemini-3.1-flash-lite-preview": {
        "ON_DEMAND": {
            "input": 0.25,
            "output": 1.50,
        },
        "ON_DEMAND_PRIORITY": {
            "input": 0.45,
            "output": 2.70,
        },
        "ON_DEMAND_FLEX": {
            "input": 0.13,
            "output": 0.75,
        },
    },
    "gemini-3-flash-preview": {
        "ON_DEMAND": {
            "input": 0.50,
            "output": 3.00,
        },
        "ON_DEMAND_PRIORITY": {
            "input": 0.90,
            "output": 5.40,
        },
        "ON_DEMAND_FLEX": {
            "input": 0.25,
            "output": 1.50,
        },
    },
}
# google-genai reads Vertex routing from environment variables.
GOOGLE_GENAI_VERTEX_ENV_VAR = "GOOGLE_GENAI_USE_VERTEXAI"


class TrajectoryGenerationError(RuntimeError):
    pass


@dataclass(frozen=True)
class GenerationUsage:
    prompt_tokens: int | None = None
    candidates_tokens: int | None = None
    thoughts_tokens: int | None = None
    cached_content_tokens: int | None = None
    tool_use_prompt_tokens: int | None = None
    total_tokens: int | None = None
    traffic_type: str | None = None


@dataclass(frozen=True)
class GenerationResult:
    payload: Any
    usage: GenerationUsage | None = None


@dataclass(frozen=True)
class AttemptUsage:
    prompt_tokens: int
    cached_input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    total_tokens: int
    source: str
    traffic_type: str

    def non_cached_input_tokens(self) -> int:
        """Return the prompt tokens that should bill at the standard input rate."""

        return max(self.prompt_tokens - self.cached_input_tokens, 0)

    def billable_output_tokens(self) -> int:
        """Return all tokens billed at the model output rate."""
        return self.output_tokens + self.reasoning_tokens


@dataclass(frozen=True)
class PricingTier:
    model: str
    traffic_type: str
    input_usd_per_million_tokens: float
    cached_input_usd_per_million_tokens: float
    output_usd_per_million_tokens: float


@dataclass(frozen=True)
class AttemptCostBreakdown:
    usage: AttemptUsage
    pricing: PricingTier | None
    input_cost_usd: float | None
    output_cost_usd: float | None
    total_cost_usd: float | None


class BaseGenerationClient:
    def generate(
        self,
        *,
        model: str,
        prompt: str,
        response_schema: dict[str, Any] | None,
        temperature: float,
        thinking_level: str | None = None,
        thinking_budget: int | None = None,
    ) -> Any:
        raise NotImplementedError


def _generation_error_status_code(exc: Exception) -> int | None:
    """Extracts one HTTP status code from SDK exceptions with varying shapes."""

    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code

    response = getattr(exc, "response", None)
    response_status_code = getattr(response, "status_code", None)
    if isinstance(response_status_code, int):
        return response_status_code

    return None


def _verbalized_response_count(
    response_schema: dict[str, Any] | None,
) -> int | None:
    """Returns the verbalized response count encoded in one response schema."""

    if response_schema is None:
        return None
    responses_schema = response_schema.get("properties", {}).get("responses")
    if not isinstance(responses_schema, dict):
        return None

    min_items = responses_schema.get("minItems")
    max_items = responses_schema.get("maxItems")
    if isinstance(min_items, int) and min_items == max_items:
        return min_items

    return None


def _build_attempt_usage(
    *,
    prompt: str,
    candidate: dict[str, Any],
    usage: GenerationUsage | None,
    default_traffic_type: str = DEFAULT_TRAFFIC_TYPE,
) -> AttemptUsage:
    prompt_tokens = coerce_int(getattr(usage, "prompt_tokens", None))
    if prompt_tokens is not None:
        candidate_tokens = coerce_int(getattr(usage, "candidates_tokens", None)) or 0
        reasoning_tokens = coerce_int(getattr(usage, "thoughts_tokens", None)) or 0
        cached_input_tokens = min(
            coerce_int(getattr(usage, "cached_content_tokens", None)) or 0,
            prompt_tokens,
        )
        tool_use_prompt_tokens = (
            coerce_int(getattr(usage, "tool_use_prompt_tokens", None)) or 0
        )
        total_tokens = coerce_int(getattr(usage, "total_tokens", None))
        output_tokens = candidate_tokens
        # Some SDK responses omit candidate token counts, so derive them from totals.
        if output_tokens <= 0 and total_tokens is not None:
            output_tokens = max(
                total_tokens
                - prompt_tokens
                - tool_use_prompt_tokens
                - reasoning_tokens,
                0,
            )
        return AttemptUsage(
            prompt_tokens=prompt_tokens,
            cached_input_tokens=cached_input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            total_tokens=(
                total_tokens
                if total_tokens is not None
                else (
                    prompt_tokens
                    + output_tokens
                    + reasoning_tokens
                    + tool_use_prompt_tokens
                )
            ),
            source="api_usage_metadata",
            traffic_type=getattr(usage, "traffic_type", None) or default_traffic_type,
        )

    output_text = json.dumps(candidate, sort_keys=True, separators=(",", ":"))
    estimated_prompt_tokens = estimate_text_tokens(
        prompt,
        chars_per_token=HEURISTIC_CHARS_PER_TOKEN,
    )
    estimated_output_tokens = estimate_text_tokens(
        output_text,
        chars_per_token=HEURISTIC_CHARS_PER_TOKEN,
    )
    return AttemptUsage(
        prompt_tokens=estimated_prompt_tokens,
        cached_input_tokens=0,
        output_tokens=estimated_output_tokens,
        reasoning_tokens=0,
        total_tokens=estimated_prompt_tokens + estimated_output_tokens,
        source=f"heuristic_{HEURISTIC_CHARS_PER_TOKEN}_chars_per_token",
        traffic_type=default_traffic_type,
    )


def _normalize_pricing_model(model: str) -> str | None:
    for pricing_model in MODEL_TEXT_PRICING_USD_PER_MILLION:
        if model == pricing_model or model.startswith(f"{pricing_model}-"):
            return pricing_model
    return None


def _resolve_pricing_tier(model: str, traffic_type: str | None) -> PricingTier | None:
    pricing_model = _normalize_pricing_model(model)
    if pricing_model is None:
        return None

    normalized_traffic_type = (traffic_type or DEFAULT_TRAFFIC_TYPE).upper()
    model_pricing = MODEL_TEXT_PRICING_USD_PER_MILLION[pricing_model]
    if normalized_traffic_type not in model_pricing:
        normalized_traffic_type = DEFAULT_TRAFFIC_TYPE
    tier_pricing = model_pricing[normalized_traffic_type]
    return PricingTier(
        model=pricing_model,
        traffic_type=normalized_traffic_type,
        input_usd_per_million_tokens=tier_pricing["input"],
        cached_input_usd_per_million_tokens=round(
            tier_pricing["input"] * CACHED_INPUT_TOKEN_DISCOUNT,
            6,
        ),
        output_usd_per_million_tokens=tier_pricing["output"],
    )


def _build_attempt_cost_breakdown(
    *,
    usage: AttemptUsage,
    model: str,
) -> AttemptCostBreakdown:
    pricing = _resolve_pricing_tier(model, usage.traffic_type)
    if pricing is None:
        return AttemptCostBreakdown(
            usage=usage,
            pricing=None,
            input_cost_usd=None,
            output_cost_usd=None,
            total_cost_usd=None,
        )

    input_cost = (
        (usage.non_cached_input_tokens() / 1_000_000)
        * pricing.input_usd_per_million_tokens
    ) + (
        (usage.cached_input_tokens / 1_000_000)
        * pricing.cached_input_usd_per_million_tokens
    )
    # Vertex bills reasoning tokens at the same rate as other output tokens.
    output_cost = (
        usage.billable_output_tokens() / 1_000_000
    ) * pricing.output_usd_per_million_tokens
    return AttemptCostBreakdown(
        usage=usage,
        pricing=pricing,
        input_cost_usd=input_cost,
        output_cost_usd=output_cost,
        total_cost_usd=input_cost + output_cost,
    )


def _serialize_pricing_tier(pricing: PricingTier) -> dict[str, Any]:
    """Converts a pricing tier into the persisted JSON payload shape."""

    return {
        "model": pricing.model,
        "input_usd_per_million_tokens": pricing.input_usd_per_million_tokens,
        "cached_input_usd_per_million_tokens": (
            pricing.cached_input_usd_per_million_tokens
        ),
        "output_usd_per_million_tokens": pricing.output_usd_per_million_tokens,
    }


def _build_attempt_usage_from_generation_usage(
    generation_usage: dict[str, Any],
    *,
    traffic_type: str,
) -> AttemptUsage:
    """Builds attempt usage from persisted token counts before repricing."""

    prompt_tokens = coerce_int(generation_usage.get("prompt_tokens")) or 0
    cached_input_tokens = min(
        coerce_int(generation_usage.get("cached_input_tokens")) or 0,
        prompt_tokens,
    )
    output_tokens = coerce_int(generation_usage.get("output_tokens")) or 0
    reasoning_tokens = coerce_int(generation_usage.get("reasoning_tokens")) or 0
    total_tokens = coerce_int(generation_usage.get("total_tokens"))
    if total_tokens is None:
        total_tokens = prompt_tokens + output_tokens + reasoning_tokens

    usage_source = generation_usage.get("usage_source")
    if not isinstance(usage_source, str) or not usage_source:
        usage_source = "persisted_generation_usage"

    return AttemptUsage(
        prompt_tokens=prompt_tokens,
        cached_input_tokens=cached_input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
        total_tokens=total_tokens,
        source=usage_source,
        traffic_type=traffic_type,
    )


def _serialize_generation_usage(
    *,
    attempt_number: int,
    cost_breakdown: AttemptCostBreakdown,
) -> dict[str, Any]:
    payload = {
        "successful_attempt_number": attempt_number,
        "prompt_tokens": cost_breakdown.usage.prompt_tokens,
        "cached_input_tokens": cost_breakdown.usage.cached_input_tokens,
        "output_tokens": cost_breakdown.usage.output_tokens,
        "reasoning_tokens": cost_breakdown.usage.reasoning_tokens,
        "total_tokens": cost_breakdown.usage.total_tokens,
        "usage_source": cost_breakdown.usage.source,
        "traffic_type": cost_breakdown.usage.traffic_type,
        "observed_cost_usd": round_cost(
            cost_breakdown.total_cost_usd,
            decimal_places=COST_DECIMAL_PLACES,
        ),
    }
    if cost_breakdown.pricing is not None:
        payload["pricing"] = _serialize_pricing_tier(cost_breakdown.pricing)
    return payload


def build_generation_usage(
    *,
    model: str,
    prompt: str,
    candidate: dict[str, Any],
    usage: GenerationUsage | None,
    attempt_number: int,
    default_traffic_type: str = DEFAULT_TRAFFIC_TYPE,
) -> dict[str, Any]:
    return _serialize_generation_usage(
        attempt_number=attempt_number,
        cost_breakdown=_build_attempt_cost_breakdown(
            usage=_build_attempt_usage(
                prompt=prompt,
                candidate=candidate,
                usage=usage,
                default_traffic_type=default_traffic_type,
            ),
            model=model,
        ),
    )


def reprice_generation_usage(
    generation_usage: dict[str, Any],
    *,
    model: str,
    traffic_type: str,
    round_observed_cost: bool = True,
) -> dict[str, Any]:
    updated_usage = dict(generation_usage)
    updated_usage["traffic_type"] = traffic_type
    cost_breakdown = _build_attempt_cost_breakdown(
        usage=_build_attempt_usage_from_generation_usage(
            updated_usage,
            traffic_type=traffic_type,
        ),
        model=model,
    )
    if cost_breakdown.pricing is None:
        updated_usage.pop("pricing", None)
        updated_usage["observed_cost_usd"] = None
        return updated_usage

    updated_usage["pricing"] = _serialize_pricing_tier(cost_breakdown.pricing)
    updated_usage["observed_cost_usd"] = (
        round_cost(
            cost_breakdown.total_cost_usd,
            decimal_places=COST_DECIMAL_PLACES,
        )
        if round_observed_cost
        else cost_breakdown.total_cost_usd
    )
    return updated_usage


def load_dotenv_file(
    path: Path | None = None,
    *,
    override: bool = False,
) -> dict[str, str]:
    dotenv_path = path or DEFAULT_DOTENV_PATH
    loaded_values: dict[str, str] = {}
    if not dotenv_path.exists():
        return loaded_values

    # Mirror common dotenv parsing without adding another runtime dependency.
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue

        key, raw_value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = parse_dotenv_value(raw_value)
        loaded_values[key] = value
        if override or key not in os.environ:
            os.environ[key] = value

    return loaded_values


def configure_google_genai_environment(
    *,
    project: str | None,
    location: str | None,
) -> None:
    os.environ[GOOGLE_GENAI_VERTEX_ENV_VAR] = "True"
    if project:
        os.environ["GOOGLE_CLOUD_PROJECT"] = project
    if location:
        os.environ["GOOGLE_CLOUD_LOCATION"] = location


def validate_google_auth(project: str | None) -> None:
    configured_project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
    try:
        import google.auth
        from google.auth.exceptions import DefaultCredentialsError
    except ImportError:
        # Keep local tests working even when google-auth is not installed.
        return

    try:
        _, detected_project = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
    except DefaultCredentialsError as exc:
        raise TrajectoryGenerationError(
            "Google ADC credentials were not found. Use either "
            "`gcloud auth application-default login` or set "
            "`GOOGLE_APPLICATION_CREDENTIALS` in your environment or .env."
        ) from exc

    if not configured_project and not detected_project:
        raise TrajectoryGenerationError(
            "Google Cloud project was not found. Set --project or "
            "GOOGLE_CLOUD_PROJECT."
        )


def build_generation_usage_metadata(
    usage_metadata: Any,
    *,
    default_traffic_type: str = DEFAULT_TRAFFIC_TYPE,
) -> GenerationUsage | None:
    if usage_metadata is None:
        return None
    return GenerationUsage(
        prompt_tokens=usage_field(
            usage_metadata, "prompt_token_count", "promptTokenCount"
        ),
        candidates_tokens=usage_field(
            usage_metadata,
            "candidates_token_count",
            "candidatesTokenCount",
        ),
        thoughts_tokens=usage_field(
            usage_metadata, "thoughts_token_count", "thoughtsTokenCount"
        ),
        cached_content_tokens=usage_field(
            usage_metadata,
            "cached_content_token_count",
            "cachedContentTokenCount",
        ),
        tool_use_prompt_tokens=usage_field(
            usage_metadata,
            "tool_use_prompt_token_count",
            "toolUsePromptTokenCount",
        ),
        total_tokens=usage_field(
            usage_metadata, "total_token_count", "totalTokenCount"
        ),
        traffic_type=(
            normalize_traffic_type(
                usage_field(usage_metadata, "traffic_type", "trafficType")
            )
            or default_traffic_type
        ),
    )


DEFAULT_GENERATION_TIMEOUT_SEC = 300
"""Default per-request wall-clock cap for SDK calls.

Without this, a stalled HTTP request hangs the worker forever (silent TCP
drops, server-side hangs, idle timeouts at intermediate proxies). Five
minutes is generous enough for the largest trajectory generations we have
observed and short enough that retries can recover within `--max-retries`.
"""


def build_raw_google_genai_client(
    project: str | None,
    location: str,
    *,
    timeout_sec: int | None = DEFAULT_GENERATION_TIMEOUT_SEC,
) -> Any:
    try:
        from google import genai
        from google.genai.types import HttpOptions
    except ImportError as exc:
        raise TrajectoryGenerationError(
            "google-genai is not installed. Install it with "
            "`uv pip install google-genai`."
        ) from exc

    configure_google_genai_environment(project=project, location=location)
    validate_google_auth(project)
    http_options_kwargs: dict[str, Any] = {"api_version": "v1"}
    if timeout_sec is not None:
        # HttpOptions.timeout is in milliseconds.
        http_options_kwargs["timeout"] = int(timeout_sec) * 1000
    return genai.Client(
        http_options=HttpOptions(**http_options_kwargs),
    )


DEFAULT_GOOGLE_GENAI_MAX_OUTPUT_TOKENS = 32768


class GoogleGenAIClient(BaseGenerationClient):
    def __init__(
        self,
        project: str | None,
        location: str,
        *,
        timeout_sec: int | None = DEFAULT_GENERATION_TIMEOUT_SEC,
    ):
        self._client = build_raw_google_genai_client(
            project=project,
            location=location,
            timeout_sec=timeout_sec,
        )

    def _build_generation_config(
        self,
        *,
        temperature: float,
        response_schema: dict[str, Any] | None,
        thinking_level: str | None,
        thinking_budget: int | None,
    ) -> dict[str, Any]:
        """Builds one google-genai generation config payload."""

        config: dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": DEFAULT_GOOGLE_GENAI_MAX_OUTPUT_TOKENS,
            "response_mime_type": "application/json",
        }
        # Pass `response_schema` only when the caller supplied one. Phase 1
        # (TaskSpec generation) omits the schema because the spec uses dicts
        # with dynamic string keys that Gemini's structured-output mode
        # cannot express — keeping the schema would force Gemini to return
        # `{}` for any unenumerated object.
        if response_schema is not None:
            config["response_schema"] = response_schema
        thinking_config: dict[str, Any] = {}
        if thinking_level is not None:
            thinking_config["thinking_level"] = thinking_level
        if thinking_budget is not None:
            thinking_config["thinking_budget"] = thinking_budget
        if thinking_config:
            config["thinking_config"] = thinking_config
        return config

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        response_schema: dict[str, Any] | None,
        temperature: float,
        thinking_level: str | None = None,
        thinking_budget: int | None = None,
    ) -> Any:
        try:
            response = self._client.models.generate_content(
                model=model,
                contents=prompt,
                config=self._build_generation_config(
                    temperature=temperature,
                    response_schema=response_schema,
                    thinking_level=thinking_level,
                    thinking_budget=thinking_budget,
                ),
            )
        except Exception as exc:
            message = str(exc)
            status_code = _generation_error_status_code(exc)
            if (
                "403 PERMISSION_DENIED" in message
                and "aiplatform.endpoints.predict" in message
            ):
                raise TrajectoryGenerationError(
                    "The configured Google principal does not have permission to call "
                    "Vertex AI `GenerateContent`. Grant a role that includes "
                    "`aiplatform.endpoints.predict` on project "
                    f"`{os.environ.get('GOOGLE_CLOUD_PROJECT', '<unknown-project>')}` "
                    "such as Vertex AI User, then retry."
                ) from exc
            if status_code == 400 and "INVALID_ARGUMENT" in message:
                verbalized_response_count = _verbalized_response_count(response_schema)
                verbalized_hint = ""
                if verbalized_response_count is not None:
                    verbalized_hint = (
                        " This request uses verbalized structured output, and the "
                        f"requested response count ({verbalized_response_count}) may exceed "
                        "Vertex AI request limits. Retry with a smaller "
                        "`--verbalized-k`, such as 4 or lower."
                    )
                raise TrajectoryGenerationError(
                    "Vertex AI rejected the generation request with "
                    "`400 INVALID_ARGUMENT`."
                    f"{verbalized_hint} Original error: {message}"
                ) from exc
            raise
        # Usage metadata is optional and field names vary a bit across SDK releases.
        usage = build_generation_usage_metadata(
            getattr(response, "usage_metadata", None)
        )
        return GenerationResult(
            payload=getattr(response, "text", None) or str(response),
            usage=usage,
        )


def build_generation_client(
    sdk: str,
    project: str | None,
    location: str,
    *,
    timeout_sec: int | None = DEFAULT_GENERATION_TIMEOUT_SEC,
) -> BaseGenerationClient:
    if sdk == DEFAULT_SDK:
        return GoogleGenAIClient(
            project=project,
            location=location,
            timeout_sec=timeout_sec,
        )
    raise TrajectoryGenerationError(
        f"Unsupported SDK '{sdk}'. Expected one of: {DEFAULT_SDK}."
    )
