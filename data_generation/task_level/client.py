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
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOTENV_PATH = REPO_ROOT / ".env"
COST_DECIMAL_PLACES = 4
DEFAULT_TRAFFIC_TYPE = "ON_DEMAND"
BATCH_TRAFFIC_TYPE = "ON_DEMAND_FLEX"
HEURISTIC_CHARS_PER_TOKEN = 4
VERTEX_AI_PRICING_URL = "https://cloud.google.com/vertex-ai/generative-ai/pricing"
MODEL_TEXT_PRICING_USD_PER_MILLION = {
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
    }
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
    output_tokens: int
    total_tokens: int
    source: str
    traffic_type: str


@dataclass(frozen=True)
class PricingTier:
    model: str
    traffic_type: str
    input_usd_per_million_tokens: float
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
        response_schema: dict[str, Any],
        temperature: float,
    ) -> Any:
        raise NotImplementedError


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
        thoughts_tokens = coerce_int(getattr(usage, "thoughts_tokens", None)) or 0
        tool_use_prompt_tokens = (
            coerce_int(getattr(usage, "tool_use_prompt_tokens", None)) or 0
        )
        total_tokens = coerce_int(getattr(usage, "total_tokens", None))
        output_tokens = candidate_tokens + thoughts_tokens
        if output_tokens <= 0 and total_tokens is not None:
            output_tokens = max(total_tokens - prompt_tokens - tool_use_prompt_tokens, 0)
        return AttemptUsage(
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            total_tokens=(
                total_tokens
                if total_tokens is not None
                else prompt_tokens + output_tokens + tool_use_prompt_tokens
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
        output_tokens=estimated_output_tokens,
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
        usage.prompt_tokens / 1_000_000
    ) * pricing.input_usd_per_million_tokens
    output_cost = (
        usage.output_tokens / 1_000_000
    ) * pricing.output_usd_per_million_tokens
    return AttemptCostBreakdown(
        usage=usage,
        pricing=pricing,
        input_cost_usd=input_cost,
        output_cost_usd=output_cost,
        total_cost_usd=input_cost + output_cost,
    )


def _serialize_generation_usage(
    *,
    attempt_number: int,
    cost_breakdown: AttemptCostBreakdown,
) -> dict[str, Any]:
    payload = {
        "successful_attempt_number": attempt_number,
        "prompt_tokens": cost_breakdown.usage.prompt_tokens,
        "output_tokens": cost_breakdown.usage.output_tokens,
        "total_tokens": cost_breakdown.usage.total_tokens,
        "usage_source": cost_breakdown.usage.source,
        "traffic_type": cost_breakdown.usage.traffic_type,
        "observed_cost_usd": round_cost(
            cost_breakdown.total_cost_usd,
            decimal_places=COST_DECIMAL_PLACES,
        ),
    }
    if cost_breakdown.pricing is not None:
        payload["pricing"] = {
            "model": cost_breakdown.pricing.model,
            "input_usd_per_million_tokens": cost_breakdown.pricing.input_usd_per_million_tokens,
            "output_usd_per_million_tokens": cost_breakdown.pricing.output_usd_per_million_tokens,
        }
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
    pricing = _resolve_pricing_tier(model, traffic_type)
    if pricing is None:
        updated_usage.pop("pricing", None)
        updated_usage["observed_cost_usd"] = None
        return updated_usage

    prompt_tokens = coerce_int(updated_usage.get("prompt_tokens")) or 0
    output_tokens = coerce_int(updated_usage.get("output_tokens")) or 0
    total_cost = (
        (prompt_tokens / 1_000_000) * pricing.input_usd_per_million_tokens
        + (output_tokens / 1_000_000) * pricing.output_usd_per_million_tokens
    )
    updated_usage["pricing"] = {
        "model": pricing.model,
        "input_usd_per_million_tokens": pricing.input_usd_per_million_tokens,
        "output_usd_per_million_tokens": pricing.output_usd_per_million_tokens,
    }
    updated_usage["observed_cost_usd"] = (
        round_cost(
            total_cost,
            decimal_places=COST_DECIMAL_PLACES,
        )
        if round_observed_cost
        else total_cost
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


def build_raw_google_genai_client(project: str | None, location: str) -> Any:
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
    return genai.Client(
        http_options=HttpOptions(api_version="v1"),
    )


class GoogleGenAIClient(BaseGenerationClient):
    def __init__(self, project: str | None, location: str):
        self._client = build_raw_google_genai_client(project=project, location=location)

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        response_schema: dict[str, Any],
        temperature: float,
    ) -> Any:
        try:
            response = self._client.models.generate_content(
                model=model,
                contents=prompt,
                config={
                    "temperature": temperature,
                    "response_mime_type": "application/json",
                    "response_schema": response_schema,
                },
            )
        except Exception as exc:
            message = str(exc)
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
            raise
        # Usage metadata is optional and field names vary a bit across SDK releases.
        usage = build_generation_usage_metadata(getattr(response, "usage_metadata", None))
        return GenerationResult(
            payload=getattr(response, "text", None) or str(response),
            usage=usage,
        )


def build_generation_client(
    sdk: str,
    project: str | None,
    location: str,
) -> BaseGenerationClient:
    if sdk == DEFAULT_SDK:
        return GoogleGenAIClient(project=project, location=location)
    raise TrajectoryGenerationError(
        f"Unsupported SDK '{sdk}'. Expected one of: {DEFAULT_SDK}."
    )
