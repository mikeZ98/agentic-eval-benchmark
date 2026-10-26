"""Service-layer configuration: HTTP API limits and judge provider credentials.

Each class reads its own environment prefix, disjoint from ``BenchmarkSettings`` field
names, so the ``EVAL_*`` families never shadow each other:

- ``EVAL_API_*``      → :class:`ApiSettings`
- ``EVAL_PROVIDER_*`` → :class:`ProviderSettings`
- ``EVAL_RAG_*``      → :class:`~agentic_eval_benchmark.tools.RAGEndpointConfig`
"""

from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    AnyHttpUrl,
    Field,
    SecretStr,
    StringConstraints,
    field_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

ModelId = Annotated[
    str,
    StringConstraints(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/@-]*$"),
]
Hostname = Annotated[
    str, StringConstraints(min_length=1, max_length=253, pattern=r"^[A-Za-z0-9.:\[\]-]+$")
]


class ApiSettings(BaseSettings):
    """Limits and access control for the HTTP service."""

    model_config = SettingsConfigDict(
        env_prefix="EVAL_API_", extra="forbid", frozen=True, validate_default=True
    )

    api_key: SecretStr | None = None
    """When set, ``/v1/eval/run`` and ``/v1/eval/report/*`` require it as
    ``Authorization: Bearer <key>`` or ``X-API-Key``. Health stays open."""

    max_request_bytes: int = Field(default=16 * 1024 * 1024, ge=1024, le=256 * 1024 * 1024)
    """Largest accepted request body (inline Golden Datasets included)."""

    max_concurrent_runs: int = Field(default=1, ge=1, le=16)
    """Runs executing at once; further triggers are refused with 429."""

    dataset_dir: Path | None = None
    """Root for ``dataset_path`` references; ``None`` disables them (inline datasets and
    the built-in reference dataset still work). Paths cannot escape this root."""

    allowed_rag_hosts: tuple[Hostname, ...] = ()
    """Hosts a request may point ``rag_url`` at, besides the configured endpoint's host.
    Empty means only the configured host: the service never becomes an open proxy."""

    @field_validator("api_key")
    @classmethod
    def _key_strength(cls, value: SecretStr | None) -> SecretStr | None:
        if value is not None and len(value.get_secret_value()) < 32:  # noqa: PLR2004
            raise ValueError("api_key must be at least 32 characters")
        return value

    @field_validator("dataset_dir")
    @classmethod
    def _absolute_dataset_dir(cls, value: Path | None) -> Path | None:
        return None if value is None else value.expanduser().resolve()


class ProviderSettings(BaseSettings):
    """Model endpoint for the provider-backed judges (``ragas`` / ``trulens``).

    No model IDs are defaulted (they go stale) and no SDK reads ambient credentials:
    the key comes from ``EVAL_PROVIDER_API_KEY`` only. A ``base_url`` without a key
    targets a local OpenAI-compatible server (vLLM, Ollama, LM Studio).
    """

    model_config = SettingsConfigDict(
        env_prefix="EVAL_PROVIDER_", extra="forbid", frozen=True, validate_default=True
    )

    provider: Literal["openai", "anthropic"] = "openai"
    """``openai`` covers every OpenAI-compatible endpoint via ``base_url``."""

    llm_model: ModelId | None = None
    embedding_model: ModelId | None = None
    embeddings: Literal["provider", "hashing"] = "provider"
    """``hashing`` uses the offline HashingEmbeddings for answer relevancy; required with
    ``anthropic``, which has no embedding API."""

    base_url: AnyHttpUrl | None = None
    api_key: SecretStr | None = None
    request_timeout_seconds: float = Field(default=60.0, gt=0.0, le=600.0)
    fallback_to_heuristic: bool = True
    """Rescore metrics the provider judge fails on with the HeuristicJudge."""
