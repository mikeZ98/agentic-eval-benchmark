"""Runtime configuration, loaded from ``EVAL_*`` environment variables.

Unlike the schemas, settings are not ``strict``: environment values always arrive as
strings, so lax coercion is required. Bounds are enforced explicitly.
"""

from pathlib import Path
from typing import Annotated, Final, Self

from pydantic import Field, StringConstraints, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

OFFLINE_JUDGE_MODELS: Final = frozenset({"heuristic", "mock"})
"""``judge_model`` values served by the deterministic, network-free HeuristicJudge."""

ModelId = Annotated[
    str,
    StringConstraints(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/@-]*$"),
]


class BenchmarkSettings(BaseSettings):
    """Validated, immutable benchmark configuration."""

    model_config = SettingsConfigDict(
        env_prefix="EVAL_", extra="forbid", frozen=True, validate_default=True
    )

    concurrency_limit: int = Field(default=5, ge=1, le=256)
    """Maximum number of samples evaluated at the same time."""

    judge_model: ModelId = "heuristic"
    """Judge identifier: ``heuristic`` / ``mock`` (offline) or a provider model ID such
    as ``gpt-4o``. Provider-backed judges are wired up by the tools layer."""

    timeout_seconds: float = Field(default=30.0, gt=0.0, le=600.0)
    """Per-attempt deadline for one RAG query or one judge call."""

    retry_attempts: int = Field(default=3, ge=0, le=10)
    """Retries after the first attempt on transient errors (so ``3`` means 4 tries)."""

    retry_backoff_seconds: float = Field(default=0.5, ge=0.0, le=60.0)
    """Delay before the first retry; doubles on every further retry."""

    retry_backoff_max_seconds: float = Field(default=8.0, ge=0.0, le=300.0)
    """Upper bound on a single backoff delay."""

    output_dir: Path = Path(".cache/eval_runs")
    """Where run reports are persisted. ``.envrc`` pins it to ``./.cache/eval_runs`` on
    the project SSD; relative paths resolve against the working directory."""

    @field_validator("judge_model", mode="before")
    @classmethod
    def _lower_model(cls, value: object) -> object:
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("output_dir")
    @classmethod
    def _absolute_output_dir(cls, value: Path) -> Path:
        return value.expanduser().resolve()

    @model_validator(mode="after")
    def _check_backoff(self) -> Self:
        if self.retry_backoff_max_seconds < self.retry_backoff_seconds:
            raise ValueError("retry_backoff_max_seconds must be >= retry_backoff_seconds")
        return self

    @property
    def uses_offline_judge(self) -> bool:
        """Whether ``judge_model`` selects the deterministic offline judge."""
        return self.judge_model in OFFLINE_JUDGE_MODELS

    def backoff_delay(self, retry: int) -> float:
        """Delay before retry number ``retry`` (1-based): exponential, capped."""
        if retry < 1:
            raise ValueError("retry numbers start at 1")
        return min(self.retry_backoff_max_seconds, self.retry_backoff_seconds * 2.0 ** (retry - 1))

    def run_dir(self, run_id: str) -> Path:
        """Directory holding the artifacts of run ``run_id``."""
        return self.output_dir / run_id
