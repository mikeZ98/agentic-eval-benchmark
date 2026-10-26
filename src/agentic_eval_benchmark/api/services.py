"""Service layer shared by the CLI and the HTTP API.

:class:`EvaluationServices` resolves datasets, builds judges and RAG clients from
settings, executes runs and persists them in a :class:`RunStore`. Failures surface as
:class:`ServiceError` subclasses carrying a client-safe message, an HTTP status and a
CLI exit code, so both front ends report them the same way.
"""

import asyncio
import importlib.util
import os
import re
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Final
from urllib.parse import urlsplit

from pydantic import AnyHttpUrl, TypeAdapter, ValidationError

from agentic_eval_benchmark.api.settings import ApiSettings, ProviderSettings
from agentic_eval_benchmark.core import (
    OFFLINE_JUDGE_MODELS,
    BenchmarkSettings,
    EvaluationEngine,
    EvaluationRun,
    HeuristicJudge,
    JudgeBackend,
    JudgeProtocol,
    RAGClientProtocol,
    write_run,
)
from agentic_eval_benchmark.core.engine import REPORT_FILE
from agentic_eval_benchmark.schemas import (
    BenchmarkReport,
    BenchmarkRunSummary,
    GoldenDataset,
    ThresholdPolicy,
)
from agentic_eval_benchmark.tools._isolation import cache_root, trulens_home
from agentic_eval_benchmark.tools.dataset_loader import (
    DatasetLoadError,
    load_golden_dataset,
    load_reference_dataset,
)
from agentic_eval_benchmark.tools.rag_client import (
    HttpRAGClient,
    RAGEndpointConfig,
    StaticRAGClient,
)

if TYPE_CHECKING:
    from openai import AsyncOpenAI

_RUN_ID_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_URL: Final = TypeAdapter(AnyHttpUrl)
_LOCAL_PLACEHOLDER_KEY: Final = "not-required"  # local OpenAI-compatible servers ignore it


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ServiceError(Exception):
    """A failure with a client-safe message; never carries secrets or dataset content."""

    status: ClassVar[int] = 500
    kind: ClassVar[str] = "internal-error"
    title: ClassVar[str] = "Internal Server Error"
    exit_code: ClassVar[int] = 2

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class InvalidInputError(ServiceError):
    status, kind, title = 422, "invalid-input", "Unprocessable Content"


class NotFoundError(ServiceError):
    status, kind, title = 404, "not-found", "Not Found"


class ConflictError(ServiceError):
    status, kind, title = 409, "conflict", "Conflict"


class ForbiddenError(ServiceError):
    status, kind, title = 403, "forbidden", "Forbidden"


class BusyError(ServiceError):
    status, kind, title = 429, "busy", "Too Many Requests"


class ConfigurationError(ServiceError):
    """Server-side misconfiguration (missing model, key or optional dependency)."""

    status, kind, title = 503, "misconfigured", "Service Unavailable"


class StorageError(ServiceError):
    status, kind, title = 500, "storage-error", "Internal Server Error"


# ---------------------------------------------------------------------------
# Run storage
# ---------------------------------------------------------------------------


class RunStore:
    """Persisted runs under ``root/<run_id>/``, as written by ``core.write_run``."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def run_dir(self, run_id: str) -> Path:
        """Directory of ``run_id``; the ID is validated so it can never escape ``root``."""
        if not _RUN_ID_RE.fullmatch(run_id):
            raise InvalidInputError("run_id must match [A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")
        return self.root / run_id

    def exists(self, run_id: str) -> bool:
        return (self.run_dir(run_id) / REPORT_FILE).is_file()

    def save(self, run: EvaluationRun) -> Path:
        try:
            return write_run(run, self.root)
        except FileExistsError:
            raise ConflictError(f"run {run.summary.run_id} already exists") from None
        except OSError as exc:
            raise StorageError(f"cannot persist the run ({type(exc).__name__})") from exc

    def load(self, run_id: str) -> BenchmarkReport:
        return self.load_path(self.run_dir(run_id), label=f"run {run_id}")

    @staticmethod
    def load_path(path: Path, *, label: str | None = None) -> BenchmarkReport:
        """Load a run directory or a ``report.json`` file."""
        file = path / REPORT_FILE if path.is_dir() else path
        name = label or path.name
        if not file.is_file():
            raise NotFoundError(f"{name} not found")
        try:
            return BenchmarkReport.from_json(file.read_bytes())
        except ValidationError as exc:
            raise StorageError(
                f"{name} is not a valid report ({exc.error_count()} errors)"
            ) from exc
        except OSError as exc:
            raise StorageError(f"{name} is unreadable ({type(exc).__name__})") from exc

    def list_runs(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(p.name for p in self.root.iterdir() if (p / REPORT_FILE).is_file())

    def writable(self) -> bool:
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except OSError:
            return False
        return os.access(self.root, os.W_OK)


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

type EngineFactory = Callable[
    [RAGClientProtocol, JudgeProtocol, BenchmarkSettings, ThresholdPolicy], EvaluationEngine
]


def default_engine_factory(
    rag: RAGClientProtocol,
    judge: JudgeProtocol,
    settings: BenchmarkSettings,
    policy: ThresholdPolicy,
) -> EvaluationEngine:
    return EvaluationEngine(rag, judge, settings=settings, policy=policy)


@dataclass(frozen=True, slots=True)
class RunPlan:
    """Everything one run needs, already validated."""

    dataset: GoldenDataset
    policy: ThresholdPolicy
    judge: JudgeBackend
    rag_url: str | None = None
    oracle: bool = False
    baseline_run_id: str | None = None
    baseline: BenchmarkRunSummary | None = None
    """A baseline loaded elsewhere (e.g. a report path); takes precedence over the ID."""
    run_id: str | None = None


@dataclass(frozen=True, slots=True)
class JudgeStatus:
    """Whether a judge backend can be built in this process and configuration."""

    available: bool
    configured: bool
    detail: str

    def as_dict(self) -> dict[str, object]:
        return {"available": self.available, "configured": self.configured, "detail": self.detail}


class EvaluationServices:
    """Settings, factories and storage behind both the CLI and the API."""

    def __init__(
        self,
        settings: BenchmarkSettings | None = None,
        *,
        api: ApiSettings | None = None,
        provider: ProviderSettings | None = None,
        rag: RAGEndpointConfig | None = None,
        engine_factory: EngineFactory = default_engine_factory,
    ) -> None:
        self.settings = settings or BenchmarkSettings()
        self.api = api or ApiSettings()
        self.provider = provider or ProviderSettings()
        self.rag = rag or RAGEndpointConfig()
        self.engine_factory = engine_factory
        self.store = RunStore(self.settings.output_dir)
        self.run_slots = asyncio.Semaphore(self.api.max_concurrent_runs)
        self._openai: AsyncOpenAI | None = None
        self._anthropic: Any = None

    async def aclose(self) -> None:
        """Close provider clients created by the judge factory."""
        if self._openai is not None:
            await self._openai.close()
            self._openai = None
        if self._anthropic is not None:
            await self._anthropic.close()
            self._anthropic = None

    # -- datasets -----------------------------------------------------------

    @staticmethod
    def load_dataset_file(path: Path) -> GoldenDataset:
        try:
            return load_golden_dataset(path)
        except DatasetLoadError as exc:
            raise InvalidInputError(str(exc)) from None

    def resolve_api_dataset_path(self, relative: str) -> Path:
        """Resolve an API ``dataset_path`` strictly inside ``ApiSettings.dataset_dir``."""
        root = self.api.dataset_dir
        if root is None:
            raise ForbiddenError("dataset_path references are disabled on this server")
        candidate = Path(relative)
        if candidate.is_absolute() or not relative.strip():
            raise InvalidInputError("dataset_path must be relative to the dataset directory")
        resolved = (root / candidate).resolve()
        if not resolved.is_relative_to(root) or not resolved.is_file():
            raise NotFoundError("dataset not found")
        return resolved

    @staticmethod
    def reference_dataset() -> GoldenDataset:
        return load_reference_dataset()

    # -- RAG targets ----------------------------------------------------------

    def check_rag_override(self, url: str) -> str:
        """Validate an API ``rag_url`` override against the host allowlist."""
        try:
            parsed = _URL.validate_python(url)
        except ValidationError:
            raise InvalidInputError("rag_url must be an absolute http(s) URL") from None
        configured = urlsplit(str(self.rag.url)).hostname
        allowed = {h.lower() for h in self.api.allowed_rag_hosts} | {configured or ""}
        if (parsed.host or "").lower() not in allowed:
            raise ForbiddenError("rag_url host is not in EVAL_API_ALLOWED_RAG_HOSTS")
        return str(parsed)

    def rag_client(
        self, *, dataset: GoldenDataset, url: str | None, oracle: bool
    ) -> AbstractAsyncContextManager[RAGClientProtocol]:
        if oracle:
            return nullcontext(StaticRAGClient.from_dataset(dataset))
        config = self.rag
        if url is not None:
            try:
                config = config.model_copy(update={"url": _URL.validate_python(url)})
            except ValidationError:
                raise InvalidInputError("rag_url must be an absolute http(s) URL") from None
        return HttpRAGClient(config)

    # -- judges ---------------------------------------------------------------

    def judge_status(self, backend: JudgeBackend) -> JudgeStatus:
        if backend in OFFLINE_JUDGE_MODELS:
            return JudgeStatus(
                available=True, configured=True, detail="deterministic offline judge"
            )
        module = "ragas" if backend == "ragas" else "trulens.providers.openai"
        available = importlib.util.find_spec(module.split(".", maxsplit=1)[0]) is not None
        try:
            self._require_provider(backend)
        except ConfigurationError as exc:
            return JudgeStatus(available=available, configured=False, detail=exc.message)
        detail = f"{self.provider.provider}:{self.provider.llm_model}"
        return JudgeStatus(available=available, configured=True, detail=detail)

    def _require_provider(self, backend: JudgeBackend) -> str:
        provider = self.provider
        if provider.llm_model is None:
            raise ConfigurationError(f"the {backend} judge needs EVAL_PROVIDER_LLM_MODEL")
        if provider.api_key is None and provider.base_url is None:
            raise ConfigurationError(
                f"the {backend} judge needs EVAL_PROVIDER_API_KEY "
                "(or EVAL_PROVIDER_BASE_URL for a local server)"
            )
        if backend == "trulens" and provider.provider != "openai":
            raise ConfigurationError("the trulens judge supports OpenAI-compatible providers only")
        if backend == "ragas" and provider.embeddings == "provider":
            if provider.provider != "openai":
                raise ConfigurationError(
                    "anthropic has no embedding API: set EVAL_PROVIDER_EMBEDDINGS=hashing"
                )
            if provider.embedding_model is None:
                raise ConfigurationError(
                    "the ragas judge needs EVAL_PROVIDER_EMBEDDING_MODEL "
                    "(or EVAL_PROVIDER_EMBEDDINGS=hashing)"
                )
        return provider.llm_model

    def _api_key(self) -> str:
        key = self.provider.api_key
        return key.get_secret_value() if key is not None else _LOCAL_PLACEHOLDER_KEY

    def _openai_client(self) -> "AsyncOpenAI":
        if self._openai is None:
            from openai import AsyncOpenAI  # noqa: PLC0415

            base_url = self.provider.base_url
            self._openai = AsyncOpenAI(
                api_key=self._api_key(),
                base_url=None if base_url is None else str(base_url),
                timeout=self.provider.request_timeout_seconds,
                max_retries=0,  # the engine owns retries and backoff
            )
        return self._openai

    def _anthropic_client(self) -> object:
        if self._anthropic is None:
            try:
                from anthropic import AsyncAnthropic  # noqa: PLC0415
            except ImportError:
                raise ConfigurationError(
                    "install the anthropic extra: uv sync --extra anthropic"
                ) from None
            base_url = self.provider.base_url
            self._anthropic = AsyncAnthropic(
                api_key=self._api_key(),
                base_url=None if base_url is None else str(base_url),
                timeout=self.provider.request_timeout_seconds,
                max_retries=0,
            )
        return self._anthropic

    def build_judge(self, backend: JudgeBackend, policy: ThresholdPolicy) -> JudgeProtocol:
        """The judge for ``backend``; provider judges fall back to the HeuristicJudge."""
        if backend in OFFLINE_JUDGE_MODELS:
            return HeuristicJudge(policy=policy)
        model = self._require_provider(backend)
        provider = self.provider
        fallback = HeuristicJudge(policy=policy) if provider.fallback_to_heuristic else None
        timeout = provider.request_timeout_seconds
        if backend == "ragas":
            from agentic_eval_benchmark.tools.offline import HashingEmbeddings  # noqa: PLC0415
            from agentic_eval_benchmark.tools.ragas_adapter import (  # noqa: PLC0415
                RagasJudge,
                ragas_embeddings,
                ragas_llm,
            )

            client = (
                self._openai_client() if provider.provider == "openai" else self._anthropic_client()
            )
            llm = ragas_llm(client, model, provider=provider.provider)
            embeddings = (
                HashingEmbeddings()
                if provider.embeddings == "hashing"
                else ragas_embeddings(self._openai_client(), str(provider.embedding_model))
            )
            return RagasJudge(
                llm, embeddings, policy=policy, metric_timeout_seconds=timeout, fallback=fallback
            )
        from trulens.providers.openai import OpenAI as TruLensOpenAI  # noqa: PLC0415

        from agentic_eval_benchmark.tools.trulens_adapter import TruLensJudge  # noqa: PLC0415

        options: dict[str, Any] = {"model_engine": model, "api_key": self._api_key()}
        if provider.base_url is not None:
            options["base_url"] = str(provider.base_url)
        return TruLensJudge(
            TruLensOpenAI(**options), policy=policy, timeout_seconds=timeout, fallback=fallback
        )

    # -- runs -----------------------------------------------------------------

    def baseline(self, run_id: str | None) -> BenchmarkRunSummary | None:
        return None if run_id is None else self.store.load(run_id).summary

    @asynccontextmanager
    async def run_slot(self) -> AsyncIterator[None]:
        """Admission control: refuse instead of queueing when all slots are busy."""
        if self.run_slots.locked():
            raise BusyError("an evaluation run is already in progress; retry later")
        async with self.run_slots:
            yield

    async def execute(self, plan: RunPlan) -> tuple[EvaluationRun, Path]:
        """Run, gate and persist; returns the run and its directory."""
        if plan.run_id is not None and self.store.exists(plan.run_id):
            raise ConflictError(f"run {plan.run_id} already exists")
        baseline = plan.baseline or self.baseline(plan.baseline_run_id)
        judge = self.build_judge(plan.judge, plan.policy)
        settings = self.settings.model_copy(update={"judge_model": plan.judge})
        async with self.rag_client(
            dataset=plan.dataset, url=plan.rag_url, oracle=plan.oracle
        ) as rag:
            engine = self.engine_factory(rag, judge, settings, plan.policy)
            try:
                run = await engine.run(plan.dataset, run_id=plan.run_id, baseline=baseline)
            except ValueError as exc:  # baseline mismatch: messages are built by the engine
                raise InvalidInputError(str(exc)) from None
        return run, self.store.save(run)

    def health(self) -> dict[str, object]:
        """Service status, pinned storage paths and judge availability."""
        writable = self.store.writable()
        judges = {b: self.judge_status(b).as_dict() for b in ("heuristic", "ragas", "trulens")}
        default = self.judge_status(self.settings.judge_model)
        healthy = writable and default.available and default.configured
        return {
            "status": "ok" if healthy else "degraded",
            "default_judge": self.settings.judge_model,
            "judges": judges,
            "storage": {
                "output_dir": str(self.store.root),
                "output_dir_writable": writable,
                "cache_root": str(cache_root()),
                "trulens_home": str(trulens_home()),
            },
            "runs": {
                "stored": len(self.store.list_runs()),
                "max_concurrent": self.api.max_concurrent_runs,
                "busy": self.run_slots.locked(),
            },
        }
