"""Application factory: services, middleware, routes and zero-leakage problem handlers."""

import asyncio
import importlib
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from http import HTTPStatus

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from agentic_eval_benchmark import __version__
from agentic_eval_benchmark.api.middleware import RequestContextMiddleware
from agentic_eval_benchmark.api.problems import ProblemError, make_problem, problem_response
from agentic_eval_benchmark.api.routes import get_request_id, router
from agentic_eval_benchmark.api.services import EvaluationServices, ServiceError
from agentic_eval_benchmark.core import BenchmarkSettings
from agentic_eval_benchmark.tools._isolation import pin_trulens_offline

_log = logging.getLogger("agentic_eval_benchmark.api")

_WARMUP_MODULES = {
    "ragas": "agentic_eval_benchmark.tools.ragas_adapter",
    "trulens": "agentic_eval_benchmark.tools.trulens_adapter",
}


def _install_problem_handlers(app: FastAPI) -> None:
    @app.exception_handler(ProblemError)
    async def _problem(_request: Request, exc: ProblemError) -> Response:
        return problem_response(exc.problem, headers=exc.headers)

    @app.exception_handler(ServiceError)
    async def _service(request: Request, exc: ServiceError) -> Response:
        if exc.status >= 500:  # noqa: PLR2004
            _log.warning("service error %s: %s", exc.kind, exc.message)
        headers = {"retry-after": "30"} if exc.status == 429 else None  # noqa: PLR2004
        problem = make_problem(
            exc.status, exc.kind, exc.title, exc.message, request_id=get_request_id(request)
        )
        return problem_response(problem, headers=headers)

    @app.exception_handler(RequestValidationError)
    async def _invalid(request: Request, exc: RequestValidationError) -> Response:
        issues = [f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()[:5]]
        problem = make_problem(
            422,
            "invalid-input",
            "Unprocessable Content",
            "Invalid request: " + "; ".join(issues),
            request_id=get_request_id(request),
        )
        return problem_response(problem)

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException) -> Response:
        phrase = HTTPStatus(exc.status_code).phrase
        problem = make_problem(
            exc.status_code, "http-error", phrase, phrase, request_id=get_request_id(request)
        )
        return problem_response(problem)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> Response:
        _log.error("unhandled %s", type(exc).__name__, exc_info=exc)
        problem = make_problem(
            500,
            "internal-error",
            "Internal Server Error",
            "An unexpected error occurred.",
            request_id=get_request_id(request),
        )
        return problem_response(problem)


def create_app(
    services: EvaluationServices | None = None,
    *,
    settings: BenchmarkSettings | None = None,
) -> FastAPI:
    """Build the evaluation service.

    Pass ``services`` to inject settings, provider configuration, storage and the engine
    factory (tests do); otherwise everything is read from the ``EVAL_*`` environment.
    Startup verifies the pinned output directory and pre-imports the default provider
    judge so the first run does not pay the import cost; neither failure stops the app,
    which then reports ``degraded`` on ``/v1/eval/health``.
    """
    evaluation = services or EvaluationServices(settings)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        if not await asyncio.to_thread(evaluation.store.writable):
            _log.warning("output directory %s is not writable", evaluation.store.root)
        backend = evaluation.settings.judge_model
        if backend == "trulens":
            pin_trulens_offline()
        if backend in _WARMUP_MODULES:
            try:
                await asyncio.to_thread(importlib.import_module, _WARMUP_MODULES[backend])
            except ImportError as exc:
                _log.warning("default judge %s unavailable: %s", backend, type(exc).__name__)
        try:
            yield
        finally:
            await evaluation.aclose()

    app = FastAPI(
        title="agentic-eval-benchmark",
        version=__version__,
        summary="LLM-as-a-Judge evaluation of RAG endpoints with strict quality gates.",
        lifespan=lifespan,
        # No OpenTelemetry spans, metrics or OTLP export, whatever the environment says.
        telemetry={"auto_configure": False, "tracing": False, "metrics": False, "logs": False},
    )
    app.state.services = evaluation
    app.include_router(router)
    _install_problem_handlers(app)
    app.add_middleware(RequestContextMiddleware, max_body_bytes=evaluation.api.max_request_bytes)
    return app
