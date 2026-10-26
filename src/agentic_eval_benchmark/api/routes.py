"""``/v1/eval`` endpoints: trigger runs, fetch stored reports, report health."""

import asyncio
import secrets
from typing import Annotated, Final, Self

from fastapi import APIRouter, Depends, Path, Request, Response
from fastapi.responses import JSONResponse
from pydantic import Field, ValidationError, model_validator

from agentic_eval_benchmark import __version__
from agentic_eval_benchmark.api.problems import ProblemError, make_problem
from agentic_eval_benchmark.api.services import (
    EvaluationServices,
    InvalidInputError,
    RunPlan,
)
from agentic_eval_benchmark.core import JudgeBackend
from agentic_eval_benchmark.schemas import (
    BenchmarkReport,
    BenchmarkRunSummary,
    EvalModel,
    GoldenDataset,
    Identifier,
    ThresholdPolicy,
)

RUN_ID_PATTERN: Final = r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$"
_MAX_VALIDATION_ERRORS: Final = 5
_JSON_TYPES: Final = ("application/json", "application/problem+json")


class RunRequest(EvalModel):
    """Body of ``POST /v1/eval/run``; every field is optional.

    The dataset is ``dataset`` (inline upload), ``dataset_path`` (relative to the
    server's ``EVAL_API_DATASET_DIR``) or, when both are absent, the built-in synthetic
    reference dataset. ``oracle`` answers from the dataset itself instead of calling a
    RAG endpoint, which smoke-tests the pipeline offline.
    """

    dataset: GoldenDataset | None = None
    dataset_path: str | None = Field(default=None, min_length=1, max_length=1024)
    rag_url: str | None = Field(default=None, min_length=1, max_length=2048)
    oracle: bool = False
    judge: JudgeBackend | None = None
    policy: ThresholdPolicy | None = None
    baseline_run_id: Identifier | None = None
    run_id: Identifier | None = None

    @model_validator(mode="after")
    def _exclusive(self) -> Self:
        if self.dataset is not None and self.dataset_path is not None:
            raise ValueError("pass either dataset or dataset_path, not both")
        if self.oracle and self.rag_url is not None:
            raise ValueError("oracle runs do not call a RAG endpoint; drop rag_url")
        return self


def validation_detail(exc: ValidationError) -> str:
    """Field locations and pydantic's messages, which never contain the submitted values
    (``include_input=False``); validator messages are written by this package."""
    errors = exc.errors(include_url=False, include_input=False, include_context=False)
    issues = [
        f"{'.'.join(map(str, e['loc'])) or '<body>'}: {e['msg']}"
        for e in errors[:_MAX_VALIDATION_ERRORS]
    ]
    more = len(errors) - _MAX_VALIDATION_ERRORS
    return "Invalid request: " + "; ".join(issues) + (f" (+{more} more)" if more > 0 else "")


def get_request_id(request: Request) -> str | None:
    value = getattr(request.state, "request_id", None)
    return value if isinstance(value, str) else None


def get_services(request: Request) -> EvaluationServices:
    services: EvaluationServices = request.app.state.services
    return services


ServicesDep = Annotated[EvaluationServices, Depends(get_services)]
RunId = Annotated[str, Path(pattern=RUN_ID_PATTERN, max_length=128)]


def require_api_key(request: Request, services: ServicesDep) -> None:
    """Enforce ``EVAL_API_API_KEY`` when configured (constant-time comparison)."""
    expected = services.api.api_key
    if expected is None:
        return
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    presented = (
        token.strip() if scheme.lower() == "bearer" else request.headers.get("x-api-key", "")
    )
    if not secrets.compare_digest(presented.encode(), expected.get_secret_value().encode()):
        raise ProblemError(
            make_problem(
                401,
                "unauthorized",
                "Unauthorized",
                "A valid API key is required.",
                request_id=get_request_id(request),
            ),
            headers={"www-authenticate": "Bearer"},
        )


router = APIRouter(prefix="/v1/eval", tags=["evaluation"])
_protected: Final = [Depends(require_api_key)]


async def _resolve_dataset(services: EvaluationServices, body: RunRequest) -> GoldenDataset:
    if body.dataset is not None:
        return body.dataset
    if body.dataset_path is not None:
        path = services.resolve_api_dataset_path(body.dataset_path)
        return await asyncio.to_thread(services.load_dataset_file, path)
    return await asyncio.to_thread(services.reference_dataset)


@router.post(
    "/run",
    status_code=201,
    response_model=BenchmarkRunSummary,
    dependencies=_protected,
    summary="Run a benchmark and return its gated summary",
    openapi_extra={
        "requestBody": {
            "required": False,
            "content": {"application/json": {"schema": RunRequest.model_json_schema()}},
        }
    },
)
async def run_benchmark(request: Request, services: ServicesDep) -> Response:
    """Evaluate synchronously; ``X-Eval-Gate`` carries the verdict. Gate failures are
    still ``201``: the run happened and its summary says why it failed."""
    content_type = request.headers.get("content-type", "application/json").split(";")[0]
    raw = await request.body()
    if raw.strip() and content_type.strip().lower() not in _JSON_TYPES:
        raise ProblemError(
            make_problem(
                415,
                "unsupported-media-type",
                "Unsupported Media Type",
                "The run request must be application/json.",
                request_id=get_request_id(request),
            )
        )
    try:
        body = RunRequest.model_validate_json(raw if raw.strip() else b"{}")
    except ValidationError as exc:
        raise InvalidInputError(validation_detail(exc)) from None
    dataset = await _resolve_dataset(services, body)
    rag_url = None if body.rag_url is None else services.check_rag_override(body.rag_url)
    plan = RunPlan(
        dataset=dataset,
        policy=body.policy or ThresholdPolicy(),
        judge=body.judge or services.settings.judge_model,
        rag_url=rag_url,
        oracle=body.oracle,
        baseline_run_id=body.baseline_run_id,
        run_id=body.run_id,
    )
    async with services.run_slot():
        run, _ = await services.execute(plan)
    summary = run.summary
    return Response(
        summary.to_json(),
        status_code=201,
        media_type="application/json",
        headers={
            "location": f"{router.prefix}/report/{summary.run_id}",
            "x-eval-gate": "passed" if summary.gate_passed else "failed",
            "x-eval-sample-errors": str(len(run.errors)),
        },
    )


@router.get(
    "/report/{run_id}",
    response_model=BenchmarkReport,
    dependencies=_protected,
    summary="Stored report with per-sample evaluations",
)
async def get_report(run_id: RunId, services: ServicesDep) -> Response:
    report = await asyncio.to_thread(services.store.load, run_id)
    return Response(report.to_json(), media_type="application/json")


@router.get(
    "/report/{run_id}/summary",
    response_model=BenchmarkRunSummary,
    dependencies=_protected,
    summary="Stored run summary only",
)
async def get_summary(run_id: RunId, services: ServicesDep) -> Response:
    report = await asyncio.to_thread(services.store.load, run_id)
    return Response(report.summary.to_json(), media_type="application/json")


@router.get("/health", summary="Service status, pinned storage paths and judge availability")
async def health(services: ServicesDep) -> JSONResponse:
    status = await asyncio.to_thread(services.health)
    status["version"] = __version__
    return JSONResponse(status, status_code=200 if status["status"] == "ok" else 503)
