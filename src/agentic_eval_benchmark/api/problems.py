"""RFC 7807 problem details. Problems never echo the request: no dataset content,
answers, URLs or credentials reach the body."""

from typing import Final

from pydantic import BaseModel, ConfigDict, Field
from starlette.responses import Response

PROBLEM_MEDIA_TYPE: Final = "application/problem+json"
_TYPE_PREFIX: Final = "urn:agentic-eval-benchmark:problem:"
_MAX_DETAIL: Final = 512


class ProblemResponse(BaseModel):
    """``application/problem+json`` body."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: str = Field(min_length=1, max_length=128)
    title: str = Field(min_length=1, max_length=128)
    status: int = Field(ge=400, le=599)
    detail: str = Field(min_length=1, max_length=_MAX_DETAIL)
    instance: str | None = Field(default=None, max_length=160)


class ProblemError(Exception):
    """Raise from routes or dependencies to answer with a problem response."""

    def __init__(self, problem: ProblemResponse, headers: dict[str, str] | None = None) -> None:
        super().__init__(problem.detail)
        self.problem = problem
        self.headers = headers or {}


def make_problem(
    status: int, kind: str, title: str, detail: str, *, request_id: str | None = None
) -> ProblemResponse:
    """Build a problem; ``kind`` becomes the ``type`` URN suffix."""
    text = " ".join(detail.split()) or title
    return ProblemResponse(
        type=_TYPE_PREFIX + kind,
        title=title,
        status=status,
        detail=text if len(text) <= _MAX_DETAIL else text[: _MAX_DETAIL - 1] + "…",
        instance=f"urn:request:{request_id}" if request_id else None,
    )


def problem_response(
    problem: ProblemResponse, *, headers: dict[str, str] | None = None
) -> Response:
    """Starlette response for ``problem`` (never cached)."""
    return Response(
        problem.model_dump_json(exclude_none=True),
        status_code=problem.status,
        media_type=PROBLEM_MEDIA_TYPE,
        headers={"cache-control": "no-store", **(headers or {})},
    )
