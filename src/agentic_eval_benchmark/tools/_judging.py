"""Shared fault classification and failure verdicts for the provider-backed judges."""

import asyncio
from collections.abc import Awaitable, Iterator, Sequence
from dataclasses import dataclass
from typing import Final

import httpx

from agentic_eval_benchmark.core import TransientError
from agentic_eval_benchmark.schemas import MetricScore, MetricType

# 408 request timeout, 429 rate limit, 5xx server faults, 529 overloaded (Anthropic).
TRANSIENT_STATUS_CODES: Final = frozenset({408, 429, 500, 502, 503, 504, 529})
_MAX_REASON_CHARS: Final = 300


def _chain(exc: BaseException) -> Iterator[BaseException]:
    """``exc`` and its causes / contexts (SDK wrappers hide the root fault)."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def is_timeout(exc: BaseException) -> bool:
    """Whether ``exc`` (or a cause) is a timeout, whatever SDK raised it."""
    return any(
        isinstance(e, TimeoutError | httpx.TimeoutException) or "Timeout" in type(e).__name__
        for e in _chain(exc)
    )


def is_transient(exc: BaseException) -> bool:
    """Whether ``exc`` (or a cause) is a retryable transport or capacity fault.

    Timeouts are excluded: judges degrade on them instead of retrying. Provider SDK
    errors are recognized by their HTTP ``status_code`` (OpenAI and Anthropic both
    expose it), so no SDK has to be imported here.
    """
    if is_timeout(exc):
        return False
    for e in _chain(exc):
        if isinstance(e, ConnectionError | httpx.TransportError):
            return True
        if "Connection" in type(e).__name__ or "RateLimit" in type(e).__name__:
            return True
        status = getattr(e, "status_code", None)
        if isinstance(status, int) and status in TRANSIENT_STATUS_CODES:
            return True
    return False


def describe(exc: BaseException) -> str:
    """Short, single-line fault description for a verdict's reasoning."""
    detail = " ".join(str(exc).split())
    text = f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__
    return text if len(text) <= _MAX_REASON_CHARS else text[: _MAX_REASON_CHARS - 1] + "…"


def failure_verdict(metric: MetricType, reason: str, *, aspect: str | None = None) -> MetricScore:
    """A failed, zero-score verdict recording why the metric could not be judged.

    It carries no threshold so ``passed=False`` holds for gated and ungated metrics
    alike; the engine re-applies the policy threshold to gated metrics.
    """
    return MetricScore(metric=metric, aspect=aspect, score=0.0, passed=False, reasoning=reason)


@dataclass(frozen=True, slots=True)
class MetricFailure:
    """A metric the provider could not score (parse error, timeout, invalid value)."""

    metric: MetricType
    reason: str
    aspect: str | None = None


type Outcome = MetricScore | MetricFailure


async def run_metrics(jobs: Sequence[Awaitable[Outcome]]) -> list[Outcome]:
    """Run per-metric jobs concurrently, preserving order.

    Jobs return a score or a :class:`MetricFailure`; a job raising
    :class:`TransientError` cancels its siblings and surfaces as a single
    ``TransientError`` so the engine retries the whole sample.
    """
    try:
        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(_as_coroutine(job)) for job in jobs]
    except ExceptionGroup as errors:
        transient = errors.subgroup(lambda e: isinstance(e, TransientError) or is_transient(e))
        if transient is not None:
            raise TransientError(describe(transient.exceptions[0])) from errors
        raise
    return [task.result() for task in tasks]


async def _as_coroutine(job: Awaitable[Outcome]) -> Outcome:
    return await job
