"""TruLens-backed judge: groundedness, answer relevance and custom feedback functions.

TruLens feedback providers are synchronous; each call runs in a worker thread so the
event loop stays free. Groundedness is configured with ``use_sent_tokenize=False``
because the NLTK tokenizer path downloads ``punkt_tab`` at call time, which would
break hermetic runs; the provider splits statements instead.
"""

import asyncio
import functools
import inspect
import math
import re
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from agentic_eval_benchmark.tools._isolation import pin_trulens_offline, trulens_home

pin_trulens_offline()

from trulens.core.metric.metric import GroundednessConfigs  # noqa: E402
from trulens.feedback.llm_provider import LLMProvider  # noqa: E402

from agentic_eval_benchmark.core import JudgeProtocol, TransientError, build_score  # noqa: E402
from agentic_eval_benchmark.core._text import excerpt  # noqa: E402
from agentic_eval_benchmark.schemas import (  # noqa: E402
    GoldenSample,
    MetricScore,
    MetricType,
    ThresholdPolicy,
)
from agentic_eval_benchmark.tools._judging import (  # noqa: E402
    MetricFailure,
    Outcome,
    describe,
    failure_verdict,
    is_transient,
    run_metrics,
)

if TYPE_CHECKING:
    from trulens.core.session import TruSession

TRULENS_METRICS: Final = (MetricType.FAITHFULNESS, MetricType.ANSWER_RELEVANCY)
"""Metrics with a TruLens equivalent: groundedness and answer relevance."""

DATABASE_FILE: Final = "trulens.sqlite"
_ASPECT_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_GROUNDED: Final = 0.5
_NO_SENTENCE_SPLIT: Final = GroundednessConfigs(
    use_sent_tokenize=False, filter_trivial_statements=False
)

type FeedbackResult = tuple[float, str]
type FeedbackFunction = Callable[
    [GoldenSample, str, tuple[str, ...]], FeedbackResult | Awaitable[FeedbackResult]
]


@dataclass(frozen=True, slots=True)
class CustomFeedback:
    """A standalone feedback loop reported as an ``aspect_critique`` metric.

    ``function(sample, answer, contexts)`` returns ``(score in [0, 1], reason)``; it
    may be sync (run in a worker thread) or async. Without a ``threshold`` the
    critique is informational and always passes.
    """

    name: str
    function: FeedbackFunction
    threshold: float | None = None

    def __post_init__(self) -> None:
        if not _ASPECT_RE.fullmatch(self.name):
            raise ValueError(f"feedback name {self.name!r} is not a valid identifier")
        if self.threshold is not None and not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")


def criteria_feedback(
    name: str,
    criteria: str,
    provider: LLMProvider,
    *,
    threshold: float | None = None,
    max_score: int = 3,
) -> CustomFeedback:
    """A TruLens chain-of-thought feedback scoring the answer against free-text ``criteria``."""
    if not criteria.strip():
        raise ValueError("criteria must not be empty")
    system_prompt = (
        "You are a strict evaluator of answers produced by a retrieval-augmented system.\n"
        f"Score the ANSWER from 0 to {max_score} against this criterion: {criteria.strip()}\n"
        f"0 means the criterion is not met at all; {max_score} means it is fully met."
    )

    def evaluate(sample: GoldenSample, answer: str, contexts: tuple[str, ...]) -> FeedbackResult:
        user_prompt = (
            f"QUESTION: {sample.question}\n\nCONTEXT:\n" + "\n---\n".join(contexts)
            + f"\n\nANSWER: {answer}"
        )  # fmt: skip
        score, details = provider.generate_score_and_reasons(
            system_prompt=system_prompt, user_prompt=user_prompt, max_score_val=max_score
        )
        return float(score), _reason_text(details) or f"TruLens criterion '{name}' scored."

    return CustomFeedback(name=name, function=evaluate, threshold=threshold)


def _reason_text(details: object) -> str:
    """Flatten a TruLens reason payload (``{"reason": str}`` or nested) into one line."""
    if isinstance(details, Mapping):
        reason = details.get("reason")
        if isinstance(reason, str):
            return " ".join(reason.split())
    return ""


def _statements(details: object) -> list[tuple[str, float]]:
    """``(statement, score)`` pairs from a TruLens groundedness reason payload."""
    reasons = details.get("reasons") if isinstance(details, Mapping) else None
    pairs: list[tuple[str, float]] = []
    for item in reasons if isinstance(reasons, list) else ():
        if isinstance(item, Mapping):
            statement, score = item.get("criteria"), item.get("score")
            if isinstance(statement, str) and isinstance(score, int | float):
                pairs.append((statement, float(score)))
    return pairs


class TruLensJudge:
    """LLM-as-a-Judge over TruLens feedback functions.

    - ``FAITHFULNESS`` ← ``groundedness_measure_with_cot_reasons`` (answer vs. contexts)
    - ``ANSWER_RELEVANCY`` ← ``relevance_with_cot_reasons`` (question vs. answer)
    - each :class:`CustomFeedback` ← an ``ASPECT_CRITIQUE`` metric named after it

    Faults: the TruLens endpoint retries provider calls itself and reports exhaustion as
    a plain ``RuntimeError``, so provider faults (and timeouts, after which the worker
    thread is abandoned) are rescored by ``fallback`` or become failure verdicts.
    Transport faults raised by custom feedback functions surface as ``TransientError``
    for the engine to retry; other feedback errors fail that critique (no fallback).
    """

    def __init__(
        self,
        provider: LLMProvider,
        *,
        metrics: Sequence[MetricType] = TRULENS_METRICS,
        feedbacks: Sequence[CustomFeedback] = (),
        policy: ThresholdPolicy | None = None,
        timeout_seconds: float | None = 120.0,
        fallback: JudgeProtocol | None = None,
    ) -> None:
        if len(set(metrics)) != len(metrics):
            raise ValueError("metrics must be unique")
        unsupported = set(metrics) - set(TRULENS_METRICS)
        if unsupported:
            raise ValueError(f"TruLensJudge does not score {sorted(m.value for m in unsupported)}")
        names = [f.name for f in feedbacks]
        if len(set(names)) != len(names):
            raise ValueError("custom feedback names must be unique")
        if not metrics and not feedbacks:
            raise ValueError("configure at least one metric or custom feedback")
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._provider = provider
        self._metrics = tuple(metrics)
        self._feedbacks = tuple(feedbacks)
        self._policy = policy or ThresholdPolicy()
        self._timeout = timeout_seconds
        self._fallback = fallback

    async def evaluate_sample(
        self,
        sample: GoldenSample,
        actual_answer: str,
        retrieved_contexts: tuple[str, ...],
    ) -> tuple[MetricScore, ...]:
        jobs: list[Awaitable[Outcome]] = [
            self._metric(m, sample, actual_answer, retrieved_contexts) for m in self._metrics
        ]
        jobs += [
            self._custom(f, sample, actual_answer, retrieved_contexts) for f in self._feedbacks
        ]
        outcomes = await run_metrics(jobs)
        failures = [o for o in outcomes if isinstance(o, MetricFailure) and o.aspect is None]
        fallback: dict[MetricType, MetricScore] = {}
        if failures and self._fallback is not None:
            try:
                scored = await self._fallback.evaluate_sample(
                    sample, actual_answer, retrieved_contexts
                )
                fallback = {s.metric: s for s in scored if s.aspect is None}
            except Exception:  # noqa: BLE001 - keep the original failure verdicts
                fallback = {}
        return tuple(self._resolve(o, fallback) for o in outcomes)

    def _resolve(self, outcome: Outcome, fallback: Mapping[MetricType, MetricScore]) -> MetricScore:
        if isinstance(outcome, MetricScore):
            return outcome
        label = outcome.aspect or outcome.metric.value
        prefix = f"TruLens {label} failed ({outcome.reason})"
        rescue = None if outcome.aspect else fallback.get(outcome.metric)
        if rescue is None:
            return failure_verdict(outcome.metric, f"{prefix}.", aspect=outcome.aspect)
        return build_score(
            outcome.metric,
            rescue.score,
            f"{prefix}; fallback judge: {rescue.reasoning}",
            threshold=self._policy.threshold_for(outcome.metric),
            latency_ms=rescue.latency_ms,
        )

    async def _timed[R](self, call: Awaitable[R]) -> tuple[R, float]:
        started = time.perf_counter()
        async with asyncio.timeout(self._timeout):
            result = await call
        return result, round((time.perf_counter() - started) * 1000.0, 3)

    async def _metric(
        self, metric: MetricType, sample: GoldenSample, answer: str, contexts: tuple[str, ...]
    ) -> Outcome:
        threshold = self._policy.threshold_for(metric)
        if not answer.strip():
            return build_score(metric, 0.0, "Answer is empty.", threshold=threshold)
        if metric is MetricType.FAITHFULNESS and not contexts:
            return build_score(metric, 0.0, "No context was retrieved.", threshold=threshold)
        try:
            if metric is MetricType.FAITHFULNESS:
                (score, details), latency = await self._timed(
                    asyncio.to_thread(
                        self._provider.groundedness_measure_with_cot_reasons,
                        source="\n\n".join(contexts),
                        statement=answer,
                        groundedness_configs=_NO_SENTENCE_SPLIT,
                    )
                )
                reasoning = self._grounding_reason(float(score), details)
            else:
                (score, details), latency = await self._timed(
                    asyncio.to_thread(
                        self._provider.relevance_with_cot_reasons,
                        prompt=sample.question,
                        response=answer,
                    )
                )
                reasoning = f"TruLens relevance = {float(score):.3f}."
                if text := _reason_text(details):
                    reasoning += f" {text}"
        except TimeoutError:
            return MetricFailure(metric, f"timed out after {self._timeout}s")
        except Exception as exc:
            if is_transient(exc):
                raise TransientError(describe(exc)) from exc
            return MetricFailure(metric, describe(exc))
        value = float(score)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            return MetricFailure(metric, f"returned invalid score {value!r}")
        return build_score(metric, value, reasoning, threshold=threshold, latency_ms=latency)

    @staticmethod
    def _grounding_reason(score: float, details: object) -> str:
        statements = _statements(details)
        reasoning = f"TruLens groundedness = {score:.3f}."
        if statements:
            grounded = sum(s >= _GROUNDED for _, s in statements)
            reasoning += f" {grounded}/{len(statements)} statements supported by the context."
            weakest = [f"“{excerpt(t, 60)}”" for t, s in statements if s < _GROUNDED][:3]
            if weakest:
                reasoning += f" Unsupported: {'; '.join(weakest)}."
        return reasoning

    async def _custom(
        self,
        feedback: CustomFeedback,
        sample: GoldenSample,
        answer: str,
        contexts: tuple[str, ...],
    ) -> Outcome:
        critique = MetricType.ASPECT_CRITIQUE
        try:
            (score, reason), latency = await self._timed(
                _run_feedback(feedback, sample, answer, contexts)
            )
        except TimeoutError:
            return MetricFailure(critique, f"timed out after {self._timeout}s", feedback.name)
        except Exception as exc:
            if is_transient(exc):
                raise TransientError(describe(exc)) from exc
            return MetricFailure(critique, describe(exc), feedback.name)
        value = float(score)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            return MetricFailure(critique, f"returned invalid score {value!r}", feedback.name)
        return build_score(
            critique,
            value,
            reason.strip() or f"Custom feedback {feedback.name} scored.",
            aspect=feedback.name,
            threshold=feedback.threshold,
            latency_ms=latency,
        )


async def _run_feedback(
    feedback: CustomFeedback, sample: GoldenSample, answer: str, contexts: tuple[str, ...]
) -> FeedbackResult:
    call = functools.partial(feedback.function, sample, answer, contexts)
    result = (
        call() if inspect.iscoroutinefunction(feedback.function) else await asyncio.to_thread(call)
    )
    if isinstance(result, tuple):
        return result
    return await result


def open_trulens_session(home: Path | None = None) -> "TruSession":
    """Open the process-wide TruSession on ``<home>/trulens.sqlite``.

    ``home`` defaults to ``$TRULENS_HOME`` (``./.cache/trulens`` via ``.envrc``), never
    TruLens' own ``./default.sqlite``. Keys are redacted in the database. Raises
    :class:`RuntimeError` if a session on another database already exists in the process.
    """
    pin_trulens_offline()
    from trulens.core.database.connector.default import DefaultDBConnector  # noqa: PLC0415
    from trulens.core.session import TruSession  # noqa: PLC0415

    state = (home or trulens_home()).expanduser().resolve()
    state.mkdir(parents=True, exist_ok=True)
    database = state / DATABASE_FILE
    url = f"sqlite:///{database}"
    session = TruSession(connector=DefaultDBConnector(database_url=url, database_redact_keys=True))
    actual: Any = getattr(getattr(session.connector, "db", None), "engine", None)
    actual_path = getattr(getattr(actual, "url", None), "database", None)
    if actual_path is not None and Path(actual_path).resolve() != database:
        raise RuntimeError(f"a TruSession on {actual_path} is already open in this process")
    return session
