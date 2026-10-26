"""Ragas-backed LLM judge implementing :class:`~agentic_eval_benchmark.core.JudgeProtocol`.

The judge drives the Ragas v0.4 metric collections (``Faithfulness``,
``AnswerRelevancy``, ``ContextPrecision``, ``ContextRecall``) with an injected
structured-output LLM and embedding model, so any OpenAI-compatible endpoint, an
Anthropic client or the in-memory backends in :mod:`agentic_eval_benchmark.tools.offline`
can sit behind it.
"""

import asyncio
import math
import time
from collections.abc import Sequence
from typing import Any, Final, Protocol

from agentic_eval_benchmark.tools._isolation import pin_ragas_offline

pin_ragas_offline()  # before ragas builds any LLM, embedding or telemetry event

from ragas.embeddings.base import BaseRagasEmbedding, embedding_factory  # noqa: E402
from ragas.llms.base import InstructorBaseRagasLLM, llm_factory  # noqa: E402
from ragas.metrics.collections import (  # noqa: E402
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

from agentic_eval_benchmark.core import (  # noqa: E402
    DEFAULT_METRICS,
    JudgeProtocol,
    TransientError,
    build_score,
)
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

# Cosine-based scores can overshoot [0, 1] by float noise; larger excursions are bugs.
_RANGE_SLACK: Final = 1e-6
_NEEDS_ANSWER: Final = frozenset({MetricType.FAITHFULNESS, MetricType.ANSWER_RELEVANCY})
_NEEDS_CONTEXT: Final = frozenset(
    {MetricType.FAITHFULNESS, MetricType.CONTEXT_PRECISION, MetricType.CONTEXT_RECALL}
)


class _RagasResult(Protocol):
    @property
    def value(self) -> Any: ...

    reason: str | None


def ragas_llm(
    client: object, model: str, *, provider: str = "openai", **kwargs: Any
) -> InstructorBaseRagasLLM:
    """Wrap a pre-built SDK client as a Ragas structured-output LLM.

    ``client`` is e.g. ``openai.AsyncOpenAI(base_url=...)`` for any OpenAI-compatible
    server, or ``anthropic.AsyncAnthropic()`` with ``provider="anthropic"`` (install the
    ``anthropic`` extra). Nothing is created from ambient credentials.
    """
    pin_ragas_offline()
    llm = llm_factory(model, provider=provider, client=client, **kwargs)
    if not isinstance(llm, InstructorBaseRagasLLM):
        raise TypeError(f"llm_factory returned {type(llm).__name__}, not a structured LLM")
    return llm


def ragas_embeddings(
    client: object, model: str, *, provider: str = "openai", **kwargs: Any
) -> BaseRagasEmbedding:
    """Wrap a pre-built SDK client as a Ragas embedding model (modern interface)."""
    pin_ragas_offline()
    embeddings = embedding_factory(
        provider, model=model, client=client, interface="modern", **kwargs
    )
    if not isinstance(embeddings, BaseRagasEmbedding):
        raise TypeError(f"embedding_factory returned {type(embeddings).__name__}")
    return embeddings


class RagasJudge:
    """LLM-as-a-Judge over Ragas metrics, with fail-safe degradation.

    Per metric, a Ragas parsing/validation error, a timeout (``metric_timeout_seconds``)
    or a non-finite / out-of-range score does not fail the sample outright: the
    metric is rescored by ``fallback`` (e.g. the HeuristicJudge) when one is given,
    otherwise it becomes a structured failure verdict (score 0, failed) whose
    reasoning names the fault. Rate limits, 5xx and dropped connections raise
    :class:`~agentic_eval_benchmark.core.TransientError` so the engine retries them.
    Empty answers or contexts are scored 0 without calling the model.
    """

    def __init__(
        self,
        llm: InstructorBaseRagasLLM,
        embeddings: BaseRagasEmbedding | None = None,
        *,
        metrics: Sequence[MetricType] = DEFAULT_METRICS,
        policy: ThresholdPolicy | None = None,
        answer_relevancy_strictness: int = 3,
        metric_timeout_seconds: float | None = 120.0,
        fallback: JudgeProtocol | None = None,
    ) -> None:
        if not metrics or len(set(metrics)) != len(metrics):
            raise ValueError("metrics must be a non-empty sequence of unique metrics")
        unsupported = set(metrics) - set(DEFAULT_METRICS)
        if unsupported:
            raise ValueError(f"RagasJudge does not score {sorted(m.value for m in unsupported)}")
        if MetricType.ANSWER_RELEVANCY in metrics and embeddings is None:
            raise ValueError("answer_relevancy needs an embedding model")
        if not 1 <= answer_relevancy_strictness <= 10:  # noqa: PLR2004
            raise ValueError("answer_relevancy_strictness must be between 1 and 10")
        if metric_timeout_seconds is not None and metric_timeout_seconds <= 0:
            raise ValueError("metric_timeout_seconds must be positive")
        self._metrics = tuple(metrics)
        self._policy = policy or ThresholdPolicy()
        self._timeout = metric_timeout_seconds
        self._fallback = fallback
        self._faithfulness = Faithfulness(llm=llm)
        self._context_precision = ContextPrecision(llm=llm)
        self._context_recall = ContextRecall(llm=llm)
        self._answer_relevancy = (
            None
            if embeddings is None
            else AnswerRelevancy(
                llm=llm, embeddings=embeddings, strictness=answer_relevancy_strictness
            )
        )

    @property
    def metrics(self) -> tuple[MetricType, ...]:
        """Metrics this judge scores, in output order."""
        return self._metrics

    async def evaluate_sample(
        self,
        sample: GoldenSample,
        actual_answer: str,
        retrieved_contexts: tuple[str, ...],
    ) -> tuple[MetricScore, ...]:
        outcomes = await run_metrics(
            [self._score(m, sample, actual_answer, retrieved_contexts) for m in self._metrics]
        )
        failures = [o for o in outcomes if isinstance(o, MetricFailure)]
        if not failures:
            return tuple(o for o in outcomes if isinstance(o, MetricScore))
        recovered = await self._recover(failures, sample, actual_answer, retrieved_contexts)
        return tuple(o if isinstance(o, MetricScore) else recovered[o.metric] for o in outcomes)

    async def _score(
        self, metric: MetricType, sample: GoldenSample, answer: str, contexts: tuple[str, ...]
    ) -> Outcome:
        threshold = self._policy.threshold_for(metric)
        missing = (
            "Answer is empty."
            if metric in _NEEDS_ANSWER and not answer.strip()
            else "No context was retrieved."
            if metric in _NEEDS_CONTEXT and not contexts
            else None
        )
        if missing is not None:
            return build_score(metric, 0.0, missing, threshold=threshold)
        started = time.perf_counter()
        try:
            async with asyncio.timeout(self._timeout):
                result = await self._invoke(metric, sample, answer, list(contexts))
            value = float(result.value)
        except TimeoutError:
            return MetricFailure(metric, f"timed out after {self._timeout}s")
        except Exception as exc:
            if is_transient(exc):
                raise TransientError(describe(exc)) from exc
            return MetricFailure(metric, describe(exc))
        if not math.isfinite(value) or not -1.0 - _RANGE_SLACK <= value <= 1.0 + _RANGE_SLACK:
            return MetricFailure(metric, f"returned an invalid score {value!r}")
        reasoning = f"Ragas {metric.value} = {max(0.0, min(1.0, value)):.3f}."
        if result.reason:
            reasoning += f" {result.reason.strip()}"
        return build_score(
            metric,
            value,
            reasoning,
            threshold=threshold,
            latency_ms=round((time.perf_counter() - started) * 1000.0, 3),
        )

    async def _invoke(
        self, metric: MetricType, sample: GoldenSample, answer: str, contexts: list[str]
    ) -> _RagasResult:
        match metric:
            case MetricType.FAITHFULNESS:
                return await self._faithfulness.ascore(
                    user_input=sample.question, response=answer, retrieved_contexts=contexts
                )
            case MetricType.ANSWER_RELEVANCY if self._answer_relevancy is not None:
                return await self._answer_relevancy.ascore(
                    user_input=sample.question, response=answer
                )
            case MetricType.CONTEXT_PRECISION:
                return await self._context_precision.ascore(
                    user_input=sample.question,
                    reference=sample.ground_truth,
                    retrieved_contexts=contexts,
                )
            case MetricType.CONTEXT_RECALL:
                return await self._context_recall.ascore(
                    user_input=sample.question,
                    retrieved_contexts=contexts,
                    reference=sample.ground_truth,
                )
            case _:  # pragma: no cover - rejected in __init__
                raise ValueError(f"unsupported metric {metric.value}")

    async def _recover(
        self,
        failures: Sequence[MetricFailure],
        sample: GoldenSample,
        answer: str,
        contexts: tuple[str, ...],
    ) -> dict[MetricType, MetricScore]:
        fallback_scores: dict[MetricType, MetricScore] = {}
        fallback_error: str | None = None
        if self._fallback is not None:
            try:
                scored = await self._fallback.evaluate_sample(sample, answer, contexts)
                fallback_scores = {s.metric: s for s in scored if s.aspect is None}
            except Exception as exc:  # noqa: BLE001 - the fallback must not mask the fault
                fallback_error = describe(exc)
        recovered: dict[MetricType, MetricScore] = {}
        for failure in failures:
            fallback = fallback_scores.get(failure.metric)
            prefix = f"Ragas {failure.metric.value} failed ({failure.reason})"
            if fallback is not None:
                recovered[failure.metric] = build_score(
                    failure.metric,
                    fallback.score,
                    f"{prefix}; fallback judge: {fallback.reasoning}",
                    threshold=self._policy.threshold_for(failure.metric),
                    latency_ms=fallback.latency_ms,
                )
            else:
                why = f"fallback failed ({fallback_error})" if fallback_error else "no fallback"
                recovered[failure.metric] = failure_verdict(failure.metric, f"{prefix}; {why}.")
        return recovered
