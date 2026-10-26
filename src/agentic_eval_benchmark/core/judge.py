"""LLM-as-a-Judge port, the deterministic offline judge and the strict verdict parser."""

import asyncio
import json
import re
import time
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Final, Literal, Protocol, Self, runtime_checkable

from pydantic import Field, ValidationError, model_validator

from agentic_eval_benchmark.core._text import (
    content_terms,
    coverage,
    excerpt,
    is_negated,
    normalize,
    numbers,
    overlap_coefficient,
    split_claims,
    tokenize,
)
from agentic_eval_benchmark.schemas import (
    EvalModel,
    GoldenSample,
    Identifier,
    MetricScore,
    MetricType,
    NonEmptyText,
    ThresholdPolicy,
    UnitInterval,
    metric_key,
)
from agentic_eval_benchmark.schemas.metrics import check_aspect

DEFAULT_METRICS: Final = (
    MetricType.FAITHFULNESS,
    MetricType.ANSWER_RELEVANCY,
    MetricType.CONTEXT_PRECISION,
    MetricType.CONTEXT_RECALL,
)
"""The four RAG metrics every judge is expected to score."""


@runtime_checkable
class JudgeProtocol(Protocol):
    """Port for anything that scores a RAG answer against a Golden Dataset sample.

    Implementations return one :class:`MetricScore` per metric they evaluate. They
    raise :class:`~agentic_eval_benchmark.core.TransientError` for retryable faults
    (rate limits, timeouts); any other exception fails the sample.
    """

    async def evaluate_sample(
        self,
        sample: GoldenSample,
        actual_answer: str,
        retrieved_contexts: tuple[str, ...],
    ) -> tuple[MetricScore, ...]: ...


def build_score(
    metric: MetricType,
    score: float,
    reasoning: str,
    *,
    threshold: float | None,
    aspect: str | None = None,
    cost_usd: float | None = None,
    latency_ms: float | None = None,
) -> MetricScore:
    """Build a :class:`MetricScore`, clamping to ``[0, 1]`` and deriving ``passed``.

    Without a ``threshold`` the metric is informational and always passes.
    """
    value = min(1.0, max(0.0, round(score, 4)))
    return MetricScore(
        metric=metric,
        aspect=aspect,
        score=value,
        threshold=threshold,
        passed=True if threshold is None else value >= threshold,
        reasoning=reasoning,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
    )


# ---------------------------------------------------------------------------
# HeuristicJudge
# ---------------------------------------------------------------------------

type Support = Literal["exact", "fuzzy"]

_NONCOMMITTAL_RE: Final = re.compile(
    r"\b(?:i (?:do not|don't) know|i(?:'m| am) not sure|(?:cannot|can't|unable to)"
    r" (?:answer|determine|find|say|tell)|no (?:information|answer|relevant)"
    r"|not enough (?:information|context)|insufficient (?:information|context))\b"
)
_MAX_LISTED: Final = 3


def _flat(text: str) -> str:
    """Token sequence joined by single spaces: punctuation-insensitive exact matching."""
    return " ".join(tokenize(text))


@dataclass(frozen=True, slots=True)
class _Evidence:
    """A context chunk pre-processed once for claim verification."""

    flat: str
    terms: frozenset[str]
    numbers: frozenset[str]
    sentences: tuple[tuple[frozenset[str], bool], ...]

    @classmethod
    def of(cls, text: str) -> Self:
        sentences = tuple((content_terms(s), is_negated(s)) for s in split_claims(text))
        return cls(
            flat=f" {_flat(text)} ",
            terms=content_terms(text),
            numbers=numbers(text),
            sentences=sentences or ((content_terms(text), is_negated(text)),),
        )


class HeuristicJudge:
    """Deterministic, network-free judge for baselines, CI and offline fallback.

    It approximates the Ragas metrics with lexical evidence only:

    - **Faithfulness**: share of answer claims grounded in the retrieved context, by
      exact (punctuation-insensitive) substring or fuzzy match. A fuzzy match needs
      ``support_threshold`` content-term coverage within one chunk, no numbers the
      chunk lacks, and the polarity (negated or not) of the best-matching sentence.
    - **Answer relevancy**: an embedding-free proxy for question-answer alignment:
      the larger of the share of reference information the question asks for that
      the answer supplies and the share of question terms the answer addresses
      (weight 0.7), plus the share of answer terms that stay on topic (weight 0.3).
      Evasive answers ("I don't know") score 0.
    - **Context precision**: rank-weighted average precision of retrieved chunks,
      a chunk being relevant when it overlaps a reference context by at least
      ``relevance_threshold`` or grounds a ground-truth claim.
    - **Context recall**: share of ground-truth claims grounded in the retrieved context.

    Scores are reproducible across runs and platforms; it is a lexical baseline, not
    a substitute for a semantic judge on paraphrased answers.
    """

    def __init__(
        self,
        *,
        metrics: Sequence[MetricType] = DEFAULT_METRICS,
        policy: ThresholdPolicy | None = None,
        support_threshold: float = 0.75,
        relevance_threshold: float = 0.5,
    ) -> None:
        if not metrics:
            raise ValueError("at least one metric is required")
        if len(set(metrics)) != len(metrics):
            raise ValueError("metrics must be unique")
        if MetricType.ASPECT_CRITIQUE in metrics:
            raise ValueError("HeuristicJudge cannot score aspect_critique; use an LLM judge")
        for name, value in (
            ("support_threshold", support_threshold),
            ("relevance_threshold", relevance_threshold),
        ):
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1], got {value}")
        self._metrics = tuple(metrics)
        self._policy = policy or ThresholdPolicy()
        self._support_threshold = support_threshold
        self._relevance_threshold = relevance_threshold

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
        """Score off the event loop: the work is CPU-bound."""
        return await asyncio.to_thread(self.score, sample, actual_answer, retrieved_contexts)

    def score(
        self,
        sample: GoldenSample,
        actual_answer: str,
        retrieved_contexts: tuple[str, ...],
    ) -> tuple[MetricScore, ...]:
        """Synchronous scoring of every configured metric."""
        evidence = tuple(_Evidence.of(c) for c in retrieved_contexts)
        results: list[MetricScore] = []
        for metric in self._metrics:
            started = time.perf_counter()
            match metric:
                case MetricType.FAITHFULNESS:
                    value, reasoning = self._faithfulness(actual_answer, evidence)
                case MetricType.ANSWER_RELEVANCY:
                    value, reasoning = self._answer_relevancy(sample, actual_answer)
                case MetricType.CONTEXT_PRECISION:
                    value, reasoning = self._context_precision(sample, retrieved_contexts)
                case _:
                    value, reasoning = self._context_recall(sample, evidence)
            results.append(
                build_score(
                    metric,
                    value,
                    reasoning,
                    threshold=self._policy.threshold_for(metric),
                    cost_usd=0.0,
                    latency_ms=round((time.perf_counter() - started) * 1000.0, 3),
                )
            )
        return tuple(results)

    # -- claim grounding ----------------------------------------------------

    def _support(self, claim: str, evidence: Sequence[_Evidence]) -> Support | None:
        flat = _flat(claim)
        if flat and any(f" {flat} " in ev.flat for ev in evidence):
            return "exact"
        terms, nums, negated = content_terms(claim), numbers(claim), is_negated(claim)
        for ev in evidence:
            if not nums <= ev.numbers or coverage(terms, ev.terms) < self._support_threshold:
                continue
            _, sentence_negated = max(ev.sentences, key=lambda s: coverage(terms, s[0]))
            if sentence_negated == negated:
                return "fuzzy"
        return None

    def _faithfulness(self, answer: str, evidence: Sequence[_Evidence]) -> tuple[float, str]:
        claims = split_claims(answer)
        if not claims:
            return 0.0, "Answer contains no verifiable claims."
        if not evidence:
            return 0.0, "No retrieved context to ground the answer in."
        support = [self._support(c, evidence) for c in claims]
        exact = support.count("exact")
        fuzzy = support.count("fuzzy")
        reasoning = (
            f"{exact + fuzzy}/{len(claims)} answer claims grounded in the retrieved context "
            f"({exact} exact, {fuzzy} fuzzy)."
        )
        unsupported = [c for c, s in zip(claims, support, strict=True) if s is None]
        if unsupported:
            listed = "; ".join(f"“{excerpt(c)}”" for c in unsupported[:_MAX_LISTED])
            more = len(unsupported) - _MAX_LISTED
            reasoning += f" Unsupported: {listed}" + (f" (+{more} more)." if more > 0 else ".")
        return (exact + fuzzy) / len(claims), reasoning

    def _context_recall(
        self, sample: GoldenSample, evidence: Sequence[_Evidence]
    ) -> tuple[float, str]:
        claims = split_claims(sample.ground_truth) or [sample.ground_truth]
        if not evidence:
            return 0.0, "No context was retrieved, so no reference claim is attributable."
        attributable = sum(self._support(c, evidence) is not None for c in claims)
        reasoning = f"{attributable}/{len(claims)} reference claims attributable to the context."
        return attributable / len(claims), reasoning

    # -- relevance ----------------------------------------------------------

    def _answer_relevancy(self, sample: GoldenSample, answer: str) -> tuple[float, str]:
        if not answer.strip():
            return 0.0, "Answer is empty."
        if _NONCOMMITTAL_RE.search(normalize(answer)):
            return 0.0, "Answer is evasive or non-committal."
        question = content_terms(sample.question)
        reply = content_terms(answer)
        reference = content_terms(sample.ground_truth)
        requested = (reference - question) or reference
        informative = coverage(requested, reply) if reply else 0.0
        on_topic = len(question & reply) / len(question) if question else 0.0
        topic = question | reference | frozenset().union(*map(content_terms, sample.contexts))
        focus = len(reply & topic) / len(reply) if reply else 0.0
        alignment = max(informative, on_topic)
        reasoning = (
            f"Answer supplies {informative:.0%} of the requested reference information and "
            f"addresses {on_topic:.0%} of the question terms; {focus:.0%} of its terms stay "
            "on topic."
        )
        return 0.7 * alignment + 0.3 * focus, reasoning

    def _context_precision(
        self, sample: GoldenSample, retrieved: tuple[str, ...]
    ) -> tuple[float, str]:
        if not retrieved:
            return 0.0, "No context was retrieved."
        references = [content_terms(c) for c in sample.contexts]
        truth_claims = split_claims(sample.ground_truth) or [sample.ground_truth]
        relevant_ranks: list[int] = []
        precision_sum = 0.0
        for rank, chunk in enumerate(retrieved, start=1):
            terms = content_terms(chunk)
            evidence = (_Evidence.of(chunk),)
            relevant = any(
                overlap_coefficient(terms, ref) >= self._relevance_threshold for ref in references
            ) or any(self._support(c, evidence) is not None for c in truth_claims)
            if relevant:
                relevant_ranks.append(rank)
                precision_sum += len(relevant_ranks) / rank
        if not relevant_ranks:
            return 0.0, f"None of the {len(retrieved)} retrieved chunks is relevant."
        score = precision_sum / len(relevant_ranks)
        ranks = ", ".join(map(str, relevant_ranks))
        reasoning = (
            f"{len(relevant_ranks)}/{len(retrieved)} retrieved chunks relevant (ranks {ranks}); "
            f"rank-weighted precision {score:.2f}."
        )
        return score, reasoning


# ---------------------------------------------------------------------------
# Structured verdict extraction for LLM judges
# ---------------------------------------------------------------------------

MAX_VERDICT_CHARS: Final = 65_536
_FENCE_RE: Final = re.compile(r"```[ \t]*(?:json)?[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_MAX_REPORTED_ERRORS: Final = 5


class JudgeVerdictError(ValueError):
    """A judge reply that is not exactly one valid, evidence-backed verdict object."""


class JudgeVerdict(EvalModel):
    """One metric verdict as an LLM judge must emit it."""

    metric: MetricType
    aspect: Identifier | None = None
    score: UnitInterval
    reasoning: NonEmptyText
    evidence: tuple[NonEmptyText, ...] = Field(default=(), max_length=16)

    @model_validator(mode="after")
    def _check_aspect(self) -> Self:
        check_aspect(self.metric, self.aspect)
        return self

    @property
    def key(self) -> str:
        """See :func:`~agentic_eval_benchmark.schemas.metric_key`."""
        return metric_key(self.metric, self.aspect)


class JudgeVerdictPayload(EvalModel):
    """The complete JSON document an LLM judge must return."""

    verdicts: tuple[JudgeVerdict, ...] = Field(min_length=1, max_length=32)


def extract_json_object(raw: str) -> str:
    """Return the single JSON object in a judge reply (bare or in one fenced block).

    Prose around the object is tolerated; several candidate objects are rejected as
    ambiguous instead of guessing which one the judge meant.
    """
    if len(raw) > MAX_VERDICT_CHARS:
        raise JudgeVerdictError(f"judge reply exceeds {MAX_VERDICT_CHARS} characters")
    fenced = _FENCE_RE.findall(raw)
    if len(fenced) > 1:
        raise JudgeVerdictError("judge reply contains several fenced blocks; expected one")
    text = fenced[0] if fenced else raw
    start = text.find("{")
    if start < 0:
        raise JudgeVerdictError("judge reply contains no JSON object")
    try:
        _, end = json.JSONDecoder().raw_decode(text, start)
    except json.JSONDecodeError as exc:
        raise JudgeVerdictError(f"judge reply is not valid JSON: {exc.msg}") from exc
    if "{" in text[end:]:
        raise JudgeVerdictError("judge reply contains several JSON objects; expected one")
    return text[start:end]


def _describe(exc: ValidationError) -> str:
    """Field locations and error types only: never echo the judge's content."""
    errors = exc.errors(include_url=False, include_input=False)
    parts = [
        f"{'.'.join(map(str, e['loc'])) or '<root>'}: {e['type']}"
        for e in errors[:_MAX_REPORTED_ERRORS]
    ]
    more = len(errors) - _MAX_REPORTED_ERRORS
    return "; ".join(parts) + (f" (+{more} more)" if more > 0 else "")


def parse_judge_verdict(
    raw: str,
    *,
    expected: Collection[MetricType],
    aspects: Collection[str] = (),
    evidence_sources: Sequence[str] = (),
    require_evidence: bool = False,
    policy: ThresholdPolicy | None = None,
    cost_usd: float | None = None,
    latency_ms: float | None = None,
) -> tuple[MetricScore, ...]:
    """Parse and validate an LLM judge reply into :class:`MetricScore` values.

    The reply must contain exactly one JSON object matching :class:`JudgeVerdictPayload`
    with one verdict per expected metric (and per critique aspect), no duplicates and
    no extras. Every ``evidence`` quote must occur verbatim (case- and punctuation-
    insensitive) in ``evidence_sources``, so a judge cannot justify a score with
    text it invented. ``cost_usd`` and ``latency_ms`` of the judge call are spread
    evenly over the returned scores.
    """
    if MetricType.ASPECT_CRITIQUE in expected and not aspects:
        raise ValueError("aspect_critique is expected but no aspects were given")
    try:
        payload = JudgeVerdictPayload.model_validate_json(extract_json_object(raw))
    except ValidationError as exc:
        raise JudgeVerdictError(f"judge verdict failed validation: {_describe(exc)}") from exc

    wanted = {metric_key(m) for m in expected if m is not MetricType.ASPECT_CRITIQUE}
    if MetricType.ASPECT_CRITIQUE in expected:
        wanted |= {metric_key(MetricType.ASPECT_CRITIQUE, a) for a in aspects}
    keys = [v.key for v in payload.verdicts]
    duplicates = sorted({k for k in keys if keys.count(k) > 1})
    missing = sorted(wanted - set(keys))
    unexpected = sorted(set(keys) - wanted)
    if duplicates or missing or unexpected:
        raise JudgeVerdictError(
            f"judge verdict metrics mismatch: duplicates={duplicates}, missing={missing}, "
            f"unexpected={unexpected}"
        )

    sources = [f" {_flat(s)} " for s in evidence_sources]
    for verdict in payload.verdicts:
        if require_evidence and not verdict.evidence:
            raise JudgeVerdictError(f"{verdict.key}: verdict cites no evidence")
        for quote in verdict.evidence:
            flat = _flat(quote)
            if not flat or not any(f" {flat} " in s for s in sources):
                raise JudgeVerdictError(
                    f"{verdict.key}: evidence not found in the sources: “{excerpt(quote, 60)}”"
                )

    policy = policy or ThresholdPolicy()
    share = len(payload.verdicts)
    return tuple(
        build_score(
            v.metric,
            v.score,
            v.reasoning,
            aspect=v.aspect,
            threshold=None if v.aspect is not None else policy.threshold_for(v.metric),
            cost_usd=None if cost_usd is None else cost_usd / share,
            latency_ms=None if latency_ms is None else latency_ms / share,
        )
        for v in payload.verdicts
    )
