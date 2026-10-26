"""TruLensJudge over the scripted provider, custom feedback loops and isolated state."""

import asyncio
from pathlib import Path

import nltk
import pytest
from pydantic import BaseModel

from agentic_eval_benchmark.core import HeuristicJudge, TransientError
from agentic_eval_benchmark.schemas import GoldenSample, MetricScore, MetricType
from agentic_eval_benchmark.tools import trulens_adapter
from agentic_eval_benchmark.tools.offline import ScriptedTruLensProvider, chain_of_thought_reply
from agentic_eval_benchmark.tools.trulens_adapter import (
    CustomFeedback,
    FeedbackResult,
    TruLensJudge,
    criteria_feedback,
    open_trulens_session,
)

F, AR = MetricType.FAITHFULNESS, MetricType.ANSWER_RELEVANCY
ANSWER = "Paris is the capital of France. Paris has ten million cats."


@pytest.fixture(autouse=True)
def no_nltk_downloads(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> bool:
        raise AssertionError("nltk.download must never be called")

    monkeypatch.setattr(nltk, "download", forbidden)


def by_key(scores: tuple[MetricScore, ...]) -> dict[str, MetricScore]:
    return {s.key: s for s in scores}


async def test_grounding_and_relevance(
    trulens_provider: ScriptedTruLensProvider, sample: GoldenSample
) -> None:
    result = by_key(
        await TruLensJudge(trulens_provider).evaluate_sample(sample, ANSWER, sample.contexts)
    )
    grounding, relevance = result["faithfulness"], result["answer_relevancy"]
    assert grounding.score == 0.5
    assert not grounding.passed
    assert "1/2 statements supported" in grounding.reasoning
    assert "Unsupported: “Paris has ten million cats.”" in grounding.reasoning
    assert relevance.score == 1.0
    assert "answers the whole prompt" in relevance.reasoning


async def test_custom_feedback_loops(
    trulens_provider: ScriptedTruLensProvider, sample: GoldenSample
) -> None:
    async def tone(sample: GoldenSample, answer: str, contexts: tuple[str, ...]) -> FeedbackResult:
        return 1.0, "polite"

    feedbacks = [
        CustomFeedback("conciseness", lambda s, a, c: (0.25, "too long"), threshold=0.5),
        CustomFeedback("tone", tone),
        CustomFeedback("blank-reason", lambda s, a, c: (0.5, "  ")),
        criteria_feedback("politeness", "The answer is polite.", trulens_provider),
    ]
    judge = TruLensJudge(trulens_provider, metrics=(), feedbacks=feedbacks)
    result = by_key(await judge.evaluate_sample(sample, ANSWER, sample.contexts))
    assert not result["aspect_critique:conciseness"].passed
    assert result["aspect_critique:tone"].score == 1.0
    assert result["aspect_critique:tone"].passed
    assert (
        result["aspect_critique:blank-reason"].reasoning == "Custom feedback blank-reason scored."
    )
    assert result["aspect_critique:politeness"].score == pytest.approx(2 / 3, abs=1e-3)


async def test_feedback_faults(
    trulens_provider: ScriptedTruLensProvider, sample: GoldenSample
) -> None:
    def buggy(s: GoldenSample, a: str, c: tuple[str, ...]) -> FeedbackResult:
        raise KeyError("bug")

    async def slow(s: GoldenSample, a: str, c: tuple[str, ...]) -> FeedbackResult:
        await asyncio.sleep(1)
        return 1.0, "late"

    feedbacks = [
        CustomFeedback("buggy", buggy),
        CustomFeedback("slow", slow),
        CustomFeedback("nan", lambda s, a, c: (float("nan"), "nan")),
    ]
    judge = TruLensJudge(trulens_provider, metrics=(), feedbacks=feedbacks, timeout_seconds=0.05)
    result = by_key(await judge.evaluate_sample(sample, ANSWER, sample.contexts))
    assert "KeyError" in result["aspect_critique:buggy"].reasoning
    assert "timed out" in result["aspect_critique:slow"].reasoning
    assert "invalid score" in result["aspect_critique:nan"].reasoning
    assert not any(s.passed for s in result.values())

    def down(s: GoldenSample, a: str, c: tuple[str, ...]) -> FeedbackResult:
        raise ConnectionError("down")

    with pytest.raises(TransientError, match="down"):
        await TruLensJudge(trulens_provider, feedbacks=[CustomFeedback("c", down)]).evaluate_sample(
            sample, ANSWER, sample.contexts
        )


async def test_unparseable_provider_falls_back(sample: GoldenSample) -> None:
    garbage = ScriptedTruLensProvider(lambda text, fmt: "not json at all")
    plain = by_key(
        await TruLensJudge(garbage).evaluate_sample(sample, sample.ground_truth, sample.contexts)
    )
    assert all(not s.passed and "TruLens" in s.reasoning for s in plain.values())
    rescued = by_key(
        await TruLensJudge(garbage, fallback=HeuristicJudge()).evaluate_sample(
            sample, sample.ground_truth, sample.contexts
        )
    )
    assert all(s.passed and "fallback judge" in s.reasoning for s in rescued.values())


async def test_failing_fallback_is_ignored(sample: GoldenSample) -> None:
    class Broken:
        async def evaluate_sample(
            self, sample: GoldenSample, answer: str, contexts: tuple[str, ...]
        ) -> tuple[MetricScore, ...]:
            raise RuntimeError("down")

    garbage = ScriptedTruLensProvider(lambda text, fmt: "not json")
    result = await TruLensJudge(garbage, fallback=Broken()).evaluate_sample(
        sample, "x y.", sample.contexts
    )
    assert all(not s.passed for s in result)


async def test_provider_faults_become_failure_verdicts(sample: GoldenSample) -> None:
    class RateLimitError(Exception):
        pass

    def limited(text: str, fmt: type[BaseModel] | None) -> str:
        raise RateLimitError("429")

    # The TruLens endpoint retries and re-raises a bare RuntimeError: no TransientError.
    result = by_key(
        await TruLensJudge(ScriptedTruLensProvider(limited)).evaluate_sample(
            sample, "x.", sample.contexts
        )
    )
    assert all(not s.passed and "RuntimeError" in s.reasoning for s in result.values())

    def slow(text: str, fmt: type[BaseModel] | None) -> str:
        import time  # noqa: PLC0415

        time.sleep(0.3)
        return chain_of_thought_reply(3, "late")

    timed_out = by_key(
        await TruLensJudge(
            ScriptedTruLensProvider(slow), metrics=(AR,), timeout_seconds=0.05
        ).evaluate_sample(sample, "x.", sample.contexts)
    )
    assert "timed out" in timed_out["answer_relevancy"].reasoning


async def test_out_of_range_provider_score(
    sample: GoldenSample, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = ScriptedTruLensProvider(lambda text, fmt: chain_of_thought_reply(3, "x"))
    monkeypatch.setattr(
        type(provider), "relevance_with_cot_reasons", lambda self, **kw: (1.7, {"reason": 3})
    )
    result = by_key(
        await TruLensJudge(provider, metrics=(AR,)).evaluate_sample(sample, "x.", sample.contexts)
    )
    assert "invalid score 1.7" in result["answer_relevancy"].reasoning


async def test_missing_inputs(
    trulens_provider: ScriptedTruLensProvider, sample: GoldenSample
) -> None:
    judge = TruLensJudge(trulens_provider)
    empty = by_key(await judge.evaluate_sample(sample, " ", sample.contexts))
    assert empty["faithfulness"].reasoning == "Answer is empty."
    no_context = by_key(await judge.evaluate_sample(sample, ANSWER, ()))
    assert no_context["faithfulness"].reasoning == "No context was retrieved."


def test_reason_payload_parsing() -> None:
    assert trulens_adapter._reason_text("plain") == ""
    assert trulens_adapter._reason_text({"reason": "  a \n b "}) == "a b"
    assert trulens_adapter._statements(
        {"reasons": [{"criteria": "s", "score": 1}, {"criteria": 2}, "x"]}
    ) == [("s", 1.0)]
    assert trulens_adapter._statements(None) == []
    assert TruLensJudge._grounding_reason(0.0, {}) == "TruLens groundedness = 0.000."


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"metrics": (F, F)}, "unique"),
        ({"metrics": (MetricType.CONTEXT_RECALL,)}, "does not score"),
        ({"metrics": (), "feedbacks": ()}, "at least one"),
        (
            {
                "feedbacks": [
                    CustomFeedback("a", lambda s, a, c: (1.0, "")),
                    CustomFeedback("a", lambda s, a, c: (1.0, "")),
                ]
            },
            "unique",
        ),
        ({"timeout_seconds": 0}, "positive"),
    ],
)
def test_configuration(
    trulens_provider: ScriptedTruLensProvider, kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        TruLensJudge(trulens_provider, **kwargs)  # type: ignore[arg-type]


def test_custom_feedback_validation(trulens_provider: ScriptedTruLensProvider) -> None:
    with pytest.raises(ValueError, match="valid identifier"):
        CustomFeedback("bad name", lambda s, a, c: (1.0, ""))
    with pytest.raises(ValueError, match="threshold"):
        CustomFeedback("ok", lambda s, a, c: (1.0, ""), threshold=1.5)
    with pytest.raises(ValueError, match="criteria"):
        criteria_feedback("x", "   ", trulens_provider)


def test_isolated_sqlite_session(tmp_path: Path) -> None:
    home = tmp_path / "trulens-state"
    session = open_trulens_session(home)
    assert (home / trulens_adapter.DATABASE_FILE).is_file()
    assert not (tmp_path / "default.sqlite").exists()
    try:
        assert open_trulens_session(home) is session
        with pytest.raises(RuntimeError, match="already open"):
            open_trulens_session(tmp_path / "elsewhere")
    finally:
        session.connector.db.engine.dispose()  # type: ignore[union-attr]
