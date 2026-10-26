"""Text primitives, the deterministic HeuristicJudge and the strict LLM verdict parser."""

import re

import pytest

from agentic_eval_benchmark.core import (
    HeuristicJudge,
    JudgeProtocol,
    JudgeVerdictError,
    build_score,
    extract_json_object,
    parse_judge_verdict,
)
from agentic_eval_benchmark.core._text import (
    content_terms,
    coverage,
    excerpt,
    is_negated,
    normalize,
    numbers,
    overlap_coefficient,
    split_claims,
    stem,
    tokenize,
)
from agentic_eval_benchmark.schemas import GoldenSample, MetricScore, MetricType

F, AR = MetricType.FAITHFULNESS, MetricType.ANSWER_RELEVANCY
CP, CR = MetricType.CONTEXT_PRECISION, MetricType.CONTEXT_RECALL
CONTEXT = "Paris is the capital of France. It has a population of 2.1 million people."
DISTRACTOR = "The Eiffel Tower was completed in 1889."

# ---------------------------------------------------------------------------
# Text primitives
# ---------------------------------------------------------------------------


def test_normalize_and_tokenize() -> None:
    assert normalize("  Ｐａｒｉｓ\n\tIS   “Big” ") == 'paris is "big"'
    assert tokenize("It costs 1,000 EUR (3.5%)") == ["it", "costs", "1000", "eur", "3.5"]
    assert numbers("Opened 1889, rebuilt 2019.") == {"1889", "2019"}


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("cities", "city"),
        ("replicated", "replicat"),
        ("running", "runn"),
        ("is", "is"),
        ("2019", "2019"),
    ],
)
def test_stem(token: str, expected: str) -> None:
    assert stem(token) == expected


def test_content_terms_drop_stopwords_and_negations() -> None:
    assert content_terms("What is not the capital of France?") == {"capital", "france"}
    assert "won" in content_terms("Who won the cup?")


@pytest.mark.parametrize(
    ("text", "negated"),
    [
        ("Paris is not the capital", True),
        ("Paris isn't the capital", True),
        ("Paris isn’t the capital", True),
        ("You can't recover it", True),
        ("No backups exist", True),
        ("Paris is the capital", False),
        ("You can recover it", False),
    ],
)
def test_polarity_cues(text: str, negated: bool) -> None:
    assert is_negated(text) is negated


def test_split_claims() -> None:
    text = "First claim. Second claim! Third; fourth?\n- bullet five\n1. numbered six\n\n   ."
    assert split_claims(text) == [
        "First claim", "Second claim", "Third", "fourth", "bullet five", "numbered six",
    ]  # fmt: skip
    assert split_claims("The value is 3.5 percent.") == ["The value is 3.5 percent"]
    assert split_claims("No.") == []
    assert split_claims("1. Create a key.\n2. Deploy it.") == ["Create a key", "Deploy it"]


def test_similarity_helpers() -> None:
    assert coverage(frozenset(), frozenset({"a"})) == 1.0
    assert coverage(frozenset({"a", "b"}), frozenset({"a"})) == 0.5
    assert overlap_coefficient(frozenset(), frozenset({"a"})) == 0.0
    assert overlap_coefficient(frozenset({"a", "b"}), frozenset({"a", "b", "c", "d"})) == 1.0
    assert excerpt("short") == "short"
    assert excerpt("x" * 100, limit=10) == "x" * 9 + "…"


# ---------------------------------------------------------------------------
# HeuristicJudge
# ---------------------------------------------------------------------------


def scores(
    judge: HeuristicJudge, sample: GoldenSample, answer: str, contexts: tuple[str, ...] = (CONTEXT,)
) -> dict[MetricType, MetricScore]:
    return {s.metric: s for s in judge.score(sample, answer, contexts)}


def test_heuristic_judge_rewards_a_grounded_answer(
    judge: HeuristicJudge, sample: GoldenSample
) -> None:
    result = scores(judge, sample, "Paris is the capital of France.")
    assert result[F].score == 1.0
    assert "1 exact" in result[F].reasoning
    assert result[AR].score >= 0.85
    assert result[CP].score == 1.0
    assert result[CR].score == 1.0
    assert all(s.passed and s.cost_usd == 0.0 and s.latency_ms is not None for s in result.values())
    assert isinstance(judge, JudgeProtocol)
    assert judge.metrics == (F, AR, CP, CR)


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("France's capital city is Paris.", 1.0),
        ("Paris is not the capital of France.", 0.0),
        ("Paris isn't the capital of France.", 0.0),
        ("Paris has a population of 3.5 million people.", 0.0),
        ("Paris is the capital of France. It was founded by aliens in 3000 BC.", 0.5),
    ],
)
def test_heuristic_faithfulness(
    judge: HeuristicJudge, sample: GoldenSample, answer: str, expected: float
) -> None:
    assert scores(judge, sample, answer)[F].score == expected


def test_unsupported_claims_are_listed_and_capped(
    judge: HeuristicJudge, sample: GoldenSample
) -> None:
    answer = " ".join(f"Unicorns number {i} live on Mars." for i in range(5))
    reasoning = scores(judge, sample, answer)[F].reasoning
    assert "0/5 answer claims" in reasoning
    assert "(+2 more)." in reasoning


def test_heuristic_relevancy(judge: HeuristicJudge, sample: GoldenSample) -> None:
    assert scores(judge, sample, "Paris.")[AR].score >= 0.85
    assert scores(judge, sample, "I don't know.")[AR].score == 0.0
    assert (
        scores(judge, sample, "I don’t know.")[AR].reasoning
        == "Answer is evasive or non-committal."
    )
    assert scores(judge, sample, "Bananas are yellow and rich in potassium.")[AR].score < 0.3
    wrong_but_on_topic = scores(judge, sample, "The capital of France is Lyon.")
    assert wrong_but_on_topic[AR].score >= 0.85
    assert wrong_but_on_topic[F].score == 0.0


def test_heuristic_context_precision_is_rank_weighted(
    judge: HeuristicJudge, sample: GoldenSample
) -> None:
    second = scores(judge, sample, "Paris is the capital of France.", (DISTRACTOR, CONTEXT))
    assert second[CP].score == 0.5
    assert "ranks 2" in second[CP].reasoning
    none = scores(judge, sample, "Paris is the capital of France.", (DISTRACTOR,))
    assert none[CP].score == 0.0
    assert none[CR].score == 0.0


def test_heuristic_handles_missing_inputs(judge: HeuristicJudge, sample: GoldenSample) -> None:
    empty_context = scores(judge, sample, "Paris is the capital of France.", ())
    assert empty_context[F].score == empty_context[CP].score == empty_context[CR].score == 0.0
    assert not empty_context[F].passed
    empty_answer = scores(judge, sample, "")
    assert empty_answer[F].reasoning == "Answer contains no verifiable claims."
    assert empty_answer[AR].reasoning == "Answer is empty."


def test_heuristic_judge_is_deterministic(judge: HeuristicJudge, sample: GoldenSample) -> None:
    first = judge.score(sample, "Paris.", (CONTEXT,))
    second = judge.score(sample, "Paris.", (CONTEXT,))
    assert [s.score for s in first] == [s.score for s in second]


async def test_heuristic_async_entry_point(judge: HeuristicJudge, sample: GoldenSample) -> None:
    result = await judge.evaluate_sample(sample, "Paris is the capital of France.", (CONTEXT,))
    assert len(result) == 4


def test_heuristic_reference_dataset_oracle_passes(judge: HeuristicJudge) -> None:
    from agentic_eval_benchmark.tools import load_reference_dataset  # noqa: PLC0415

    for golden in load_reference_dataset().samples:
        result = judge.score(golden, golden.ground_truth, golden.contexts)
        assert all(s.passed for s in result), (golden.sample_id, result)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"metrics": ()}, "at least one"),
        ({"metrics": (F, F)}, "unique"),
        ({"metrics": (MetricType.ASPECT_CRITIQUE,)}, "aspect_critique"),
        ({"support_threshold": 0.0}, "support_threshold"),
        ({"relevance_threshold": 1.5}, "relevance_threshold"),
    ],
)
def test_heuristic_judge_configuration(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        HeuristicJudge(**kwargs)  # type: ignore[arg-type]


def test_build_score_clamps_and_derives_passed() -> None:
    assert build_score(F, 1.2, "r", threshold=0.85).score == 1.0
    assert not build_score(F, -0.5, "r", threshold=0.85).passed
    assert build_score(F, 0.0, "r", threshold=None).passed


# ---------------------------------------------------------------------------
# parse_judge_verdict / extract_json_object
# ---------------------------------------------------------------------------

SOURCES = ("Paris is the capital of France.",)


def verdicts(*items: str) -> str:
    body = ",".join(items or ('{"metric":"faithfulness","score":0.9,"reasoning":"ok"}',))
    return '{"verdicts":[' + body + "]}"


def test_parse_valid_verdict_with_evidence_and_cost() -> None:
    raw = (
        "Sure! Here is my verdict:\n```json\n"
        + verdicts(
            '{"metric":"faithfulness","score":0.9,"reasoning":"ok",'
            '"evidence":["paris is the CAPITAL of france"]}',
            '{"metric":"answer_relevancy","score":0.8,"reasoning":"ok"}',
        )
        + "\n```"
    )
    result = parse_judge_verdict(
        raw, expected=(F, AR), evidence_sources=SOURCES, cost_usd=0.02, latency_ms=10.0
    )
    assert [s.passed for s in result] == [True, False]
    assert result[0].cost_usd == pytest.approx(0.01)
    assert result[0].latency_ms == pytest.approx(5.0)


def test_parse_aspect_critique_is_informational() -> None:
    raw = verdicts(
        '{"metric":"aspect_critique","aspect":"harmfulness","score":0,"reasoning":"benign"}'
    )
    (critique,) = parse_judge_verdict(
        raw, expected=(MetricType.ASPECT_CRITIQUE,), aspects=("harmfulness",)
    )
    assert critique.aspect == "harmfulness"
    assert critique.passed
    assert critique.threshold is None


@pytest.mark.parametrize(
    ("raw", "kwargs"),
    [
        ("the answer is great", {}),
        (verdicts() + " and " + verdicts(), {}),
        (f"```json\n{verdicts()}\n```\n```json\n{verdicts()}\n```", {}),
        (verdicts('{"metric":"faithfulness","score":1.5,"reasoning":"ok"}'), {}),
        (verdicts('{"metric":"faithfulness","score":NaN,"reasoning":"ok"}'), {}),
        (verdicts('{"metric":"faithfulness","score":"0.9","reasoning":"ok"}'), {}),
        (verdicts('{"metric":"faithfulness","score":0.9,"reasoning":"ok","confidence":1}'), {}),
        (verdicts('{"metric":"toxicity","score":0.9,"reasoning":"ok"}'), {}),
        (verdicts('{"metric":"faithfulness","score":0.9,"reasoning":" "}'), {}),
        (verdicts('{"metric":"aspect_critique","score":0.9,"reasoning":"no aspect"}'), {}),
        (verdicts(), {"expected": (F, CR)}),
        (verdicts(*['{"metric":"faithfulness","score":0.9,"reasoning":"ok"}'] * 2), {}),
        (
            verdicts(
                '{"metric":"faithfulness","score":0.9,"reasoning":"ok",'
                '"evidence":["Paris has 9 million people"]}'
            ),
            {},
        ),
        (verdicts('{"metric":"faithfulness","score":0.9,"reasoning":"ok","evidence":["..."]}'), {}),
        (verdicts(), {"require_evidence": True}),
        (verdicts() + " " * 70_000, {}),
        ('{"verdicts": [', {}),
    ],
)
def test_parse_rejects_untrustworthy_replies(raw: str, kwargs: dict[str, object]) -> None:
    options: dict[str, object] = {"expected": (F,), "evidence_sources": SOURCES} | kwargs
    with pytest.raises(JudgeVerdictError):
        parse_judge_verdict(raw, **options)  # type: ignore[arg-type]


def test_parse_reports_field_locations_not_content() -> None:
    raw = verdicts(*['{"metric":"faithfulness","score":7,"reasoning":"SECRET"}'] * 6)
    with pytest.raises(JudgeVerdictError) as error:
        parse_judge_verdict(raw, expected=(F,))
    assert "SECRET" not in str(error.value)
    assert re.search(r"\(\+\d+ more\)", str(error.value))


def test_parse_requires_aspects_for_critiques() -> None:
    with pytest.raises(ValueError, match="no aspects"):
        parse_judge_verdict(verdicts(), expected=(MetricType.ASPECT_CRITIQUE,))


def test_extract_json_object_tolerates_prose() -> None:
    assert extract_json_object('verdict: {"a": 1} thanks') == '{"a": 1}'
