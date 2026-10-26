"""Offline isolation, fault classification, in-memory backends and the lazy tools package."""

import os
import socket
from pathlib import Path

import httpx
import pytest
from pydantic import BaseModel
from pytest_socket import SocketBlockedError

from agentic_eval_benchmark import tools
from agentic_eval_benchmark.core import TransientError
from agentic_eval_benchmark.schemas import MetricScore, MetricType
from agentic_eval_benchmark.tools import _isolation, _judging
from agentic_eval_benchmark.tools.offline import (
    HashingEmbeddings,
    ScriptedRagasLLM,
    ScriptedTruLensProvider,
    chain_of_thought_reply,
)


def test_sockets_are_blocked() -> None:
    with pytest.warns(UserWarning, match="tried to use socket"), pytest.raises(SocketBlockedError):
        socket.socket(socket.AF_INET, socket.SOCK_STREAM)


def test_state_directories_follow_the_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    assert _isolation.cache_root() == tmp_path / "cache"
    assert _isolation.trulens_home() == tmp_path / "trulens"
    monkeypatch.delenv("CACHE_ROOT")
    monkeypatch.delenv("TRULENS_HOME")
    assert _isolation.cache_root() == tmp_path / ".cache"
    assert _isolation.trulens_home() == tmp_path / ".cache" / "trulens"
    monkeypatch.delenv("XDG_DATA_HOME")
    assert _isolation.pin_ragas_offline() == tmp_path / ".cache" / "xdg-data" / "ragas"
    assert os.environ["RAGAS_DO_NOT_TRACK"] == "true"
    assert _isolation.pin_trulens_offline().is_dir()


class _StatusError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class APIConnectionError(Exception):
    pass


def _caused_by(outer: Exception, inner: Exception) -> Exception:
    outer.__cause__ = inner
    return outer


@pytest.mark.parametrize(
    ("exc", "transient", "timeout"),
    [
        (ConnectionError("x"), True, False),
        (httpx.ConnectError("x"), True, False),
        (APIConnectionError("x"), True, False),
        (_StatusError(429), True, False),
        (_StatusError(529), True, False),
        (_StatusError(400), False, False),
        (TimeoutError(), False, True),
        (httpx.ReadTimeout("x"), False, True),
        (_caused_by(RuntimeError("wrapper"), _StatusError(503)), True, False),
        (_caused_by(RuntimeError("wrapper"), TimeoutError()), False, True),
        (ValueError("parse"), False, False),
    ],
)
def test_fault_classification(exc: Exception, transient: bool, timeout: bool) -> None:
    assert _judging.is_transient(exc) is transient
    assert _judging.is_timeout(exc) is timeout


def test_cause_cycles_terminate() -> None:
    a, b = ValueError("a"), ValueError("b")
    a.__cause__, b.__cause__ = b, a
    assert not _judging.is_transient(a)


def test_describe_and_failure_verdict() -> None:
    assert _judging.describe(ValueError()) == "ValueError"
    assert _judging.describe(ValueError("a\n  b")) == "ValueError: a b"
    assert len(_judging.describe(ValueError("x" * 1000))) == 300
    verdict = _judging.failure_verdict(MetricType.FAITHFULNESS, "failed")
    assert (verdict.score, verdict.passed, verdict.threshold) == (0.0, False, None)


async def test_run_metrics_surfaces_transient_and_other_errors() -> None:
    async def ok() -> MetricScore:
        return _judging.failure_verdict(MetricType.FAITHFULNESS, "x")

    async def transient() -> MetricScore:
        raise TransientError("again")

    async def bug() -> MetricScore:
        raise KeyError("bug")

    assert len(await _judging.run_metrics([ok(), ok()])) == 2
    with pytest.raises(TransientError, match="again"):
        await _judging.run_metrics([ok(), transient()])
    with pytest.raises(ExceptionGroup):
        await _judging.run_metrics([bug()])


class Answer(BaseModel):
    text: str


def test_scripted_ragas_llm() -> None:
    llm = ScriptedRagasLLM.from_payloads({"Answer": [{"text": "a"}, Answer(text="b")]})
    assert [llm.generate("p", Answer).text for _ in range(3)] == ["a", "b", "a"]
    assert llm.calls[0] == ("Answer", "p")
    with pytest.raises(KeyError, match="no scripted payload"):
        ScriptedRagasLLM.from_payloads({}).generate("p", Answer)
    direct = ScriptedRagasLLM(lambda prompt, model: Answer(text=prompt))
    assert direct.generate("hi", Answer).text == "hi"


async def test_hashing_embeddings() -> None:
    embeddings = HashingEmbeddings(dimensions=64)
    a = await embeddings.aembed_text("Paris is the capital")
    b = embeddings.embed_text("the capital is Paris")
    assert len(a) == 64
    assert sum(x * y for x, y in zip(a, b, strict=True)) == pytest.approx(1.0)
    assert embeddings.embed_text("") == [0.0] * 64
    with pytest.raises(ValueError, match="dimensions"):
        HashingEmbeddings(dimensions=4)


def test_scripted_trulens_provider_flattens_messages() -> None:
    seen: list[str] = []

    def respond(text: str, fmt: type[BaseModel] | None) -> str:
        seen.append(text)
        return "ok"

    provider = ScriptedTruLensProvider(respond)
    assert provider._create_chat_completion(prompt="p", messages=[{"content": "m"}, {}]) == "ok"
    assert seen == ["p\n\nm\n\n"]
    assert '"score": 2' in chain_of_thought_reply(2, "why")


def test_lazy_tools_package() -> None:
    assert tools.load_reference_dataset is tools.load_reference_dataset
    assert "RagasJudge" in dir(tools)
    with pytest.raises(AttributeError, match="no attribute 'nope'"):
        _ = tools.nope
