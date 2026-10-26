"""HTTP RAG client over a mock transport, and the in-memory StaticRAGClient oracle."""

import json
from collections.abc import AsyncIterator, Callable

import httpx
import pytest

from agentic_eval_benchmark.core import RAGClientProtocol, TransientError
from agentic_eval_benchmark.schemas import GoldenDataset
from agentic_eval_benchmark.tools import (
    HttpRAGClient,
    RAGClientError,
    RAGEndpointConfig,
    StaticRAGClient,
)

URL = "http://rag.test/rag/answer"
SECRET = "sk-secret-123"

Handler = Callable[[httpx.Request], httpx.Response]


def rag_documents_reply(request: httpx.Request) -> httpx.Response:
    query = (
        json.loads(request.content)["query"]
        if request.method == "POST"
        else request.url.params["query"]
    )
    return httpx.Response(
        200,
        json={
            "query": query,
            "answer": "Paris.",
            "sources": [{"preview": "Paris is the capital.", "source": "a.pdf"}, "plain context"],
            "model": "m",
        },
    )


class Recorder:
    def __init__(self, handler: Handler) -> None:
        self.handler = handler
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.handler(request)


@pytest.fixture
async def transport_client() -> AsyncIterator[
    Callable[[Handler], tuple[httpx.AsyncClient, Recorder]]
]:
    clients: list[httpx.AsyncClient] = []

    def build(handler: Handler) -> tuple[httpx.AsyncClient, Recorder]:
        recorder = Recorder(handler)
        client = httpx.AsyncClient(transport=httpx.MockTransport(recorder))
        clients.append(client)
        return client, recorder

    yield build
    for client in clients:
        await client.aclose()


def config(**overrides: object) -> RAGEndpointConfig:
    fields: dict[str, object] = {
        "url": URL,
        "api_key": SECRET,
        "extra_payload": {"k": 5},
        "max_response_bytes": 1024,
    }
    return RAGEndpointConfig.model_validate(fields | overrides)


async def test_post_payload_auth_and_rag_documents_mapping(
    transport_client: Callable[[Handler], tuple[httpx.AsyncClient, Recorder]],
) -> None:
    client, recorder = transport_client(rag_documents_reply)
    rag = HttpRAGClient(config(), client=client)
    response = await rag.query("What is the capital?")
    assert response.answer == "Paris."
    assert response.contexts == ("Paris is the capital.", "plain context")
    request = recorder.requests[-1]
    assert json.loads(request.content) == {"k": 5, "query": "What is the capital?"}
    assert request.headers["authorization"] == f"Bearer {SECRET}"
    assert request.headers["accept"] == "application/json"
    assert isinstance(rag, RAGClientProtocol)
    assert rag.config.url.host == "rag.test"


async def test_get_mode_nested_paths_and_raw_key(
    transport_client: Callable[[Handler], tuple[httpx.AsyncClient, Recorder]],
) -> None:
    def nested(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"text": "A1", "docs": ["c1", {"body": "c2"}]}]})

    client, recorder = transport_client(nested)
    rag = HttpRAGClient(
        config(
            method="GET",
            answer_path="data.0.text",
            contexts_path="data.0.docs",
            context_text_field="body",
            api_key_header="X-Api-Key",
            api_key_scheme=None,
            headers={"X-Tenant": "t1"},
        ),
        client=client,
    )
    response = await rag.query("nested?")
    request = recorder.requests[-1]
    assert (response.answer, response.contexts) == ("A1", ("c1", "c2"))
    assert request.method == "GET"
    assert request.url.params["k"] == "5"
    assert request.headers["x-api-key"] == SECRET
    assert request.headers["x-tenant"] == "t1"


async def test_endpoint_without_contexts(
    transport_client: Callable[[Handler], tuple[httpx.AsyncClient, Recorder]],
) -> None:
    client, _ = transport_client(lambda r: httpx.Response(200, json={"answer": "x"}))
    rag = HttpRAGClient(config(contexts_path=None, api_key=None), client=client)
    assert (await rag.query("q")).contexts == ()


def _raise(exc: type[httpx.RequestError]) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        raise exc("boom", request=request)

    return handler


@pytest.mark.parametrize(
    ("handler", "error", "message"),
    [
        (lambda r: httpx.Response(503), TransientError, "HTTP 503"),
        (lambda r: httpx.Response(429), TransientError, "HTTP 429"),
        (_raise(httpx.ConnectError), TransientError, "unreachable"),
        (_raise(httpx.ReadTimeout), TimeoutError, "timed out"),
        (lambda r: httpx.Response(404, json={"error": "nope"}), RAGClientError, "HTTP 404"),
        (lambda r: httpx.Response(501), RAGClientError, "HTTP 501"),
        (
            lambda r: httpx.Response(302, headers={"Location": "http://elsewhere/"}),
            RAGClientError,
            "HTTP 302",
        ),
        (lambda r: httpx.Response(200, text="<html>"), RAGClientError, "non-JSON"),
        (lambda r: httpx.Response(200, content=b'{"answer": NaN}'), RAGClientError, "non-finite"),
        (
            lambda r: httpx.Response(200, content=b'{"answer": "' + b"a" * 5000 + b'"}'),
            RAGClientError,
            "exceeds 1024",
        ),
        (lambda r: httpx.Response(200, json={"result": "x"}), RAGClientError, "no field 'answer'"),
        (lambda r: httpx.Response(200, json={"answer": 3}), RAGClientError, "not a string"),
        (
            lambda r: httpx.Response(200, json={"answer": "x", "sources": {}}),
            RAGClientError,
            "not a list",
        ),
        (
            lambda r: httpx.Response(200, json={"answer": "x", "sources": [3]}),
            RAGClientError,
            "context 0",
        ),
        (
            lambda r: httpx.Response(200, json={"answer": "x", "sources": ["c"] * 300}),
            RAGClientError,
            "out of bounds",
        ),
    ],
)
async def test_fault_mapping_never_leaks_the_key(
    transport_client: Callable[[Handler], tuple[httpx.AsyncClient, Recorder]],
    handler: Handler,
    error: type[Exception],
    message: str,
) -> None:
    client, _ = transport_client(handler)
    rag = HttpRAGClient(
        config(max_response_bytes=4096 if message == "out of bounds" else 1024), client=client
    )
    with pytest.raises(error, match=message) as raised:
        await rag.query("q")
    assert SECRET not in str(raised.value)


async def test_declared_content_length_is_checked_before_reading(
    transport_client: Callable[[Handler], tuple[httpx.AsyncClient, Recorder]],
) -> None:
    client, _ = transport_client(
        lambda r: httpx.Response(200, headers={"Content-Length": "999999"}, content=b"{}")
    )
    with pytest.raises(RAGClientError, match="999999 bytes"):
        await HttpRAGClient(config(), client=client).query("q")


async def test_owned_client_lifecycle() -> None:
    async with HttpRAGClient(config()) as rag:
        assert rag.config.timeout_seconds == 30.0
    injected = httpx.AsyncClient()
    await HttpRAGClient(config(), client=injected).aclose()
    assert not injected.is_closed
    await injected.aclose()


def test_endpoint_config_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVAL_RAG_URL", "http://env.test/answer")
    monkeypatch.setenv("EVAL_RAG_EXTRA_PAYLOAD", '{"k": 3}')
    loaded = RAGEndpointConfig()
    assert str(loaded.url) == "http://env.test/answer"
    assert loaded.extra_payload == {"k": 3}
    assert RAGEndpointConfig.model_fields["answer_path"].default == "answer"


async def test_static_oracle(reference_dataset: GoldenDataset) -> None:
    oracle = StaticRAGClient.from_dataset(reference_dataset)
    first = reference_dataset.samples[0]
    response = await oracle.query(first.question)
    assert (response.answer, response.contexts) == (first.ground_truth, first.contexts)
    with pytest.raises(RAGClientError, match="no canned response"):
        await oracle.query("unknown?")
