"""Clients for the RAG system under test, implementing ``RAGClientProtocol``."""

import json
import re
from collections.abc import Mapping
from types import TracebackType
from typing import Annotated, Any, Final, Literal, Self

import httpx
from pydantic import AnyHttpUrl, Field, SecretStr, StringConstraints, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from agentic_eval_benchmark.core import RAGResponse, TransientError
from agentic_eval_benchmark.schemas import GoldenDataset, Scalar
from agentic_eval_benchmark.tools._judging import TRANSIENT_STATUS_CODES

FieldPath = Annotated[
    str, StringConstraints(pattern=r"^[A-Za-z0-9_-]+(\.[A-Za-z0-9_-]+)*$", max_length=256)
]
"""Dotted path into a JSON document; integer segments index lists (``choices.0.text``)."""

HeaderName = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9-]{1,64}$")]

_PATH_SEGMENT: Final = re.compile(r"^\d+$")


class RAGClientError(Exception):
    """A non-retryable fault: HTTP 4xx / non-transient 5xx, or a response that does not
    match the configured mapping."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class RAGEndpointConfig(BaseSettings):
    """Request/response mapping for an HTTP RAG endpoint, from ``EVAL_RAG_*`` variables.

    Defaults target the ``rag_documents`` service: ``POST /rag/answer`` with
    ``{"query": ...}`` returning ``{"answer": ..., "sources": [{"preview": ...}]}``.
    Note that ``rag_documents`` only returns a 280-character preview per source, which
    bounds what context-based metrics can verify.
    """

    model_config = SettingsConfigDict(
        env_prefix="EVAL_RAG_", extra="forbid", frozen=True, validate_default=True
    )

    url: AnyHttpUrl = AnyHttpUrl("http://127.0.0.1:8000/rag/answer")
    method: Literal["POST", "GET"] = "POST"
    query_field: Annotated[str, StringConstraints(min_length=1, max_length=128)] = "query"
    """Request key carrying the question (JSON body for POST, query string for GET)."""
    extra_payload: dict[str, Scalar] = Field(default_factory=dict)
    """Constant request fields, e.g. ``{"k": 5}``."""
    answer_path: FieldPath = "answer"
    contexts_path: FieldPath | None = "sources"
    """Path to the list of retrieved contexts; ``None`` when the endpoint returns none."""
    context_text_field: FieldPath | None = "preview"
    """Path to the text inside each context object; plain string items are used as-is."""
    api_key: SecretStr | None = None
    api_key_header: HeaderName = "Authorization"
    api_key_scheme: str | None = "Bearer"
    headers: dict[HeaderName, str] = Field(default_factory=dict)
    timeout_seconds: float = Field(default=30.0, gt=0.0, le=600.0)
    max_response_bytes: int = Field(default=4 * 1024 * 1024, ge=1024, le=64 * 1024 * 1024)
    verify_tls: bool = True


def _lookup(document: object, path: str) -> object:
    current = document
    for segment in path.split("."):
        if isinstance(current, Mapping) and segment in current:
            current = current[segment]
        elif (
            isinstance(current, list)
            and _PATH_SEGMENT.match(segment)
            and int(segment) < len(current)
        ):
            current = current[int(segment)]
        else:
            raise RAGClientError(f"response has no field {path!r}")
    return current


def _reject_constant(token: str) -> float:
    raise RAGClientError(f"response contains non-finite number {token}")


class HttpRAGClient:
    """Async HTTP client for an external RAG endpoint.

    Fault mapping for the engine's retry loop: transport errors and HTTP 408, 429, 500,
    502, 503, 504 and 529 raise :class:`~agentic_eval_benchmark.core.TransientError`; client
    timeouts raise :class:`TimeoutError`; other HTTP errors and responses that do not
    fit the mapping raise :class:`RAGClientError`. Redirects are not followed and the
    body is read with a hard size limit. Inject ``client`` (e.g. with an
    ``httpx.MockTransport``) for hermetic tests; an owned client is closed by
    :meth:`aclose` or ``async with``.
    """

    def __init__(
        self,
        config: RAGEndpointConfig | None = None,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._config = config or RAGEndpointConfig()
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            verify=self._config.verify_tls, follow_redirects=False
        )

    @property
    def config(self) -> RAGEndpointConfig:
        """The endpoint mapping in use."""
        return self._config

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close the HTTP client if this instance created it."""
        if self._owns_client:
            await self._client.aclose()

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json", **self._config.headers}
        if self._config.api_key is not None:
            secret = self._config.api_key.get_secret_value()
            scheme = self._config.api_key_scheme
            headers[self._config.api_key_header] = f"{scheme} {secret}" if scheme else secret
        return headers

    async def query(self, question: str) -> RAGResponse:
        config = self._config
        payload: dict[str, Any] = {**config.extra_payload, config.query_field: question}
        request_kwargs: dict[str, Any] = (
            {"json": payload} if config.method == "POST" else {"params": payload}
        )
        try:
            async with self._client.stream(
                config.method,
                str(config.url),
                headers=self._headers(),
                timeout=httpx.Timeout(config.timeout_seconds),
                **request_kwargs,
            ) as response:
                status = response.status_code
                if status in TRANSIENT_STATUS_CODES:
                    raise TransientError(f"RAG endpoint returned HTTP {status}")
                if response.is_error or response.is_redirect:
                    raise RAGClientError(f"RAG endpoint returned HTTP {status}", status_code=status)
                body = await self._read_limited(response)
        except httpx.TimeoutException as exc:
            raise TimeoutError(f"RAG endpoint timed out ({type(exc).__name__})") from exc
        except httpx.TransportError as exc:
            raise TransientError(f"RAG endpoint unreachable ({type(exc).__name__})") from exc
        return self._parse(body)

    async def _read_limited(self, response: httpx.Response) -> bytes:
        limit = self._config.max_response_bytes
        declared = response.headers.get("Content-Length")
        if declared is not None and declared.isdigit() and int(declared) > limit:
            raise RAGClientError(f"response of {declared} bytes exceeds {limit}")
        chunks: list[bytes] = []
        received = 0
        async for chunk in response.aiter_bytes():
            received += len(chunk)
            if received > limit:
                raise RAGClientError(f"response exceeds {limit} bytes")
            chunks.append(chunk)
        return b"".join(chunks)

    def _parse(self, body: bytes) -> RAGResponse:
        config = self._config
        try:
            document = json.loads(body, parse_constant=_reject_constant)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RAGClientError("RAG endpoint returned a non-JSON body") from exc
        answer = _lookup(document, config.answer_path)
        if not isinstance(answer, str):
            raise RAGClientError(f"field {config.answer_path!r} is not a string")
        contexts: list[str] = []
        if config.contexts_path is not None:
            items = _lookup(document, config.contexts_path)
            if not isinstance(items, list):
                raise RAGClientError(f"field {config.contexts_path!r} is not a list")
            for index, item in enumerate(items):
                text = (
                    _lookup(item, config.context_text_field)
                    if isinstance(item, Mapping) and config.context_text_field
                    else item
                )
                if not isinstance(text, str):
                    raise RAGClientError(f"context {index} has no text")
                contexts.append(text)
        try:
            return RAGResponse(answer=answer, contexts=tuple(contexts))
        except ValidationError as exc:
            raise RAGClientError(
                f"RAG response out of bounds ({exc.error_count()} error(s))"
            ) from exc


class StaticRAGClient:
    """In-memory RAG target answering from canned responses keyed by question.

    :meth:`from_dataset` builds an *oracle* that returns each sample's ground truth and
    reference contexts: a run against it should clear the gate, which makes it the
    smoke test for the evaluation pipeline itself.
    """

    def __init__(self, responses: Mapping[str, RAGResponse]) -> None:
        self._responses = dict(responses)

    @classmethod
    def from_dataset(cls, dataset: GoldenDataset) -> Self:
        """Oracle responses: ground truth as the answer, reference contexts as retrieval."""
        return cls(
            {
                s.question: RAGResponse(answer=s.ground_truth, contexts=s.contexts)
                for s in dataset.samples
            }
        )

    async def query(self, question: str) -> RAGResponse:
        try:
            return self._responses[question]
        except KeyError:
            raise RAGClientError("no canned response for this question") from None
