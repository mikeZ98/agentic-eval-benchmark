"""FastAPI service: endpoints, RFC 7807 problems, zero leakage, limits, isolation, services."""

import asyncio
import json
import sys
import types
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, Protocol

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr, ValidationError
from starlette.types import Message, Receive, Scope, Send

from agentic_eval_benchmark.api import (
    PROBLEM_MEDIA_TYPE,
    ApiSettings,
    BusyError,
    ConfigurationError,
    EvaluationServices,
    InvalidInputError,
    NotFoundError,
    ProviderSettings,
    RunStore,
    StorageError,
    create_app,
    make_problem,
)
from agentic_eval_benchmark.api import app as app_module
from agentic_eval_benchmark.api.middleware import RequestContextMiddleware, _BodyTooLargeError
from agentic_eval_benchmark.core import BenchmarkSettings
from agentic_eval_benchmark.schemas import GoldenDataset, ThresholdPolicy
from conftest import API_KEY, MakeRun, RagRecorder


class ResponseLike(Protocol):
    """Starlette's TestClient returns ``httpx2`` responses; only this surface is used."""

    @property
    def status_code(self) -> int: ...

    @property
    def headers(self) -> Mapping[str, str]: ...

    def json(self) -> Any: ...


def is_problem(response: ResponseLike, status: int, kind: str) -> bool:
    body = response.json()
    return (
        response.status_code == status
        and response.headers["content-type"] == PROBLEM_MEDIA_TYPE
        and response.headers["cache-control"] == "no-store"
        and body["type"].endswith(f":{kind}")
        and body["status"] == status
    )


# ---------------------------------------------------------------------------
# Health and auth
# ---------------------------------------------------------------------------


def test_health_is_open_and_reports_pinned_paths(api_client: TestClient, tmp_path: Path) -> None:
    response = api_client.get("/v1/eval/health")
    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "ok"
    assert body["storage"]["output_dir"] == str(tmp_path / "runs")
    assert body["storage"]["trulens_home"] == str(tmp_path / "trulens")
    assert set(body["judges"]) == {"heuristic", "ragas", "trulens"}
    assert not body["judges"]["ragas"]["configured"]
    assert "x-request-id" in response.headers


def test_health_degrades_when_storage_is_unwritable(tmp_path: Path) -> None:
    blocker = tmp_path / "blocked"
    blocker.write_text("a file, not a directory")
    services = EvaluationServices(BenchmarkSettings(output_dir=blocker / "runs"))
    with TestClient(create_app(services)) as client:
        response = client.get("/v1/eval/health")
    assert response.status_code == 503
    assert response.json()["status"] == "degraded"


@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "wrong"}, {"Authorization": "Basic abc"}])
def test_protected_endpoints_require_the_key(
    api_client: TestClient, headers: dict[str, str]
) -> None:
    response = api_client.post("/v1/eval/run", json={"oracle": True}, headers=headers)
    assert is_problem(response, 401, "unauthorized")
    assert response.headers["www-authenticate"] == "Bearer"
    assert is_problem(api_client.get("/v1/eval/report/x", headers=headers), 401, "unauthorized")


def test_api_key_strength() -> None:
    with pytest.raises(ValidationError, match="32 characters"):
        ApiSettings(api_key=SecretStr("short"))


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


def test_oracle_run_and_report_round_trip(api_client: TestClient, auth: dict[str, str]) -> None:
    response = api_client.post(
        "/v1/eval/run",
        json={"oracle": True, "run_id": "api-1"},
        headers={"X-API-Key": API_KEY, "X-Request-ID": "req-123"},
    )
    summary = response.json()
    assert response.status_code == 201
    assert response.headers["x-eval-gate"] == "passed"
    assert response.headers["x-eval-sample-errors"] == "0"
    assert response.headers["location"] == "/v1/eval/report/api-1"
    assert response.headers["x-request-id"] == "req-123"
    assert summary["run_id"] == "api-1"
    assert "evaluations" not in summary
    report = api_client.get("/v1/eval/report/api-1", headers=auth).json()
    assert report["summary"] == summary
    assert len(report["evaluations"]) == 8
    assert api_client.get("/v1/eval/report/api-1/summary", headers=auth).json() == summary
    assert is_problem(
        api_client.post("/v1/eval/run", json={"oracle": True, "run_id": "api-1"}, headers=auth),
        409,
        "conflict",
    )


def test_dataset_sources(
    api_client: TestClient, auth: dict[str, str], reference_dataset: GoldenDataset
) -> None:
    by_path = api_client.post(
        "/v1/eval/run",
        json={
            "oracle": True,
            "dataset_path": "ref.json",
            "policy": {"faithfulness_min": 0.9, "relevancy_min": 0.9},
        },
        headers=auth,
    )
    assert by_path.status_code == 201
    assert by_path.json()["policy"]["faithfulness_min"] == 0.9
    inline = reference_dataset.model_dump(mode="json") | {"name": "inline"}
    uploaded = api_client.post(
        "/v1/eval/run", json={"oracle": True, "dataset": inline}, headers=auth
    )
    assert uploaded.status_code == 201
    assert uploaded.json()["dataset_name"] == "inline"


@pytest.mark.parametrize(
    ("path", "status"),
    [("../outside.json", 404), ("/etc/passwd", 422), ("nope.json", 404), ("  ", 422)],
)
def test_dataset_path_is_confined(
    api_client: TestClient, auth: dict[str, str], path: str, status: int
) -> None:
    response = api_client.post(
        "/v1/eval/run", json={"oracle": True, "dataset_path": path}, headers=auth
    )
    assert response.status_code == status
    assert "outside" not in response.text
    assert "passwd" not in response.text


def test_dataset_path_disabled_without_root(services: EvaluationServices) -> None:
    services.api = ApiSettings()
    with TestClient(create_app(services)) as client:
        response = client.post("/v1/eval/run", json={"oracle": True, "dataset_path": "ref.json"})
    assert is_problem(response, 403, "forbidden")


def test_invalid_dataset_in_a_file(
    api_client: TestClient, auth: dict[str, str], dataset_dir: Path
) -> None:
    (dataset_dir / "bad.json").write_text(
        '[{"question": "Hi", "contexts": ["SECRET"], "ground_truth": "b"}]'
    )
    response = api_client.post(
        "/v1/eval/run", json={"oracle": True, "dataset_path": "bad.json"}, headers=auth
    )
    assert is_problem(response, 422, "invalid-input")
    assert "SECRET" not in response.text


def test_validation_errors_never_echo_content(
    api_client: TestClient, auth: dict[str, str], reference_dataset: GoldenDataset
) -> None:
    inline = reference_dataset.model_dump(mode="json")
    inline["samples"] = [inline["samples"][0] | {"question": "SECRET-QUESTION?", "contexts": []}]
    response = api_client.post(
        "/v1/eval/run", json={"oracle": True, "dataset": inline}, headers=auth
    )
    assert is_problem(response, 422, "invalid-input")
    assert "SECRET" not in response.text
    assert "dataset.samples.0.contexts" in response.json()["detail"]


@pytest.mark.parametrize(
    ("body", "status", "kind", "fragment"),
    [
        ({"oracle": True, "unexpected": 1}, 422, "invalid-input", "Extra inputs are not permitted"),
        (
            {"oracle": True, "policy": {"faithfulness_min": 0.5}},
            422,
            "invalid-input",
            "strict_mode",
        ),
        ({"oracle": True, "rag_url": "http://rag.internal/x"}, 422, "invalid-input", "oracle"),
        ({"dataset_path": "ref.json", "dataset": {}}, 422, "invalid-input", "dataset"),
        ({"rag_url": "http://169.254.169.254/latest"}, 403, "forbidden", "ALLOWED_RAG_HOSTS"),
        ({"rag_url": "not a url"}, 422, "invalid-input", "absolute"),
        ({"oracle": True, "judge": "ragas"}, 503, "misconfigured", "EVAL_PROVIDER_LLM_MODEL"),
        ({"oracle": True, "baseline_run_id": "missing"}, 404, "not-found", "not found"),
    ],
)
def test_run_request_problems(
    api_client: TestClient,
    auth: dict[str, str],
    body: dict[str, Any],
    status: int,
    kind: str,
    fragment: str,
) -> None:
    response = api_client.post("/v1/eval/run", json=body, headers=auth)
    assert is_problem(response, status, kind), response.text
    assert fragment in response.json()["detail"]
    assert "169.254" not in response.text


def test_malformed_and_wrong_media_type(api_client: TestClient, auth: dict[str, str]) -> None:
    json_headers = auth | {"content-type": "application/json"}
    assert is_problem(
        api_client.post("/v1/eval/run", content=b"{not json", headers=json_headers),
        422,
        "invalid-input",
    )
    form = auth | {"content-type": "application/x-www-form-urlencoded"}
    assert is_problem(
        api_client.post("/v1/eval/run", content=b"q=1", headers=form), 415, "unsupported-media-type"
    )


def test_request_body_limits(api_client: TestClient, auth: dict[str, str]) -> None:
    headers = auth | {"content-type": "application/json"}
    assert is_problem(
        api_client.post("/v1/eval/run", content=b"x" * 300_000, headers=headers),
        413,
        "payload-too-large",
    )

    def chunks() -> Iterator[bytes]:
        for _ in range(30):
            yield b"x" * 10_000

    assert is_problem(
        api_client.post("/v1/eval/run", content=chunks(), headers=headers), 413, "payload-too-large"
    )


def test_rag_target_runs_and_fails_closed(
    api_client: TestClient, auth: dict[str, str], mock_rag: RagRecorder
) -> None:
    unreachable = api_client.post("/v1/eval/run", headers=auth)
    assert unreachable.status_code == 201
    assert unreachable.headers["x-eval-gate"] == "failed"
    assert unreachable.headers["x-eval-sample-errors"] == "8"

    def answer(request: httpx.Request) -> httpx.Response:
        query = json.loads(request.content)["query"]
        return httpx.Response(200, json={"answer": f"I cannot answer {query}", "sources": []})

    mock_rag.handler = answer
    allowed = api_client.post(
        "/v1/eval/run", json={"rag_url": "http://rag.internal/answer"}, headers=auth
    )
    assert allowed.status_code == 201
    assert allowed.headers["x-eval-sample-errors"] == "0"
    assert mock_rag.requests[-1].url.host == "rag.internal"


def test_baselines_and_regression(
    api_client: TestClient, auth: dict[str, str], reference_dataset: GoldenDataset
) -> None:
    assert (
        api_client.post(
            "/v1/eval/run", json={"oracle": True, "run_id": "base"}, headers=auth
        ).status_code
        == 201
    )
    stable = api_client.post(
        "/v1/eval/run", json={"oracle": True, "baseline_run_id": "base"}, headers=auth
    )
    assert stable.json()["regression_status"] == "stable"
    other = reference_dataset.model_dump(mode="json") | {"name": "other"}
    mismatch = api_client.post(
        "/v1/eval/run",
        json={"oracle": True, "dataset": other, "baseline_run_id": "base"},
        headers=auth,
    )
    assert is_problem(mismatch, 422, "invalid-input")
    assert "baseline" in mismatch.json()["detail"]


def test_report_lookups(
    api_client: TestClient, auth: dict[str, str], services: EvaluationServices
) -> None:
    assert is_problem(api_client.get("/v1/eval/report/missing-run", headers=auth), 404, "not-found")
    traversal = api_client.get("/v1/eval/report/..%2F..%2Fetc", headers=auth)
    assert traversal.status_code in {404, 422}
    assert "etc" not in traversal.json()["detail"]
    assert is_problem(
        api_client.get("/v1/eval/report/bad%20id", headers=auth), 422, "invalid-input"
    )
    corrupt = services.store.root / "corrupt"
    corrupt.mkdir(parents=True)
    (corrupt / "report.json").write_text("{}")
    assert is_problem(api_client.get("/v1/eval/report/corrupt", headers=auth), 500, "storage-error")


def test_unknown_routes_and_methods(api_client: TestClient) -> None:
    assert is_problem(api_client.get("/nope"), 404, "http-error")
    assert is_problem(api_client.get("/v1/eval/run"), 405, "http-error")
    paths = set(api_client.get("/openapi.json").json()["paths"])
    assert {
        "/v1/eval/run",
        "/v1/eval/report/{run_id}",
        "/v1/eval/report/{run_id}/summary",
        "/v1/eval/health",
    } <= paths


def test_unhandled_errors_do_not_leak(settings: BenchmarkSettings) -> None:
    def explode(*args: object) -> Any:
        raise RuntimeError("SECRET internal detail")

    services = EvaluationServices(settings, engine_factory=explode)
    with TestClient(create_app(services), raise_server_exceptions=False) as client:
        response = client.post("/v1/eval/run", json={"oracle": True})
    assert is_problem(response, 500, "internal-error")
    assert "SECRET" not in response.text


async def test_admission_control(services: EvaluationServices) -> None:
    async with services.run_slot():
        with pytest.raises(BusyError):
            async with services.run_slot():
                pass


def test_busy_service_answers_429(
    api_client: TestClient, auth: dict[str, str], services: EvaluationServices
) -> None:
    services.run_slots = asyncio.Semaphore(0)
    response = api_client.post("/v1/eval/run", json={"oracle": True}, headers=auth)
    assert is_problem(response, 429, "busy")
    assert response.headers["retry-after"] == "30"


def test_sessions_are_isolated_per_app(settings: BenchmarkSettings, tmp_path: Path) -> None:
    other = EvaluationServices(BenchmarkSettings(output_dir=tmp_path / "other-runs"))
    with (
        TestClient(create_app(EvaluationServices(settings))) as first,
        TestClient(create_app(other)) as second,
    ):
        assert (
            first.post("/v1/eval/run", json={"oracle": True, "run_id": "only-here"}).status_code
            == 201
        )
        assert second.get("/v1/eval/report/only-here").status_code == 404
        assert first.get("/v1/eval/report/only-here").status_code == 200


# ---------------------------------------------------------------------------
# Lifespan, middleware, problems
# ---------------------------------------------------------------------------


def test_lifespan_warms_up_and_survives_import_errors(
    settings: BenchmarkSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    for backend in ("ragas", "trulens"):
        with TestClient(
            create_app(EvaluationServices(settings.model_copy(update={"judge_model": backend})))
        ):
            pass

    def broken(name: str) -> object:
        raise ImportError(name)

    monkeypatch.setattr(app_module, "importlib", types.SimpleNamespace(import_module=broken))
    with TestClient(
        create_app(EvaluationServices(settings.model_copy(update={"judge_model": "ragas"})))
    ) as client:
        assert client.get("/v1/eval/health").json()["default_judge"] == "ragas"


def test_create_app_from_settings(settings: BenchmarkSettings) -> None:
    app = create_app(settings=settings)
    assert app.state.services.settings == settings


async def test_middleware_passes_other_scopes_and_reraises_after_start() -> None:
    seen: list[str] = []

    async def lifespan_app(scope: Scope, receive: Receive, send: Send) -> None:
        seen.append(scope["type"])

    await RequestContextMiddleware(lifespan_app, max_body_bytes=10)(
        {"type": "lifespan"}, _receive([]), _send([])
    )
    assert seen == ["lifespan"]

    async def streaming_app(scope: Scope, receive: Receive, send: Send) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await receive()

    middleware = RequestContextMiddleware(streaming_app, max_body_bytes=10)
    scope: Scope = {"type": "http", "headers": [(b"x-request-id", b"bad id!")]}
    sent: list[Message] = []
    with pytest.raises(_BodyTooLargeError):
        await middleware(
            scope, _receive([{"type": "http.request", "body": b"x" * 20}]), _send(sent)
        )
    assert sent[0]["headers"][-1][0] == b"x-request-id"
    assert sent[0]["headers"][-1][1] != b"bad id!"


def _receive(messages: list[Message]) -> Receive:
    async def receive() -> Message:
        return messages.pop(0) if messages else {"type": "http.disconnect"}

    return receive


def _send(sink: list[Message]) -> Send:
    async def send(message: Message) -> None:
        sink.append(message)

    return send


def test_problem_detail_is_truncated() -> None:
    problem = make_problem(400, "x", "Bad Request", "y" * 2000)
    assert len(problem.detail) == 512
    assert problem.detail.endswith("…")
    assert make_problem(400, "x", "Bad Request", "   ").detail == "Bad Request"


# ---------------------------------------------------------------------------
# Service layer: judges, providers, storage
# ---------------------------------------------------------------------------


def provider_services(settings: BenchmarkSettings, **provider: Any) -> EvaluationServices:
    return EvaluationServices(settings, provider=ProviderSettings(**provider))


@pytest.mark.parametrize(
    ("backend", "provider", "message"),
    [
        ("ragas", {}, "EVAL_PROVIDER_LLM_MODEL"),
        ("ragas", {"llm_model": "m"}, "EVAL_PROVIDER_API_KEY"),
        (
            "trulens",
            {"llm_model": "m", "api_key": "k", "provider": "anthropic", "embeddings": "hashing"},
            "OpenAI-compatible",
        ),
        ("ragas", {"llm_model": "m", "api_key": "k", "provider": "anthropic"}, "no embedding API"),
        ("ragas", {"llm_model": "m", "api_key": "k"}, "EVAL_PROVIDER_EMBEDDING_MODEL"),
    ],
)
def test_provider_configuration_errors(
    settings: BenchmarkSettings, backend: str, provider: dict[str, Any], message: str
) -> None:
    services = provider_services(settings, **provider)
    status = services.judge_status(backend)  # type: ignore[arg-type]
    assert not status.configured
    assert message in status.detail
    with pytest.raises(ConfigurationError, match=message):
        services.build_judge(backend, ThresholdPolicy())  # type: ignore[arg-type]


async def test_provider_judges_build_without_network(settings: BenchmarkSettings) -> None:
    from agentic_eval_benchmark.tools import RagasJudge, TruLensJudge  # noqa: PLC0415

    ragas = provider_services(settings, llm_model="m", api_key="sk-test", embedding_model="e")
    assert isinstance(ragas.build_judge("ragas", ThresholdPolicy()), RagasJudge)
    assert ragas.judge_status("ragas").configured
    hashing = provider_services(
        settings,
        llm_model="m",
        base_url="http://127.0.0.1:9/v1",
        embeddings="hashing",
        fallback_to_heuristic=False,
    )
    assert isinstance(hashing.build_judge("ragas", ThresholdPolicy()), RagasJudge)
    assert isinstance(hashing.build_judge("trulens", ThresholdPolicy()), TruLensJudge)
    await ragas.aclose()
    await hashing.aclose()
    await hashing.aclose()  # idempotent


async def test_anthropic_client_path(
    settings: BenchmarkSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    services = provider_services(
        settings, provider="anthropic", llm_model="m", api_key="k", embeddings="hashing"
    )
    monkeypatch.setitem(sys.modules, "anthropic", None)
    with pytest.raises(ConfigurationError, match="anthropic extra"):
        services.build_judge("ragas", ThresholdPolicy())

    closed: list[bool] = []

    class FakeAnthropic:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        async def close(self) -> None:
            closed.append(True)

    fake = types.ModuleType("anthropic")
    fake.AsyncAnthropic = FakeAnthropic  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    captured: dict[str, object] = {}

    def fake_llm(
        client: object, model: str, *, provider: str = "openai", **kwargs: object
    ) -> object:
        captured.update(client=client, provider=provider)
        from agentic_eval_benchmark.tools.offline import ScriptedRagasLLM  # noqa: PLC0415

        return ScriptedRagasLLM(lambda p, m: {})

    from agentic_eval_benchmark.tools import ragas_adapter  # noqa: PLC0415

    monkeypatch.setattr(ragas_adapter, "ragas_llm", fake_llm)
    services.build_judge("ragas", ThresholdPolicy())
    assert isinstance(captured["client"], FakeAnthropic)
    assert captured["provider"] == "anthropic"
    await services.aclose()
    assert closed == [True]


def test_run_store(tmp_path: Path, make_run: MakeRun) -> None:
    store = RunStore(tmp_path / "store")
    assert store.list_runs() == []
    with pytest.raises(InvalidInputError):
        store.run_dir("../escape")
    run = make_run("stored", [0.9, 0.9])
    run_dir = store.save(run)
    assert store.exists("stored")
    assert store.list_runs() == ["stored"]
    assert store.load("stored") == run.report == RunStore.load_path(run_dir / "report.json")
    with pytest.raises(NotFoundError):
        store.load("absent")
    (tmp_path / "unreadable.json").mkdir()
    with pytest.raises((StorageError, NotFoundError)):
        RunStore.load_path(tmp_path / "unreadable.json" / "report.json")


def test_run_store_storage_failures(
    tmp_path: Path, make_run: MakeRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x")
    broken = RunStore(blocker / "runs")
    assert not broken.writable()
    with pytest.raises(StorageError, match="cannot persist"):
        broken.save(make_run("r", [0.9]))
    good = RunStore(tmp_path / "ok")
    good.save(make_run("r", [0.9]))

    def unreadable(self: Path) -> bytes:
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "read_bytes", unreadable)
    with pytest.raises(StorageError, match="unreadable"):
        good.load("r")


def test_rag_client_factory_validation(
    services: EvaluationServices, reference_dataset: GoldenDataset
) -> None:
    with pytest.raises(InvalidInputError):
        services.rag_client(dataset=reference_dataset, url="nope", oracle=False)
