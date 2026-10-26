"""``agentic-eval`` command line: run benchmarks, inspect reports, verify a setup.

Exit codes:

- ``0`` quality gate passed (``run``) / command succeeded
- ``1`` quality gate failed, including a regression against ``--baseline``
- ``2`` invalid input or configuration (bad dataset, options, settings, unknown run)
- ``3`` unexpected internal error

Data goes to stdout (so ``--format json | jq`` works); progress and errors to stderr.
"""

import asyncio
import os
import sys
from collections.abc import Callable
from enum import IntEnum, StrEnum
from pathlib import Path
from typing import Annotated, Final, NoReturn

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from agentic_eval_benchmark import __version__
from agentic_eval_benchmark.api.routes import validation_detail
from agentic_eval_benchmark.api.services import (
    ConfigurationError,
    EvaluationServices,
    RunPlan,
    RunStore,
    ServiceError,
)
from agentic_eval_benchmark.core import BenchmarkSettings, HeuristicJudge, JudgeBackend
from agentic_eval_benchmark.guardrails import (
    BaselineMismatchError,
    DriftTracker,
    QualityGate,
    QualityGateFailureError,
    enforce_gate,
    render_gate_ansi,
    render_gate_markdown,
    strictest_policy,
    write_step_summary,
)
from agentic_eval_benchmark.reporting import render_markdown, render_table
from agentic_eval_benchmark.schemas import BenchmarkReport, GoldenDataset, ThresholdPolicy


class ExitCode(IntEnum):
    OK = 0
    GATE_FAILED = 1
    INVALID = 2
    INTERNAL = 3


class OutputFormat(StrEnum):
    TABLE = "table"
    JSON = "json"
    MARKDOWN = "markdown"


class GateFormat(StrEnum):
    ANSI = "ansi"
    MARKDOWN = "markdown"
    JSON = "json"


class JudgeChoice(StrEnum):
    HEURISTIC = "heuristic"
    MOCK = "mock"
    RAGAS = "ragas"
    TRULENS = "trulens"


def _stdout() -> Console:
    """A console built per call, so color follows the environment at invocation time."""
    return Console(soft_wrap=True)


def _stderr() -> Console:
    return Console(stderr=True, soft_wrap=True)


app = typer.Typer(
    name="agentic-eval",
    help="LLM-as-a-Judge benchmark for RAG endpoints with strict quality gates.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)

_RAG_HELP: Final = "RAG endpoint URL (overrides EVAL_RAG_URL)."


def _fail(code: ExitCode, message: str) -> NoReturn:
    _stderr().print(f"[red]error:[/red] {message}", highlight=False)
    raise typer.Exit(int(code))


def _guarded[T](action: Callable[[], T]) -> T:
    """Map failures onto the documented exit codes."""
    try:
        return action()
    except typer.Exit:
        raise
    except ServiceError as exc:
        _fail(ExitCode(exc.exit_code), exc.message)
    except ValidationError as exc:
        _fail(ExitCode.INVALID, validation_detail(exc))
    except Exception as exc:  # noqa: BLE001 - last resort: never a traceback with exit 1
        _fail(ExitCode.INTERNAL, f"internal error ({type(exc).__name__})")


def _services(
    *,
    output_dir: Path | None = None,
    judge: JudgeChoice | None = None,
    concurrency: int | None = None,
) -> EvaluationServices:
    overrides: dict[str, object] = {}
    if output_dir is not None:
        overrides["output_dir"] = output_dir
    if judge is not None:
        overrides["judge_model"] = judge.value
    if concurrency is not None:
        overrides["concurrency_limit"] = concurrency
    return EvaluationServices(BenchmarkSettings(**overrides))  # type: ignore[arg-type]


def _color(console: Console) -> bool:
    return not os.environ.get("NO_COLOR") and console.is_terminal


def _dataset(services: EvaluationServices, path: Path | None) -> GoldenDataset:
    return services.reference_dataset() if path is None else services.load_dataset_file(path)


def _load_report(store: RunStore, target: str) -> BenchmarkReport:
    path = Path(target).expanduser()
    if path.exists():
        return store.load_path(path.resolve())
    return store.load(target)


def _write_plain(text: str) -> None:
    """Machine-readable output (JSON, Markdown) bypasses rich: FORCE_COLOR or a TTY must
    never inject terminal escape codes into what `jq` or a step summary will parse."""
    sys.stdout.write(text if text.endswith("\n") else text + "\n")
    sys.stdout.flush()


def _emit(report: BenchmarkReport, fmt: OutputFormat, *, full_json: bool) -> None:
    summary = report.summary
    if fmt is OutputFormat.JSON:
        _write_plain((report if full_json else summary).to_json(indent=2))
    elif fmt is OutputFormat.MARKDOWN:
        _write_plain(render_markdown(summary, evaluations=report.evaluations))
    else:
        _stdout().print(render_table(summary))


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


@app.command()
def run(
    dataset_path: Annotated[
        Path | None,
        typer.Option(help="Golden Dataset file (.json/.jsonl/.parquet/.csv). Default: built-in."),
    ] = None,
    rag_url: Annotated[str | None, typer.Option(help=_RAG_HELP)] = None,
    oracle: Annotated[
        bool, typer.Option(help="Answer from the dataset itself (offline pipeline smoke test).")
    ] = False,
    judge_model: Annotated[
        JudgeChoice | None, typer.Option(help="Judge backend (overrides EVAL_JUDGE_MODEL).")
    ] = None,
    threshold_faithfulness: Annotated[
        float | None, typer.Option(min=0.0, max=1.0, help="Faithfulness gate (default 0.85).")
    ] = None,
    threshold_relevancy: Annotated[
        float | None, typer.Option(min=0.0, max=1.0, help="Answer relevancy gate (default 0.85).")
    ] = None,
    non_strict: Annotated[
        bool, typer.Option(help="Allow thresholds below 0.85 and gate on means only.")
    ] = False,
    regression_tolerance: Annotated[
        float | None, typer.Option(min=0.0, max=1.0, help="Allowed drop of a metric mean.")
    ] = None,
    baseline: Annotated[
        str | None, typer.Option(help="Baseline run ID or report path for regression checks.")
    ] = None,
    run_id: Annotated[
        str | None, typer.Option(help="Explicit run ID (default: generated).")
    ] = None,
    output_dir: Annotated[
        Path | None, typer.Option(help="Run storage (overrides EVAL_OUTPUT_DIR).")
    ] = None,
    concurrency: Annotated[int | None, typer.Option(min=1, max=256)] = None,
    output_format: Annotated[
        OutputFormat, typer.Option("--format", help="Output format.")
    ] = OutputFormat.TABLE,
) -> None:
    """Evaluate a dataset against a RAG target; exit 0 on pass, 1 on gate failure."""
    if oracle and rag_url is not None:
        _fail(ExitCode.INVALID, "--oracle does not call a RAG endpoint; drop --rag-url")

    def action() -> ExitCode:
        services = _services(output_dir=output_dir, judge=judge_model, concurrency=concurrency)
        policy_overrides: dict[str, object] = {"strict_mode": not non_strict}
        if threshold_faithfulness is not None:
            policy_overrides["faithfulness_min"] = threshold_faithfulness
        if threshold_relevancy is not None:
            policy_overrides["relevancy_min"] = threshold_relevancy
        if regression_tolerance is not None:
            policy_overrides["regression_tolerance"] = regression_tolerance
        policy = ThresholdPolicy(**policy_overrides)  # type: ignore[arg-type]
        baseline_summary = (
            None if baseline is None else _load_report(services.store, baseline).summary
        )
        backend: JudgeBackend = services.settings.judge_model
        plan = RunPlan(
            dataset=_dataset(services, dataset_path),
            policy=policy,
            judge=backend,
            rag_url=rag_url,
            oracle=oracle,
            baseline=baseline_summary,
            run_id=run_id,
        )
        _stderr().print(
            f"Evaluating [bold]{plan.dataset.name}[/bold] v{plan.dataset.dataset_version} "
            f"({len(plan.dataset.samples)} samples) with the [bold]{backend}[/bold] judge…"
        )

        async def execute() -> tuple[BenchmarkReport, int, Path]:
            try:
                result, run_dir = await services.execute(plan)
            finally:
                await services.aclose()
            return result.report, len(result.errors), run_dir

        report, errors, run_dir = asyncio.run(execute())
        if errors:
            _stderr().print(f"[yellow]{errors} sample(s) could not be evaluated (see errors.json)")
        _stderr().print(f"Saved to {run_dir}")
        _emit(report, output_format, full_json=False)
        result = QualityGate(policy).evaluate(report.summary, evaluations=report.evaluations)
        _stderr().file.write(render_gate_ansi(result, color=_color(_stderr())))
        write_step_summary(
            render_gate_markdown(result, report.summary, evaluations=report.evaluations)
        )
        passed = report.summary.gate_passed and result.passed
        return ExitCode.OK if passed else ExitCode.GATE_FAILED

    raise typer.Exit(int(_guarded(action)))


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


@app.command()
def report(
    target: Annotated[str, typer.Argument(help="Run ID, run directory or report.json path.")],
    output_dir: Annotated[
        Path | None, typer.Option(help="Run storage (overrides EVAL_OUTPUT_DIR).")
    ] = None,
    output_format: Annotated[
        OutputFormat, typer.Option("--format", help="Output format (json = full report).")
    ] = OutputFormat.MARKDOWN,
    export: Annotated[
        Path | None,
        typer.Option(help="Also write the report: .json (full report) or .parquet (per sample)."),
    ] = None,
) -> None:
    """Print a stored run; optionally export it to JSON or Parquet."""

    def action() -> None:
        services = _services(output_dir=output_dir)
        loaded = _load_report(services.store, target)
        if export is not None:
            suffix = export.suffix.lower()
            if suffix == ".json":
                export.parent.mkdir(parents=True, exist_ok=True)
                export.write_text(loaded.to_json(indent=2) + "\n", encoding="utf-8")
            elif suffix in {".parquet", ".pq"}:
                loaded.to_parquet(export)
            else:
                _fail(ExitCode.INVALID, "--export must end in .json or .parquet")
            _stderr().print(f"Exported to {export}")
        _emit(loaded, output_format, full_json=True)

    _guarded(action)


# ---------------------------------------------------------------------------
# gate (the fail-CI hook)
# ---------------------------------------------------------------------------


@app.command()
def gate(
    target: Annotated[str, typer.Argument(help="Run ID, run directory or report.json path.")],
    baseline: Annotated[
        str | None, typer.Option(help="Baseline run ID or report path to compare against.")
    ] = None,
    tolerance_delta: Annotated[
        float | None,
        typer.Option(min=0.0, max=1.0, help="Allowed drop vs the baseline (default 0.02)."),
    ] = None,
    drift: Annotated[
        bool, typer.Option(help="Fail on gradual decay across stored runs of the dataset.")
    ] = True,
    drift_window: Annotated[int, typer.Option(min=1, max=100, help="Runs in the window.")] = 5,
    drift_min_history: Annotated[
        int, typer.Option(min=1, max=100, help="Earlier runs needed for a drift verdict.")
    ] = 3,
    drift_max_decline: Annotated[
        float, typer.Option(min=0.0, max=1.0, help="Allowed drop below the moving average.")
    ] = 0.03,
    output_dir: Annotated[
        Path | None, typer.Option(help="Run storage and drift history (overrides EVAL_OUTPUT_DIR).")
    ] = None,
    step_summary: Annotated[
        Path | None,
        typer.Option(help="Append the Markdown report here (default: $GITHUB_STEP_SUMMARY)."),
    ] = None,
    output_format: Annotated[GateFormat, typer.Option("--format")] = GateFormat.ANSI,
) -> None:
    """Fail CI (exit 1) unless a stored run clears the blueprint gate, the baseline and drift.

    The gate uses the stricter of the blueprint policy and the run's own policy. Under
    GitHub Actions each violation is also emitted as an ``::error`` annotation.
    """

    def action() -> ExitCode:
        services = _services(output_dir=output_dir)
        report = _load_report(services.store, target)
        summary = report.summary
        base = None if baseline is None else _load_report(services.store, baseline).summary
        policy = strictest_policy(summary.policy)
        quality_gate = QualityGate(policy, tolerance_delta=tolerance_delta)
        drift_report = (
            DriftTracker(
                services.store.root,
                window=drift_window,
                min_history=min(drift_min_history, drift_window),
                max_decline=drift_max_decline,
            ).track(summary, policy=policy)
            if drift
            else None
        )
        try:
            result = quality_gate.evaluate(
                summary, base, drift=drift_report, evaluations=report.evaluations
            )
        except BaselineMismatchError as exc:
            _fail(ExitCode.INVALID, str(exc))
        if output_format is GateFormat.JSON:
            _write_plain(result.to_json(indent=2))
        elif output_format is GateFormat.MARKDOWN:
            _write_plain(
                render_gate_markdown(
                    result, summary, evaluations=report.evaluations, drift=drift_report
                )
            )
        else:
            sys.stdout.write(render_gate_ansi(result, color=_color(_stdout())))
        try:
            enforce_gate(
                summary,
                base,
                gate=quality_gate,
                evaluations=report.evaluations,
                drift=drift_report,
                step_summary=step_summary,
            )
        except QualityGateFailureError:
            return ExitCode.GATE_FAILED
        return ExitCode.OK

    raise typer.Exit(int(_guarded(action)))


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------


class _Verification:
    """Independent setup probes; each records ``(name, ok, detail)`` and never raises."""

    def __init__(
        self, services: EvaluationServices, dataset_path: Path | None, rag_url: str | None
    ) -> None:
        self.services = services
        self.dataset_path = dataset_path
        self.rag_url = rag_url
        self.dataset: GoldenDataset | None = None
        self.results: list[tuple[str, bool, str]] = []

    def check(self, name: str, probe: Callable[[], str]) -> None:
        try:
            self.results.append((name, True, probe()))
        except ServiceError as exc:
            self.results.append((name, False, exc.message))
        except ValidationError as exc:
            self.results.append((name, False, validation_detail(exc)))
        except Exception as exc:  # noqa: BLE001 - every probe failure is reported, not raised
            self.results.append((name, False, type(exc).__name__))

    def storage(self) -> str:
        if not self.services.store.writable():
            raise PermissionError
        return str(self.services.store.root)

    def load_dataset(self) -> str:
        dataset = _dataset(self.services, self.dataset_path)
        self.dataset = dataset
        return f"{dataset.name} v{dataset.dataset_version}, {len(dataset.samples)} samples"

    def offline_judge(self) -> str:
        sample = (self.dataset or self.services.reference_dataset()).samples[0]
        scores = HeuristicJudge().score(sample, sample.ground_truth, sample.contexts)
        return f"heuristic judge scored {len(scores)} metrics offline"

    def provider_dependencies(self) -> str:
        missing = [b for b in ("ragas", "trulens") if not self.services.judge_status(b).available]
        if missing:
            raise ConfigurationError(f"not importable: {', '.join(missing)}")
        return "ragas and trulens importable"

    def selected_judge(self) -> str:
        backend = self.services.settings.judge_model
        status = self.services.judge_status(backend)
        if not status.configured:
            raise ConfigurationError(status.detail)
        self.services.build_judge(backend, ThresholdPolicy())
        return f"{backend}: {status.detail}"

    def rag_target(self) -> str:
        if self.dataset is None:
            raise ConfigurationError("no dataset to take a probe question from")
        dataset = self.dataset

        async def probe() -> str:
            client = self.services.rag_client(dataset=dataset, url=self.rag_url, oracle=False)
            async with client as rag, asyncio.timeout(self.services.settings.timeout_seconds):
                response = await rag.query(dataset.samples[0].question)
            return f"answered with {len(response.contexts)} context(s)"

        return asyncio.run(probe())


@app.command()
def verify(
    dataset_path: Annotated[
        Path | None, typer.Option(help="Dataset to validate. Default: built-in reference.")
    ] = None,
    rag_url: Annotated[str | None, typer.Option(help=_RAG_HELP)] = None,
    judge_model: Annotated[JudgeChoice | None, typer.Option(help="Judge backend to check.")] = None,
    skip_rag: Annotated[bool, typer.Option(help="Do not contact the RAG endpoint.")] = False,
) -> None:
    """Check dataset format, storage, judge dependencies and RAG reachability (exit 0/2)."""

    def action() -> None:
        probes = _Verification(_services(judge=judge_model), dataset_path, rag_url)
        probes.check("storage", probes.storage)
        probes.check("dataset", probes.load_dataset)
        probes.check("offline judge", probes.offline_judge)
        probes.check("provider judge deps", probes.provider_dependencies)
        probes.check("selected judge", probes.selected_judge)
        if not skip_rag:
            probes.check("rag target", probes.rag_target)
        table = Table(title=f"agentic-eval {__version__} verify")
        for column in ("check", "status", "detail"):
            table.add_column(column)
        for name, ok, detail in probes.results:
            table.add_row(name, "[green]ok[/green]" if ok else "[red]FAIL[/red]", detail)
        _stdout().print(table)
        if not all(ok for _, ok, _ in probes.results):
            raise typer.Exit(int(ExitCode.INVALID))

    _guarded(action)


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Bind address.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(min=1, max=65535)] = 8080,
) -> None:
    """Serve the HTTP API (``/v1/eval``) with uvicorn."""
    import uvicorn  # noqa: PLC0415

    from agentic_eval_benchmark.api import create_app  # noqa: PLC0415

    # Access logs are off: request lines can carry run IDs and query strings into shared
    # logs; problems carry the request ID for correlation instead.
    uvicorn.run(
        create_app(),
        host=host,
        port=port,
        access_log=False,
        server_header=False,
        timeout_graceful_shutdown=10,
    )


def main() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":
    main()
