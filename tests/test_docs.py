"""The README stays truthful: every setting documented, every command listed, examples run."""

import re
from pathlib import Path

import pytest
from pydantic_settings import BaseSettings

from agentic_eval_benchmark.api import ApiSettings, ProviderSettings
from agentic_eval_benchmark.cli import app
from agentic_eval_benchmark.core import BenchmarkSettings
from agentic_eval_benchmark.tools import RAGEndpointConfig

README = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "settings", [BenchmarkSettings, ApiSettings, ProviderSettings, RAGEndpointConfig]
)
def test_every_setting_is_documented(settings: type[BaseSettings]) -> None:
    prefix = settings.model_config.get("env_prefix", "")
    missing = [
        f"{prefix}{name.upper()}"
        for name in settings.model_fields
        if f"`{prefix}{name.upper()}`" not in README
    ]
    assert missing == []


def test_every_command_and_exit_code_is_documented() -> None:
    # Typer leaves `name` unset unless given explicitly; the callback name is the command.
    commands = {
        command.name or getattr(command.callback, "__name__", "")
        for command in app.registered_commands
    }
    assert commands == {"run", "report", "verify", "gate", "serve"}
    assert all(f"### `{command}`" in README for command in commands)
    for code in ("`0`", "`1`", "`2`", "`3`"):
        assert f"| {code} |" in README


def test_python_example_runs(capsys: pytest.CaptureFixture[str]) -> None:
    (example,) = re.findall(r"```python\n(.*?)```", README, re.S)
    exec(compile(example, "README.md", "exec"), {"__name__": "readme_example"})  # noqa: S102
    assert capsys.readouterr().out.strip() == "True 1.0"
    assert Path(".cache/eval_runs/library-demo/report.json").is_file()
