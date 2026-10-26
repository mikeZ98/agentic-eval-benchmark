"""Keep third-party evaluation libraries offline and their state on the project SSD.

- Ragas builds a telemetry event, and with it a user-ID file under the platform data
  dir (``~/Library/Application Support/ragas`` on macOS), *before* it checks
  ``RAGAS_DO_NOT_TRACK``. :func:`pin_ragas_offline` forces the opt-out and redirects
  that directory under ``$XDG_DATA_HOME`` (``./.cache/xdg-data`` via ``.envrc``).
- TruLens defaults to ``sqlite:///default.sqlite`` in the working directory and may
  export OpenTelemetry spans. :func:`trulens_home` resolves the pinned state dir.
"""

import os
from pathlib import Path

_CACHE_FALLBACK = Path(".cache")


def cache_root() -> Path:
    """``$CACHE_ROOT`` from ``.envrc``, else ``./.cache`` (absolute)."""
    return Path(os.environ.get("CACHE_ROOT") or _CACHE_FALLBACK).expanduser().resolve()


def trulens_home() -> Path:
    """``$TRULENS_HOME`` from ``.envrc``, else ``<cache root>/trulens`` (absolute)."""
    raw = os.environ.get("TRULENS_HOME")
    return Path(raw).expanduser().resolve() if raw else cache_root() / "trulens"


def pin_ragas_offline() -> Path:
    """Disable Ragas telemetry and pin its state dir; returns that dir. Idempotent."""
    os.environ["RAGAS_DO_NOT_TRACK"] = "true"
    from ragas import _analytics  # noqa: PLC0415 - must run before any ragas factory

    _analytics.do_not_track.cache_clear()
    data_home = os.environ.get("XDG_DATA_HOME")
    state = (Path(data_home) if data_home else cache_root() / "xdg-data").resolve() / "ragas"

    def user_data_dir(*_args: object, **_kwargs: object) -> str:
        return str(state)

    setattr(_analytics, "user_data_dir", user_data_dir)  # noqa: B010 - patch a module global
    return state


def pin_trulens_offline() -> Path:
    """Disable TruLens OpenTelemetry export unless explicitly enabled; returns the state dir."""
    os.environ.setdefault("TRULENS_OTEL_TRACING", "0")
    home = trulens_home()
    home.mkdir(parents=True, exist_ok=True)
    return home
