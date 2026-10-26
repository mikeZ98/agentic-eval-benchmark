"""Hermetic Golden Dataset loaders: local JSON, JSONL, Parquet and CSV files only.

Raw rows are mapped onto :class:`~agentic_eval_benchmark.schemas.GoldenSample` by
canonical field name or a known alias (Ragas-style ``user_input`` / ``reference`` /
``reference_contexts`` included) and then validated strictly; any problem is reported
as a :class:`DatasetLoadError` naming the row and field, never the row's content.
"""

import csv
import io
import json
from collections.abc import Iterable, Mapping
from importlib import resources
from pathlib import Path
from typing import Final, Literal

from pydantic import ValidationError

from agentic_eval_benchmark.schemas import GoldenDataset, GoldenSample

type DatasetFormat = Literal["json", "jsonl", "parquet", "csv"]
type ExtraFields = Literal["forbid", "metadata", "ignore"]

MAX_DATASET_BYTES: Final = 64 * 1024 * 1024
MAX_SAMPLES: Final = 100_000
UNVERSIONED: Final = "0.0.0"
REFERENCE_DATASET_NAME: Final = "synthetic-rag-reference"
_REFERENCE_RESOURCE: Final = "data/synthetic_golden_v1.json"

FIELD_ALIASES: Final[Mapping[str, tuple[str, ...]]] = {
    "sample_id": ("sample_id", "id"),
    "question": ("question", "user_input", "query"),
    "contexts": ("contexts", "reference_contexts", "context"),
    "ground_truth": ("ground_truth", "reference", "expected_answer"),
    "metadata": ("metadata",),
}
"""Source column names accepted per canonical field, in priority order. ``answer`` and
``response`` are deliberately absent: in most RAG datasets they hold the system's
output, not the reference answer."""

_SUFFIXES: Final[Mapping[str, DatasetFormat]] = {
    ".json": "json",
    ".jsonl": "jsonl",
    ".ndjson": "jsonl",
    ".parquet": "parquet",
    ".pq": "parquet",
    ".csv": "csv",
}
_HEADER_FIELDS: Final = frozenset({"name", "dataset_version", "schema_version", "description"})
_MAX_REPORTED_ERRORS: Final = 5


class DatasetLoadError(ValueError):
    """A dataset file that cannot be read or does not map onto a valid GoldenDataset."""


def _reject_constant(token: str) -> float:
    raise DatasetLoadError(f"non-finite number {token} is not allowed")


def _loads(text: str) -> object:
    return json.loads(text, parse_constant=_reject_constant)


def _describe(exc: ValidationError) -> str:
    errors = exc.errors(include_url=False, include_input=False)
    parts = [f"{'.'.join(map(str, e['loc'])) or '<root>'}: {e['msg']}" for e in errors]
    more = len(parts) - _MAX_REPORTED_ERRORS
    return "; ".join(parts[:_MAX_REPORTED_ERRORS]) + (f" (+{more} more)" if more > 0 else "")


def detect_format(path: Path) -> DatasetFormat:
    """Format from the file suffix: ``.json``, ``.jsonl``/``.ndjson``, ``.parquet``/``.pq``
    or ``.csv``."""
    try:
        return _SUFFIXES[path.suffix.lower()]
    except KeyError:
        formats = sorted(set(_SUFFIXES.values()))
        raise DatasetLoadError(
            f"cannot infer the format of {path.name}; pass format= one of {formats}"
        ) from None


type Rows = list[Mapping[str, object]]


def _objects(items: list[object], where: str) -> Rows:
    rows: Rows = []
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise DatasetLoadError(f"{where} sample {index}: expected a JSON object")
        rows.append(item)
    return rows


def _read_parquet(path: Path) -> Rows:
    import pyarrow.parquet as pq  # noqa: PLC0415 - heavy, only for Parquet input

    try:
        table = pq.read_table(path)
    except Exception as exc:
        raise DatasetLoadError(f"{path.name}: unreadable Parquet file ({exc})") from exc
    return _objects(list(table.to_pylist()), path.name)


def _read_csv(path: Path, text: str) -> Rows:
    rows: Rows = []
    for number, record in enumerate(csv.DictReader(io.StringIO(text, newline="")), start=2):
        if None in record:  # line 1 is the header
            raise DatasetLoadError(f"{path.name} row {number}: more cells than header columns")
        rows.append(record)
    return rows


def _read_jsonl(path: Path, text: str) -> Rows:
    rows: Rows = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = _loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetLoadError(f"{path.name} line {number}: invalid JSON ({exc.msg})") from exc
        if not isinstance(row, dict):
            raise DatasetLoadError(f"{path.name} line {number}: expected a JSON object")
        rows.append(row)
    return rows


def _read_json(path: Path, text: str) -> tuple[dict[str, object], Rows]:
    try:
        document = _loads(text)
    except json.JSONDecodeError as exc:
        raise DatasetLoadError(f"{path.name}: invalid JSON ({exc.msg})") from exc
    if isinstance(document, list):
        return {}, _objects(document, path.name)
    if isinstance(document, dict) and isinstance(document.get("samples"), list):
        unknown = set(document) - _HEADER_FIELDS - {"samples"}
        if unknown:
            raise DatasetLoadError(f"{path.name}: unknown dataset fields {sorted(unknown)}")
        header = {k: v for k, v in document.items() if k in _HEADER_FIELDS}
        return header, _objects(document["samples"], path.name)
    raise DatasetLoadError(
        f"{path.name}: expected a list of samples or an object with a 'samples' list"
    )


def _read_rows(path: Path, fmt: DatasetFormat) -> tuple[dict[str, object], Rows]:
    """``(header, rows)``; the header is only non-empty for a JSON dataset document."""
    if fmt == "parquet":
        return {}, _read_parquet(path)
    text = path.read_text(encoding="utf-8-sig")
    if fmt == "csv":
        return {}, _read_csv(path, text)
    if fmt == "jsonl":
        return {}, _read_jsonl(path, text)
    return _read_json(path, text)


def _parse_cell(value: object, *, field: str, as_list: bool) -> object:
    """Decode JSON-encoded list/object cells (CSV) and normalize sequence types."""
    if isinstance(value, str) and value.strip().startswith("[" if as_list else "{"):
        try:
            value = _loads(value)
        except json.JSONDecodeError as exc:
            raise DatasetLoadError(f"field {field}: invalid JSON ({exc.msg})") from exc
    if as_list:
        if isinstance(value, str):
            return (value,)
        if isinstance(value, list | tuple):
            return tuple(value)
    return value


def _map_row(
    row: Mapping[str, object],
    position: int,
    *,
    field_map: Mapping[str, str],
    extra: ExtraFields,
    id_prefix: str,
) -> dict[str, object]:
    mapped: dict[str, object] = {}
    used: set[str] = set()
    for field, aliases in FIELD_ALIASES.items():
        sources = (field_map[field],) if field in field_map else aliases
        source = next((s for s in sources if s in row), None)
        if source is None:
            continue
        used.add(source)
        value = row[source]
        if value is None or (isinstance(value, str) and not value.strip() and field != "question"):
            continue
        mapped[field] = value
    leftover = {k: v for k, v in row.items() if k not in used}
    if leftover and extra == "forbid":
        raise DatasetLoadError(
            f"unmapped fields {sorted(leftover)}; map them with field_map= or pass extra='metadata'"
        )

    sample_id = mapped.get("sample_id")
    mapped["sample_id"] = f"{id_prefix}-{position:04d}" if sample_id is None else str(sample_id)
    if "contexts" in mapped:
        mapped["contexts"] = _parse_cell(mapped["contexts"], field="contexts", as_list=True)
    metadata = _parse_cell(mapped.get("metadata", {}), field="metadata", as_list=False)
    if not isinstance(metadata, Mapping):
        raise DatasetLoadError("field metadata: expected an object")
    metadata = dict(metadata)
    if extra == "metadata":
        clash = set(leftover) & set(metadata)
        if clash:
            raise DatasetLoadError(f"extra fields {sorted(clash)} clash with metadata keys")
        metadata.update(leftover)
    mapped["metadata"] = metadata
    return mapped


def build_dataset(
    rows: Iterable[Mapping[str, object]],
    *,
    name: str,
    dataset_version: str = UNVERSIONED,
    description: str | None = None,
    field_map: Mapping[str, str] | None = None,
    extra: ExtraFields = "forbid",
    id_prefix: str = "sample",
    source: str = "<rows>",
) -> GoldenDataset:
    """Map raw rows onto a validated :class:`GoldenDataset`.

    ``field_map`` pins a canonical field to a source column
    (``{"question": "prompt"}``); otherwise :data:`FIELD_ALIASES` apply. Rows without
    an ID get a positional one (``<id_prefix>-0001`` …), stable only while the row
    order is. ``extra`` decides what happens to unmapped columns.
    """
    unknown = set(field_map or {}) - set(FIELD_ALIASES)
    if unknown:
        raise DatasetLoadError(f"field_map names unknown fields {sorted(unknown)}")
    samples: list[GoldenSample] = []
    for position, row in enumerate(rows, start=1):
        if position > MAX_SAMPLES:
            raise DatasetLoadError(f"{source}: more than {MAX_SAMPLES} samples")
        try:
            mapped = _map_row(
                row, position, field_map=field_map or {}, extra=extra, id_prefix=id_prefix
            )
            samples.append(GoldenSample.model_validate(mapped))
        except DatasetLoadError as exc:
            raise DatasetLoadError(f"{source} sample {position}: {exc}") from None
        except ValidationError as exc:
            raise DatasetLoadError(f"{source} sample {position}: {_describe(exc)}") from None
    try:
        return GoldenDataset(
            name=name,
            dataset_version=dataset_version,
            description=description,
            samples=tuple(samples),
        )
    except ValidationError as exc:
        raise DatasetLoadError(f"{source}: {_describe(exc)}") from None


def load_golden_dataset(
    path: Path | str,
    *,
    format: DatasetFormat | None = None,  # noqa: A002 - mirrors the user-facing option
    name: str | None = None,
    dataset_version: str | None = None,
    description: str | None = None,
    field_map: Mapping[str, str] | None = None,
    extra: ExtraFields = "forbid",
    id_prefix: str = "sample",
) -> GoldenDataset:
    """Load a local dataset file into a validated :class:`GoldenDataset`.

    A JSON file may be a full dataset document (``name``, ``dataset_version``,
    ``schema_version``, ``samples``) or a bare list of samples. For bare rows,
    ``name`` defaults to the file stem and ``dataset_version`` to ``0.0.0``
    (unversioned); explicit arguments must not contradict a document's header.
    """
    file = Path(path).expanduser().resolve()
    if not file.is_file():
        raise DatasetLoadError(f"{file} is not a file")
    size = file.stat().st_size
    if size > MAX_DATASET_BYTES:
        raise DatasetLoadError(f"{file.name} is {size} bytes; the limit is {MAX_DATASET_BYTES}")
    fmt = format or detect_format(file)
    header, rows = _read_rows(file, fmt)
    for key, given in (
        ("name", name),
        ("dataset_version", dataset_version),
        ("description", description),
    ):
        if given is not None and key in header and header[key] != given:
            raise DatasetLoadError(f"{file.name}: {key}={given!r} contradicts the file header")
    if header.get("schema_version", "1.0.0") != "1.0.0":
        raise DatasetLoadError(
            f"{file.name}: unsupported schema_version {header['schema_version']!r}"
        )
    resolved_name = name or header.get("name") or _slug(file.stem)
    resolved_version = dataset_version or header.get("dataset_version") or UNVERSIONED
    resolved_description = description or header.get("description")
    for key, value in (("name", resolved_name), ("dataset_version", resolved_version)):
        if not isinstance(value, str):
            raise DatasetLoadError(f"{file.name}: {key} must be a string")
    if resolved_description is not None and not isinstance(resolved_description, str):
        raise DatasetLoadError(f"{file.name}: description must be a string")
    return build_dataset(
        rows,
        name=str(resolved_name),
        dataset_version=str(resolved_version),
        description=resolved_description,
        field_map=field_map,
        extra=extra,
        id_prefix=id_prefix,
        source=file.name,
    )


def _slug(stem: str) -> str:
    slug = "".join(c if c.isalnum() or c in "_.-" else "-" for c in stem).strip("-._")
    return slug or "dataset"


def load_reference_dataset() -> GoldenDataset:
    """The bundled synthetic Golden Dataset (8 RAG scenarios, fictional facts)."""
    data = resources.files("agentic_eval_benchmark.tools").joinpath(_REFERENCE_RESOURCE)
    return GoldenDataset.from_json(data.read_bytes())
