"""Shared base model, constrained primitives and export helpers for every schema."""

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

type Scalar = str | int | float | bool | None
"""A flat value that maps onto a single JSON / Parquet column."""

type Record = dict[str, Scalar]
"""One flat row of a tabular (Parquet / DataFrame) export."""

Identifier = Annotated[
    str,
    StringConstraints(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$"),
]
"""Stable machine identifier (sample, dataset, run, aspect names)."""

NonEmptyText = Annotated[str, StringConstraints(min_length=1, max_length=32_768, pattern=r"\S")]
"""Free text that must contain at least one non-whitespace character."""

SemVer = Annotated[
    str,
    StringConstraints(pattern=r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$"),
]
"""``MAJOR.MINOR.PATCH`` version string without pre-release or build suffixes."""

UnitInterval = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
"""A score bounded to the closed interval ``[0.0, 1.0]``."""

NonNegativeFloat = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]

Metadata = dict[Identifier, Scalar]
"""Flat, JSON-safe annotations; nested structures are rejected on purpose."""


class EvalModel(BaseModel):
    """Immutable, strictly validated base for every schema in the package.

    ``strict=True`` disables type coercion for Python input; JSON input still maps
    strings onto enums, datetimes and tuples, so :meth:`from_json` round-trips
    everything :meth:`to_json` produces.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialize to a JSON document that :meth:`from_json` accepts unchanged."""
        return self.model_dump_json(indent=indent)

    def to_jsonable(self) -> dict[str, object]:
        """Return a JSON-compatible ``dict`` (enums as values, datetimes as ISO 8601)."""
        return self.model_dump(mode="json")

    @classmethod
    def from_json(cls, data: str | bytes) -> Self:
        """Validate a JSON document produced by :meth:`to_json` (or any conforming source)."""
        return cls.model_validate_json(data)


def write_parquet(records: Iterable[Mapping[str, Scalar]], path: Path) -> Path:
    """Write flat records to a Parquet file at ``path`` and return the path.

    The column set is taken from the first record; every record must share it so
    the file has one well-defined schema.
    """
    rows = [dict(r) for r in records]
    if not rows:
        raise ValueError("cannot write an empty Parquet export")
    columns = list(rows[0])
    for index, row in enumerate(rows[1:], start=1):
        if list(row) != columns:
            raise ValueError(f"record {index} has columns {list(row)}, expected {columns}")

    # Deferred: pyarrow is heavy and only needed when a Parquet export is requested.
    import pyarrow as pa  # noqa: PLC0415
    import pyarrow.parquet as pq  # noqa: PLC0415

    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path
