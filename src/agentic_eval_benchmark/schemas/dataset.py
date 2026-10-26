"""Golden Dataset contracts: the reference questions, contexts and answers under test."""

from typing import Annotated, Final, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from agentic_eval_benchmark.schemas._base import (
    EvalModel,
    Identifier,
    Metadata,
    NonEmptyText,
    SemVer,
)

GOLDEN_DATASET_SCHEMA_VERSION: Final = "1.0.0"

Question = Annotated[str, StringConstraints(min_length=3, max_length=8_192, pattern=r"\S")]


class GoldenSample(EvalModel):
    """One reference case: a question, the context it should be answered from, and the answer."""

    sample_id: Identifier
    question: Question
    contexts: tuple[NonEmptyText, ...] = Field(min_length=1)
    ground_truth: NonEmptyText
    metadata: Metadata = Field(default_factory=dict)


class GoldenDataset(EvalModel):
    """A versioned, non-empty collection of :class:`GoldenSample` with unique IDs.

    ``schema_version`` pins the contract the file was written against; an unknown
    version is rejected instead of being interpreted under different rules.
    """

    name: Identifier
    dataset_version: SemVer
    schema_version: Literal["1.0.0"] = GOLDEN_DATASET_SCHEMA_VERSION
    description: NonEmptyText | None = None
    samples: tuple[GoldenSample, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_unique_ids(self) -> Self:
        seen: set[str] = set()
        duplicates: list[str] = []
        for sample in self.samples:
            if sample.sample_id in seen and sample.sample_id not in duplicates:
                duplicates.append(sample.sample_id)
            seen.add(sample.sample_id)
        if duplicates:
            raise ValueError(f"duplicate sample_id values: {', '.join(duplicates)}")
        return self

    @property
    def sample_ids(self) -> tuple[str, ...]:
        """Sample IDs in dataset order."""
        return tuple(s.sample_id for s in self.samples)

    def get(self, sample_id: str) -> GoldenSample:
        """Return the sample with ``sample_id``; raise :class:`KeyError` if absent."""
        for sample in self.samples:
            if sample.sample_id == sample_id:
                return sample
        raise KeyError(sample_id)
