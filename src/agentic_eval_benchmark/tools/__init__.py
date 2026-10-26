"""Integrations: Ragas and TruLens judges, dataset loaders, RAG clients, offline backends.

Attributes load lazily (PEP 562): ``ragas`` and ``trulens`` take seconds to import, so
they are only imported when a judge or offline backend is first used. The modules that
import them pin both libraries offline (telemetry disabled, state under ``./.cache/``);
see :mod:`agentic_eval_benchmark.tools._isolation`.
"""

from importlib import import_module
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from agentic_eval_benchmark.tools.dataset_loader import (
        FIELD_ALIASES,
        MAX_DATASET_BYTES,
        REFERENCE_DATASET_NAME,
        DatasetFormat,
        DatasetLoadError,
        build_dataset,
        detect_format,
        load_golden_dataset,
        load_reference_dataset,
    )
    from agentic_eval_benchmark.tools.offline import (
        HashingEmbeddings,
        ScriptedRagasLLM,
        ScriptedTruLensProvider,
        chain_of_thought_reply,
    )
    from agentic_eval_benchmark.tools.rag_client import (
        HttpRAGClient,
        RAGClientError,
        RAGEndpointConfig,
        StaticRAGClient,
    )
    from agentic_eval_benchmark.tools.ragas_adapter import (
        RagasJudge,
        ragas_embeddings,
        ragas_llm,
    )
    from agentic_eval_benchmark.tools.trulens_adapter import (
        TRULENS_METRICS,
        CustomFeedback,
        TruLensJudge,
        criteria_feedback,
        open_trulens_session,
    )

_EXPORTS: Final[dict[str, str]] = {
    "FIELD_ALIASES": "dataset_loader",
    "MAX_DATASET_BYTES": "dataset_loader",
    "REFERENCE_DATASET_NAME": "dataset_loader",
    "DatasetFormat": "dataset_loader",
    "DatasetLoadError": "dataset_loader",
    "build_dataset": "dataset_loader",
    "detect_format": "dataset_loader",
    "load_golden_dataset": "dataset_loader",
    "load_reference_dataset": "dataset_loader",
    "HashingEmbeddings": "offline",
    "ScriptedRagasLLM": "offline",
    "ScriptedTruLensProvider": "offline",
    "chain_of_thought_reply": "offline",
    "HttpRAGClient": "rag_client",
    "RAGClientError": "rag_client",
    "RAGEndpointConfig": "rag_client",
    "StaticRAGClient": "rag_client",
    "RagasJudge": "ragas_adapter",
    "ragas_embeddings": "ragas_adapter",
    "ragas_llm": "ragas_adapter",
    "TRULENS_METRICS": "trulens_adapter",
    "CustomFeedback": "trulens_adapter",
    "TruLensJudge": "trulens_adapter",
    "criteria_feedback": "trulens_adapter",
    "open_trulens_session": "trulens_adapter",
}


def __getattr__(name: str) -> object:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{module}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))


__all__ = [
    "FIELD_ALIASES",
    "MAX_DATASET_BYTES",
    "REFERENCE_DATASET_NAME",
    "TRULENS_METRICS",
    "CustomFeedback",
    "DatasetFormat",
    "DatasetLoadError",
    "HashingEmbeddings",
    "HttpRAGClient",
    "RAGClientError",
    "RAGEndpointConfig",
    "RagasJudge",
    "ScriptedRagasLLM",
    "ScriptedTruLensProvider",
    "StaticRAGClient",
    "TruLensJudge",
    "build_dataset",
    "chain_of_thought_reply",
    "criteria_feedback",
    "detect_format",
    "load_golden_dataset",
    "load_reference_dataset",
    "open_trulens_session",
    "ragas_embeddings",
    "ragas_llm",
]
