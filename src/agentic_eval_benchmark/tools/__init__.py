"""Integrations: Ragas and TruLens judges, dataset loaders, RAG clients, offline backends.

Importing this package pins Ragas and TruLens offline (telemetry disabled, state under
``./.cache/``); see :mod:`agentic_eval_benchmark.tools._isolation`.
"""

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
from agentic_eval_benchmark.tools.ragas_adapter import RagasJudge, ragas_embeddings, ragas_llm
from agentic_eval_benchmark.tools.trulens_adapter import (
    TRULENS_METRICS,
    CustomFeedback,
    TruLensJudge,
    criteria_feedback,
    open_trulens_session,
)

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
