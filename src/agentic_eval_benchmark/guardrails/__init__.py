"""Quality gates, CI blocking and metric drift tracking over benchmark runs."""

from agentic_eval_benchmark.guardrails.ci_blocker import (
    QualityGateFailureError,
    enforce_gate,
    github_annotations,
    in_github_actions,
    render_gate_ansi,
    render_gate_markdown,
    write_step_summary,
)
from agentic_eval_benchmark.guardrails.drift import DriftReport, DriftTracker, MetricDrift
from agentic_eval_benchmark.guardrails.gates import (
    BaselineMismatchError,
    GateResult,
    GateViolation,
    MetricDelta,
    QualityGate,
    ViolationCode,
    strictest_policy,
)

__all__ = [
    "BaselineMismatchError",
    "DriftReport",
    "DriftTracker",
    "GateResult",
    "GateViolation",
    "MetricDelta",
    "MetricDrift",
    "QualityGate",
    "QualityGateFailureError",
    "ViolationCode",
    "enforce_gate",
    "github_annotations",
    "in_github_actions",
    "render_gate_ansi",
    "render_gate_markdown",
    "strictest_policy",
    "write_step_summary",
]
