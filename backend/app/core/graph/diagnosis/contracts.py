from __future__ import annotations

from typing import Any, TypedDict


class RouteDecision(TypedDict, total=False):
    department_top1: str
    department_top3: list[str]
    confidence: float
    source: str


class DiagnosisOutput(TypedDict, total=False):
    diagnosis_schema_version: str
    department_top1: str
    department_top3: list[str]
    confidence: float
    reasoning: str
    citations: list[dict[str, Any]]


class DiagnosisStateContract(TypedDict, total=False):
    messages: list[Any]
    patient_id: str
    session_id: str
    request_id: str
    event: dict[str, Any]
    user_input: str
    current_turn_input: str
    retrieval_query: str
    retrieval_top_k_override: int
    retrieval_use_rerank: bool
    retrieval_rerank_threshold: float
    query_rewrite_timeout_override_s: float
    crisis_fastlane_override: bool
    department_top1: str
    department_top3: list[str]
    confidence: float
    diagnosis_output: DiagnosisOutput
    citations: list[dict[str, Any]]
    decision_action: str
    decision_reason: str
    confidence_score: float
    grounded_flag: bool
    debug_snapshots: list[dict[str, Any]]
    runtime_config_effective: dict[str, Any]


__all__ = [
    "DiagnosisOutput",
    "DiagnosisStateContract",
    "RouteDecision",
]
