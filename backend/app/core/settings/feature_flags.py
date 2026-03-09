from __future__ import annotations

from typing import Any


FEATURE_FLAG_FIELDS: tuple[str, ...] = (
    "ENABLE_QUERY_REWRITE",
    "ENABLE_QUERY_EXPANSION",
    "ENABLE_MULTI_QUERY",
    "ENABLE_RETRIEVAL_ROUTER_ADAPTER",
    "ENABLE_DECISION_GOVERNANCE",
    "ENABLE_JSON_SCHEMA_GUARDRAIL",
    "TRIAGE_FAST_PATH_ENABLED",
    "CRISIS_FASTLANE_ENABLED",
    "UPGRADE3_CHAT_SHELL_ENABLED",
    "UPGRADE3_DIAGNOSIS_SHELL_ENABLED",
    "UPGRADE3_RETRIEVER_PIPELINE_ENABLED",
)


def snapshot(settings_obj: Any) -> dict[str, Any]:
    return {name: getattr(settings_obj, name, None) for name in FEATURE_FLAG_FIELDS}
