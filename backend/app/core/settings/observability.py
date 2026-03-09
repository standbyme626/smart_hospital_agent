from __future__ import annotations

from typing import Any


OBSERVABILITY_FIELDS: tuple[str, ...] = (
    "LANGCHAIN_TRACING_V2",
    "LANGCHAIN_ENDPOINT",
    "LANGCHAIN_PROJECT",
    "LANGFUSE_ENABLED",
    "LANGFUSE_HOST",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_ENVIRONMENT",
    "SSE_PING_INTERVAL_SECONDS",
    "DEBUG_INCLUDE_NODES",
)


def snapshot(settings_obj: Any) -> dict[str, Any]:
    return {name: getattr(settings_obj, name, None) for name in OBSERVABILITY_FIELDS}
