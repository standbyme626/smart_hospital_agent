from __future__ import annotations

from typing import Any


RAG_FIELDS: tuple[str, ...] = (
    "RAG_RECALL_WINDOW",
    "RAG_RERANK_CANDIDATE_K",
    "RAG_PURE_RETRIEVAL_MODE",
    "RAG_DISABLE_INTENT_ROUTER_WHEN_PURE",
    "RAG_DISABLE_HYDE_WHEN_PURE",
    "RAG_DISABLE_QUERY_REWRITE_WHEN_PURE",
    "RAG_DISABLE_LOW_SCORE_FALLBACK_REWRITE_WHEN_PURE",
    "RAG_DISABLE_SUMMARIZE_WHEN_PURE",
    "RAG_SEMANTIC_CACHE_ENABLED",
    "RAG_CACHE_VERIFY_ON_HIT",
    "RAG_CACHE_VERIFY_MIN_TERM_OVERLAP",
    "RAG_CACHE_VERIFY_MIN_RERANK_SCORE",
    "RAG_CACHE_TTL_SECONDS",
)


def snapshot(settings_obj: Any) -> dict[str, Any]:
    return {name: getattr(settings_obj, name, None) for name in RAG_FIELDS}
