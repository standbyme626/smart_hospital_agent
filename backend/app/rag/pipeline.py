from __future__ import annotations

import asyncio
from typing import Any, Dict, List

import structlog

from app.core.config import settings

logger = structlog.get_logger(__name__)


async def search_rag30(retriever, query: str, **kwargs: Any) -> Any:
    return await retriever.search_rag30(query=query, **kwargs)


def _legacy_search_sync(retriever, query: str, top_k: int = 3, intent: str | None = None) -> List[Dict[str, Any]]:
    if hasattr(retriever, "_legacy_search_sync_wrapper"):
        return retriever._legacy_search_sync_wrapper(query=query, top_k=top_k, intent=intent)
    try:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return loop.run_until_complete(retriever.search_rag30(query, top_k, intent))
            return loop.run_until_complete(retriever.search_rag30(query, top_k, intent))
        except RuntimeError:
            return asyncio.run(retriever.search_rag30(query, top_k, intent))
    except Exception as exc:
        logger.error("legacy_search_sync_failed", error=str(exc))
        return []


def search_sync(
    retriever,
    query: str,
    top_k: int = 3,
    intent: str | None = None,
) -> List[Dict[str, Any]]:
    pipeline_enabled = bool(getattr(settings, "UPGRADE3_RETRIEVER_PIPELINE_ENABLED", True))
    logger.info(
        "retriever_pipeline_search_sync",
        upgrade3_retriever_pipeline_enabled=pipeline_enabled,
        pipeline_path="pipeline" if pipeline_enabled else "legacy_wrapper",
        top_k=top_k,
        has_intent=bool(intent),
    )
    if not pipeline_enabled:
        return _legacy_search_sync(retriever, query=query, top_k=top_k, intent=intent)

    try:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return loop.run_until_complete(retriever.search_rag30(query, top_k, intent))
            return loop.run_until_complete(retriever.search_rag30(query, top_k, intent))
        except RuntimeError:
            return asyncio.run(retriever.search_rag30(query, top_k, intent))
    except Exception as exc:
        logger.error("search_sync_failed", error=str(exc))
        return []


def search(retriever, query: str, top_k: int = 3, intent: str | None = None):
    pipeline_enabled = bool(getattr(settings, "UPGRADE3_RETRIEVER_PIPELINE_ENABLED", True))
    logger.info(
        "retriever_pipeline_search",
        upgrade3_retriever_pipeline_enabled=pipeline_enabled,
        pipeline_path="pipeline" if pipeline_enabled else "legacy_wrapper",
        top_k=top_k,
        has_intent=bool(intent),
    )
    if not pipeline_enabled:
        return _legacy_search_sync(retriever, query=query, top_k=top_k, intent=intent)
    return search_sync(retriever, query=query, top_k=top_k, intent=intent)


def get_retriever():
    from app.rag.retriever import get_retriever as legacy_get_retriever

    return legacy_get_retriever()


__all__ = [
    "get_retriever",
    "search",
    "search_rag30",
    "search_sync",
]
