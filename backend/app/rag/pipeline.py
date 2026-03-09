from __future__ import annotations

import asyncio
from typing import Any, Dict, List

import structlog

logger = structlog.get_logger(__name__)


async def search_rag30(retriever, query: str, **kwargs: Any) -> Any:
    return await retriever.search_rag30(query=query, **kwargs)


def search_sync(
    retriever,
    query: str,
    top_k: int = 3,
    intent: str | None = None,
) -> List[Dict[str, Any]]:
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
