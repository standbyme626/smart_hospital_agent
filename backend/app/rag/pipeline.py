from __future__ import annotations

from typing import Any

from app.rag import retriever as legacy_retriever


async def search_rag30(retriever, query: str, **kwargs: Any) -> Any:
    return await legacy_retriever.MedicalRetriever.search_rag30(retriever, query=query, **kwargs)


def search_sync(retriever, query: str, top_k: int = 3, intent: str | None = None):
    return legacy_retriever.MedicalRetriever.search_sync(retriever, query=query, top_k=top_k, intent=intent)


def search(retriever, query: str, top_k: int = 3, intent: str | None = None):
    return legacy_retriever.MedicalRetriever.search(retriever, query=query, top_k=top_k, intent=intent)


def get_retriever():
    return legacy_retriever.get_retriever()


__all__ = [
    "get_retriever",
    "search",
    "search_rag30",
    "search_sync",
]
