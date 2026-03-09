from __future__ import annotations

import hashlib

from app.rag.modules.semantic_cache import SemanticCacheManager
from app.rag.query_normalizer import normalize_query


class RAGCache(SemanticCacheManager):
    """Upgrade3 compat bridge for legacy `app.rag.cache` import paths."""


CacheManager = SemanticCacheManager


def build_query_hash(query: str, cache_namespace: str) -> str:
    normalized = normalize_query(query)
    raw_query_hash = hashlib.md5(normalized.encode()).hexdigest()
    return hashlib.md5(f"{cache_namespace}|{raw_query_hash}".encode()).hexdigest()


__all__ = [
    "build_query_hash",
    "CacheManager",
    "RAGCache",
    "SemanticCacheManager",
]
