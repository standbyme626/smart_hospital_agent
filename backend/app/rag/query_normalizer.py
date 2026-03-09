from __future__ import annotations

import re


_QUERY_SANITIZE_PATTERN = re.compile(r"[^\w\s\u4e00-\u9fa5]")


def normalize_query(query: str) -> str:
    if not query:
        return ""
    normalized = query.strip().lower()
    return _QUERY_SANITIZE_PATTERN.sub("", normalized)


class QueryNormalizer:
    @staticmethod
    def normalize(query: str) -> str:
        return normalize_query(query)


__all__ = [
    "normalize_query",
    "QueryNormalizer",
]
