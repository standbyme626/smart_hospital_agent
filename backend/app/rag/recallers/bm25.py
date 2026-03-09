from __future__ import annotations

from typing import Any, Callable


def recall_bm25(
    *,
    query: str,
    top_k: int,
    recall_fn: Callable[..., list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    return recall_fn(query=query, top_k=top_k)
