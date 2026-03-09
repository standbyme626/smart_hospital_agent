from __future__ import annotations

from typing import Any, Awaitable, Callable


async def recall_vector(
    *,
    query: str,
    top_k: int,
    recall_fn: Callable[..., Awaitable[list[dict[str, Any]]]],
) -> list[dict[str, Any]]:
    return await recall_fn(query=query, top_k=top_k)
