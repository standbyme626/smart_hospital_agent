from __future__ import annotations

from typing import Any


async def run_tool_with_timeout(*, name: str, coro, timeout_s: float) -> tuple[Any, dict[str, Any]]:
    from app.core.graph.sub_graphs import diagnosis as legacy_diagnosis

    return await legacy_diagnosis._run_tool_with_timeout(name=name, coro=coro, timeout_s=timeout_s)


async def quick_query_hint(query: str) -> dict[str, Any]:
    from app.core.graph.sub_graphs import diagnosis as legacy_diagnosis

    return await legacy_diagnosis._quick_query_hint_tool(query)


async def quick_retrieval_hint(query: str, top_k: int) -> dict[str, Any]:
    from app.core.graph.sub_graphs import diagnosis as legacy_diagnosis

    return await legacy_diagnosis._quick_retrieval_hint_tool(query, top_k)


__all__ = [
    "quick_query_hint",
    "quick_retrieval_hint",
    "run_tool_with_timeout",
]
