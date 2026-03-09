from __future__ import annotations


async def query_rewrite_node(state):
    from app.core.graph.sub_graphs import diagnosis as legacy_diagnosis

    return await legacy_diagnosis.query_rewrite_node(state)


__all__ = [
    "query_rewrite_node",
]
