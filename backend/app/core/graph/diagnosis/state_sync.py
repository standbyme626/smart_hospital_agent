from __future__ import annotations

from .contracts import DiagnosisStateContract


async def state_sync(state: DiagnosisStateContract):
    from app.core.graph.sub_graphs import diagnosis as legacy_diagnosis

    return await legacy_diagnosis.state_sync_node(state)


state_sync_node = state_sync


__all__ = [
    "state_sync",
    "state_sync_node",
]
