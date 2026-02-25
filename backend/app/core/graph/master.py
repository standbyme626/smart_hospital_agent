"""
Compatibility wrapper for legacy imports.

The workflow graph is the single medical core graph used by both
external chat and internal evolution orchestration.
"""

from app.core.graph.workflow import create_agent_graph, app


def create_master_graph(checkpointer=None):
    """Backward-compatible alias to the unified medical core graph."""
    return create_agent_graph(checkpointer=checkpointer)

