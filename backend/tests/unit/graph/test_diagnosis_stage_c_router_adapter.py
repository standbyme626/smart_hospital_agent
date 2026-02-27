from unittest.mock import AsyncMock

import pytest

from app.core.config import settings
from app.core.graph.sub_graphs import diagnosis


@pytest.mark.asyncio
async def test_stage_c_router_adapter_resolves_retrieval_plan_without_breaking_pure_mode(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_RETRIEVAL_ROUTER_ADAPTER", True, raising=False)
    monkeypatch.setattr(settings, "ENABLE_MULTI_QUERY", True, raising=False)
    monkeypatch.setattr(settings, "MULTI_QUERY_FUSION_METHOD", "weighted_rrf", raising=False)
    monkeypatch.setattr(settings, "ENABLE_CONTEXT_WINDOW_AUTOMERGE", False, raising=False)
    monkeypatch.setattr(settings, "RAG_DISABLE_INTENT_ROUTER_WHEN_PURE", True, raising=False)

    mock_search = AsyncMock(return_value="router adapter context")
    monkeypatch.setattr(diagnosis.graph_rag_service, "search", mock_search)

    state = {
        "messages": [],
        "retrieval_query": "state query should be overridden",
        "retrieval_query_variants": [{"text": "state query should be overridden", "type": "original", "source": "state", "weight": 1.0}],
        "retrieval_top_k": 2,
        "retrieval_plan": {
            "primary_query": "plan query wins",
            "query_variants": ["plan query wins"],
            "top_k": 5,
            "index_scope": "section",
            "pure_mode": True,
            "enable_multi_query": False,
            "fusion_method": "rrf",
            "source_priority": ["vector", "graph", "hierarchical"],
        },
    }

    result = await diagnosis.hybrid_retriever_node(state)

    called_kwargs = mock_search.await_args.kwargs
    assert called_kwargs["query"] == "plan query wins"
    assert called_kwargs["top_k"] == 5
    assert called_kwargs["index_scope"] == "section"
    assert called_kwargs["pure_mode"] is True

    assert result["rag_pure_mode"] is True
    assert result["retrieval_query"] == "plan query wins"
    assert result["retrieval_top_k"] == 5
    assert result["retrieval_plan"]["route_source"] == "router_adapter"
    assert result["retrieval_plan"]["route_mode"] == "pure"
    assert result["retrieval_plan"]["fusion_method"] == "original_only"
    assert "pure_retrieval_result" in result
