from app.rag.adapters.retrieval_router_adapter import RetrievalRouterAdapter


def test_router_adapter_disabled_keeps_legacy_state_behavior():
    adapter = RetrievalRouterAdapter(enabled=False)
    state = {
        "retrieval_query": "state query",
        "retrieval_top_k": 4,
        "retrieval_index_scope": "section",
        "retrieval_query_variants": [{"text": "state query", "type": "original", "source": "state", "weight": 1.0}],
        "rag_pure_mode": False,
        "retrieval_plan": {
            "primary_query": "plan query",
            "top_k": 8,
            "pure_mode": True,
            "enable_multi_query": False,
        },
    }

    resolved = adapter.resolve(
        state=state,
        fallback_query="fallback query",
        query_source="message",
        default_top_k=3,
        default_index_scope="paragraph",
        default_fusion_method="weighted_rrf",
        default_enable_multi_query=True,
        default_pure_mode=False,
        disable_intent_router_when_pure=True,
    )

    assert resolved["route_source"] == "legacy_state"
    assert resolved["query"] == "state query"
    assert resolved["top_k"] == 4
    assert resolved["index_scope"] == "section"
    assert resolved["pure_mode"] is False


def test_router_adapter_enabled_prefers_retrieval_plan_contract():
    adapter = RetrievalRouterAdapter(enabled=True)
    state = {
        "retrieval_plan": {
            "primary_query": "plan query",
            "query_variants": ["plan query", "expanded query"],
            "top_k": 6,
            "index_scope": "document",
            "pure_mode": True,
            "enable_multi_query": False,
            "fusion_method": "rrf",
            "source_priority": ["vector", "hierarchical"],
        },
    }

    resolved = adapter.resolve(
        state=state,
        fallback_query="fallback query",
        query_source="message",
        default_top_k=3,
        default_index_scope="paragraph",
        default_fusion_method="weighted_rrf",
        default_enable_multi_query=True,
        default_pure_mode=False,
        disable_intent_router_when_pure=True,
    )

    assert resolved["route_source"] == "router_adapter"
    assert resolved["query"] == "plan query"
    assert resolved["top_k"] == 6
    assert resolved["index_scope"] == "document"
    assert resolved["pure_mode"] is True
    assert resolved["enable_multi_query"] is False
    assert resolved["fusion_method"] == "rrf"
    assert resolved["source_priority"] == ["vector", "hierarchical"]
    assert resolved["skip_intent_router"] is True
    assert [item["text"] for item in resolved["retrieval_query_variants"]] == ["plan query", "expanded query"]
