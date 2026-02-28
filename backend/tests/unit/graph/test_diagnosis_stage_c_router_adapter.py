import json
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


@pytest.mark.asyncio
async def test_stage_d_debug_snapshot_whitelist_and_phi_minimized(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_RETRIEVAL_ROUTER_ADAPTER", True, raising=False)
    monkeypatch.setattr(settings, "ENABLE_MULTI_QUERY", False, raising=False)
    monkeypatch.setattr(settings, "ENABLE_CONTEXT_WINDOW_AUTOMERGE", False, raising=False)
    monkeypatch.setattr(settings, "ENABLE_DEBUG_SNAPSHOT", True, raising=False)
    monkeypatch.setattr(settings, "DEBUG_INCLUDE_NODES", "", raising=False)
    monkeypatch.setattr(settings, "DIAGNOSIS_GRAPH_VERSION", "g-v1", raising=False)
    monkeypatch.setattr(settings, "DIAGNOSIS_DATA_CONTRACT_VERSION", "dc-v1", raising=False)
    monkeypatch.setattr(settings, "DIAGNOSIS_SCHEMA_VERSION", "v1", raising=False)

    mock_search = AsyncMock(return_value="router adapter context")
    monkeypatch.setattr(diagnosis.graph_rag_service, "search", mock_search)

    state = {
        "messages": [],
        "request_id": "req-stage-d",
        "debug_include_nodes": ["Hybrid_Retriever"],
        "retrieval_plan": {
            "primary_query": "plan query should not leak",
            "query_variants": ["plan query should not leak"],
            "top_k": 4,
            "index_scope": "section",
            "pure_mode": False,
            "enable_multi_query": False,
            "use_rerank": False,
            "rerank_threshold": 0.1,
            "skip_intent_router": True,
            "fusion_method": "rrf",
            "source_priority": ["vector", "graph"],
        },
    }

    result = await diagnosis.hybrid_retriever_node(state)

    called_kwargs = mock_search.await_args.kwargs
    assert called_kwargs["skip_intent_router"] is True
    assert called_kwargs["use_rerank"] is False
    assert called_kwargs["rerank_threshold"] == 0.1

    snapshots = result.get("debug_snapshots")
    assert isinstance(snapshots, dict)
    assert "Hybrid_Retriever" in snapshots
    snap = snapshots["Hybrid_Retriever"]
    assert snap["request_id"] == "req-stage-d"
    assert snap["graph_version"] == "g-v1"
    assert snap["node_version"] == "v2"
    assert snap["data_contract_version"] == "dc-v1"
    assert snap["schema_version"] == "v1"
    assert snap["payload"]["route_plan"]["query"]["hash"]
    assert "plan query should not leak" not in json.dumps(snap, ensure_ascii=False)


@pytest.mark.asyncio
async def test_stage_e_decision_judge_contract_and_router(monkeypatch):
    monkeypatch.setattr(settings, "DIAGNOSIS_DECISION_CONFIDENCE_THRESHOLD", 0.8, raising=False)
    monkeypatch.setattr(settings, "DIAGNOSIS_DECISION_MIN_EVIDENCE", 1, raising=False)

    low_state = {
        "last_tool_result": {
            "diagnosis": "疑似感染",
            "confidence": 0.45,
            "reasoning": "当前证据不足。",
        },
        "context_pack": {"evidence": []},
    }
    low_judge = await diagnosis.decision_judge_node(low_state)
    assert low_judge["decision_action"] == "retrieve_more"
    assert low_judge["decision_reason"] == "insufficient_evidence"
    assert low_judge["grounded_flag"] is False
    low_route = await diagnosis.confidence_evaluator_node({**low_state, **low_judge})
    assert low_route == "clarify_question"

    high_state = {
        "last_tool_result": {
            "diagnosis": "消化不良",
            "confidence": 0.92,
            "reasoning": "证据充分，建议消化内科评估。",
        },
        "context_pack": {
            "evidence": [
                {"doc_id": "doc-a", "chunk_id": "chunk-a", "content": "证据块"},
            ]
        },
    }
    high_judge = await diagnosis.decision_judge_node(high_state)
    assert high_judge["decision_action"] == "end_diagnosis"
    assert high_judge["decision_reason"] == "sufficient_evidence"
    assert high_judge["grounded_flag"] is True
    high_route = await diagnosis.confidence_evaluator_node({**high_state, **high_judge})
    assert high_route == "end_diagnosis"
