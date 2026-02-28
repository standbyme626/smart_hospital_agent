from unittest.mock import AsyncMock

import pytest

from app.core.config import settings
from app.core.graph.sub_graphs import diagnosis


class _FakeRetriever:
    def __init__(self):
        self.calls = []

    async def search_rag30(self, query, **kwargs):
        self.calls.append((query, kwargs))
        if query == "胃痛怎么办":
            return [
                {
                    "id": "a1",
                    "doc_id": "doc-a",
                    "chunk_id": "a1",
                    "source_id": "src-a",
                    "split_id": "split-a",
                    "split_idx": 1,
                    "score": 0.92,
                    "content": "A1 内容",
                    "child_ids": ["a2"],
                    "source": "vector",
                    "source_type": "guideline",
                },
                {
                    "id": "a2",
                    "doc_id": "doc-a",
                    "chunk_id": "a2",
                    "source_id": "src-a",
                    "split_id": "split-a",
                    "split_idx": 2,
                    "score": 0.90,
                    "content": "A2 邻近内容",
                    "parent_id": "a1",
                    "source": "vector",
                    "source_type": "guideline",
                },
            ]
        return [
            {
                "id": "b1",
                "doc_id": "doc-b",
                "chunk_id": "b1",
                "source_id": "src-b",
                "split_id": "split-b",
                "split_idx": 1,
                "score": 0.89,
                "content": "B1 内容",
                "source": "bm25",
                "source_type": "paper",
            }
        ]


@pytest.mark.asyncio
async def test_stage_b_switch_off_keeps_hybrid_message_behavior(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_MULTI_QUERY", False, raising=False)
    monkeypatch.setattr(settings, "ENABLE_CONTEXT_WINDOW_AUTOMERGE", False, raising=False)
    monkeypatch.setattr(settings, "CONTEXT_ORDERING_STRATEGY", "score_desc", raising=False)

    mock_search = AsyncMock(return_value="legacy context")
    monkeypatch.setattr(diagnosis.graph_rag_service, "search", mock_search)

    state = {
        "messages": [],
        "retrieval_query": "胃痛怎么办",
        "retrieval_query_variants": [{"text": "胃痛怎么办", "type": "original", "source": "rule", "weight": 1.0}],
        "retrieval_top_k": 3,
    }

    result = await diagnosis.hybrid_retriever_node(state)

    assert result["messages"][0].content == "Medical Context:\nlegacy context"
    assert "retrieval_query_variants" in result
    assert "variant_hits_map" in result
    assert "topk_source_ratio" in result
    assert "fusion_method" in result


@pytest.mark.asyncio
async def test_stage_b_switch_on_propagates_context_pack_contract(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_MULTI_QUERY", True, raising=False)
    monkeypatch.setattr(settings, "MULTI_QUERY_FUSION_METHOD", "weighted_rrf", raising=False)
    monkeypatch.setattr(settings, "MULTI_QUERY_RRF_K", 1, raising=False)
    monkeypatch.setattr(settings, "ENABLE_CONTEXT_WINDOW_AUTOMERGE", True, raising=False)
    monkeypatch.setattr(settings, "CONTEXT_ORDERING_STRATEGY", "lost_in_middle_mitigate", raising=False)
    monkeypatch.setattr(settings, "CONTEXT_DIVERSITY_MAX_PER_SOURCE", 1, raising=False)

    fake_retriever = _FakeRetriever()
    monkeypatch.setattr(diagnosis.graph_rag_service, "vector_retriever", fake_retriever)
    monkeypatch.setattr(diagnosis.graph_rag_service, "search", AsyncMock(return_value="hybrid context"))

    variants = [
        {"text": "胃痛怎么办", "type": "original", "source": "rule", "weight": 1.0},
        {"text": "胃痛 症状 原因 治疗", "type": "llm_expand", "source": "rule", "weight": 0.7},
    ]
    state = {
        "messages": [],
        "retrieval_query": "胃痛怎么办",
        "retrieval_query_variants": variants,
        "retrieval_top_k": 3,
    }

    result = await diagnosis.hybrid_retriever_node(state)

    assert result["retrieval_query_variants"] == variants
    assert "variant_hits_map" in result and isinstance(result["variant_hits_map"], dict)
    assert "topk_source_ratio" in result and isinstance(result["topk_source_ratio"], dict)
    assert "fusion_method" in result

    context_pack = result.get("context_pack")
    assert isinstance(context_pack, dict)
    assert isinstance(context_pack.get("evidence"), list)
    assert context_pack.get("ordering") == "lost_in_middle_mitigate"
    assert "fusion_method" in context_pack
    assert isinstance(context_pack.get("truncation"), dict)


@pytest.mark.asyncio
async def test_generate_report_with_guardrail_contract_fields(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_JSON_SCHEMA_GUARDRAIL", True, raising=False)
    monkeypatch.setattr(settings, "DIAGNOSIS_SCHEMA_VERSION", "v1", raising=False)

    state = {
        "last_tool_result": {
            "diagnosis": "消化不良",
            "confidence": 0.86,
            "reasoning": "结合症状与检索证据，倾向消化内科问题。",
            "department_top3": ["消化内科", "全科"],
        },
        "context_pack": {
            "evidence": [
                {
                    "doc_id": "doc-a",
                    "chunk_id": "a1",
                    "content": "胃部不适可先行消化内科评估。",
                }
            ],
            "ordering": "score_desc",
            "fusion_method": "weighted_rrf",
            "truncation": {"applied": False, "reason": "none"},
        },
    }

    output = await diagnosis.generate_report_node(state)

    diagnosis_output = output.get("diagnosis_output")
    assert isinstance(diagnosis_output, dict)
    assert diagnosis_output.get("diagnosis_schema_version") == "v1"
    assert isinstance(diagnosis_output.get("citations"), list)
    assert all({"doc_id", "chunk_id", "span"}.issubset(item.keys()) for item in diagnosis_output["citations"])
    assert "validated" in output
    assert "validation_error" in output
    assert "repair_attempted" in output
