import pytest

from app.rag.adapters.multi_query_retriever_adapter import MultiQueryRetrieverAdapter
from app.rag.adapters.query_expander_adapter import QueryExpanderAdapter


def test_query_expander_budget_controls():
    adapter = QueryExpanderAdapter(
        max_variants=3,
        max_query_len_per_variant=8,
        rewrite_type_budget={"synonym": 1, "typo_fix": 1, "llm_expand": 0},
    )
    variants = adapter.expand(
        query="胃疼伴随恶心反酸怎么办",
        planned_variants=[
            "胃痛伴随恶心反酸怎么办",
            "胃疼 症状 原因 治疗 建议",
            {"text": "胃同伴随恶心反酸怎么办", "type": "typo_fix", "source": "rule", "weight": 0.9},
        ],
        enable_query_expansion=True,
    )

    assert len(variants) <= 3
    assert variants[0]["type"] == "original"
    assert all(len(item["text"]) <= 8 for item in variants)
    assert sum(1 for item in variants if item["type"] == "synonym") <= 1
    assert all(item["type"] != "llm_expand" for item in variants)


class _FakeRetriever:
    def __init__(self, mapping):
        self.mapping = mapping
        self.calls = []

    async def search_rag30(self, query, **kwargs):
        self.calls.append(query)
        rows = self.mapping.get(query, [])
        return [dict(item) for item in rows]


@pytest.mark.asyncio
async def test_multi_query_weighted_rrf_fusion_with_explainability():
    fake = _FakeRetriever(
        {
            "胃痛怎么办": [
                {"id": 1, "score": 0.92, "content": "doc-1", "source": "vector"},
                {"id": 2, "score": 0.81, "content": "doc-2", "source": "vector"},
            ],
            "胃痛 症状 原因 治疗": [
                {"id": 3, "score": 0.87, "content": "doc-3", "source": "bm25"},
                {"id": 1, "score": 0.79, "content": "doc-1", "source": "bm25"},
            ],
        }
    )
    adapter = MultiQueryRetrieverAdapter(retriever=fake, fusion_method="weighted_rrf", rrf_k=1)

    result = await adapter.retrieve(
        query="胃痛怎么办",
        retrieval_query_variants=[
            {"text": "胃痛怎么办", "type": "original", "source": "rule", "weight": 1.0},
            {"text": "胃痛 症状 原因 治疗", "type": "llm_expand", "source": "rule", "weight": 0.7},
        ],
        top_k=2,
        enable_multi_query=True,
        original_only=False,
        use_rerank=None,
        rerank_threshold=None,
        pure_mode=False,
    )

    assert result["fusion_method"] == "weighted_rrf"
    assert set(result["variant_hits_map"].keys()) == {"胃痛怎么办", "胃痛 症状 原因 治疗"}
    assert len(result["fused_docs"]) == 2
    assert [item["id"] for item in result["fused_docs"]] == [1, 3]
    assert result["topk_source_ratio"]["original"] == 0.5
    assert result["topk_source_ratio"]["expanded"] == 0.5


@pytest.mark.asyncio
async def test_multi_query_original_only_fallback():
    fake = _FakeRetriever({"胃痛怎么办": [{"id": 1, "score": 0.92, "content": "doc-1", "source": "vector"}]})
    adapter = MultiQueryRetrieverAdapter(retriever=fake, fusion_method="weighted_rrf", rrf_k=60)

    result = await adapter.retrieve(
        query="胃痛怎么办",
        retrieval_query_variants=[
            {"text": "胃痛怎么办", "type": "original", "source": "rule", "weight": 1.0},
            {"text": "胃痛 症状 原因 治疗", "type": "llm_expand", "source": "rule", "weight": 0.7},
        ],
        top_k=3,
        enable_multi_query=False,
        original_only=True,
        use_rerank=None,
        rerank_threshold=None,
        pure_mode=False,
    )

    assert result["fusion_method"] == "original_only"
    assert fake.calls == ["胃痛怎么办"]
    assert set(result["variant_hits_map"].keys()) == {"胃痛怎么办"}
