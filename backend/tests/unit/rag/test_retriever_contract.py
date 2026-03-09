from types import SimpleNamespace
import os
import sys

import pytest

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)

from app.rag import retriever as retriever_module


@pytest.mark.asyncio
async def test_search_rag30_return_debug_metrics_contract() -> None:
    retriever = object.__new__(retriever_module.MedicalRetriever)
    retriever.vector_store = SimpleNamespace(
        embedding_service=SimpleNamespace(get_embedding=lambda _query: [0.1, 0.2, 0.3])
    )
    retriever.redis_client = None
    retriever.ddinter_checker = SimpleNamespace(check_query_safety=lambda _query: False)

    results, metrics = await retriever_module.MedicalRetriever.search_rag30(
        retriever,
        query="胸闷胸痛",
        top_k=2,
        return_debug=True,
        skip_intent_router=True,
        skip_hyde=True,
        skip_summarize=True,
        pure_mode=True,
    )

    assert isinstance(results, list)
    assert results and results[0].get("source") == "guardrail"

    required_metric_keys = {
        "retrieval_latency",
        "rerank_latency",
        "embedding_latency",
        "cache_hit",
        "cache_type",
        "context_length",
        "metadata_filter_hit",
        "raw_results",
        "pure_retrieval_mode",
        "cache_verify_enabled",
        "cache_verify_reason",
    }
    missing = sorted(required_metric_keys - set(metrics.keys()))
    assert not missing, f"metrics missing keys: {missing}"

    assert isinstance(metrics["retrieval_latency"], float)
    assert isinstance(metrics["rerank_latency"], float)
    assert isinstance(metrics["embedding_latency"], float)
    assert metrics["cache_hit"] is False
    assert metrics["cache_type"] == "miss"


def test_get_retriever_singleton_and_compat_methods(monkeypatch) -> None:
    class _FakeMedicalRetriever:
        async def search_rag30(self, query, **kwargs):
            del query, kwargs
            return []

        def search(self, query, top_k=3, intent=None):
            del query, top_k, intent
            return []

        def search_sync(self, query, top_k=3, intent=None):
            del query, top_k, intent
            return []

    monkeypatch.setattr(retriever_module, "MedicalRetriever", _FakeMedicalRetriever)
    monkeypatch.setattr(retriever_module, "_retriever", None)

    first = retriever_module.get_retriever()
    second = retriever_module.get_retriever()

    assert first is second
    assert hasattr(first, "search_rag30")
    assert hasattr(first, "search")
    assert hasattr(first, "search_sync")
