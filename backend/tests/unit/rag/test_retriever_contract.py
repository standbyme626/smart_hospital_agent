from types import SimpleNamespace
import os
import sys

import pytest

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)

from app.rag import retriever as retriever_module
from app.rag import pipeline as pipeline_module
from app.core.config import settings


@pytest.mark.asyncio
async def test_search_rag30_return_debug_metrics_contract(monkeypatch) -> None:
    async def _fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(retriever_module.asyncio, "to_thread", _fake_to_thread)

    retriever = object.__new__(retriever_module.MedicalRetriever)
    retriever.vector_store = SimpleNamespace(
        embedding_service=SimpleNamespace(get_embedding=lambda _query: [0.1, 0.2, 0.3])
    )
    retriever.redis_client = None
    retriever.ddinter_checker = SimpleNamespace(check_query_safety=lambda _query: False)

    search_impl = getattr(
        retriever_module.MedicalRetriever.search_rag30,
        "__wrapped__",
        retriever_module.MedicalRetriever.search_rag30,
    )
    results, metrics = await search_impl(
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


def test_retriever_contract_pipeline_flag_switches_sync_and_search(monkeypatch) -> None:
    retriever = object.__new__(retriever_module.MedicalRetriever)
    search_sync_impl = getattr(
        retriever_module.MedicalRetriever.search_sync,
        "__wrapped__",
        retriever_module.MedicalRetriever.search_sync,
    )
    monkeypatch.setattr(
        retriever_module.MedicalRetriever,
        "_legacy_search_sync_wrapper",
        lambda self, query, top_k=3, intent=None: [{"path": "legacy", "query": query, "top_k": top_k}],
    )

    monkeypatch.setattr(settings, "UPGRADE3_RETRIEVER_PIPELINE_ENABLED", True, raising=False)
    monkeypatch.setattr(
        retriever_module.retrieval_pipeline,
        "search_sync",
        lambda _retriever, query, top_k=3, intent=None: [{"path": "pipeline", "query": query, "top_k": top_k}],
    )
    monkeypatch.setattr(
        retriever_module.retrieval_pipeline,
        "search",
        lambda _retriever, query, top_k=3, intent=None: [{"path": "pipeline_search", "query": query, "top_k": top_k}],
    )
    sync_res_enabled = search_sync_impl(retriever, "头痛", top_k=2)
    search_res_enabled = retriever_module.MedicalRetriever.search(retriever, "头痛", top_k=2)
    assert isinstance(sync_res_enabled, list)
    assert isinstance(search_res_enabled, list)
    assert sync_res_enabled[0]["path"] == "pipeline"
    assert search_res_enabled[0]["path"] == "pipeline_search"

    monkeypatch.setattr(settings, "UPGRADE3_RETRIEVER_PIPELINE_ENABLED", False, raising=False)
    sync_res_disabled = search_sync_impl(retriever, "头痛", top_k=2)
    search_res_disabled = retriever_module.MedicalRetriever.search(retriever, "头痛", top_k=2)
    assert isinstance(sync_res_disabled, list)
    assert isinstance(search_res_disabled, list)
    assert sync_res_disabled[0]["path"] == "legacy"
    assert search_res_disabled[0]["path"] == "legacy"


def test_retriever_pipeline_module_respects_upgrade3_flag(monkeypatch) -> None:
    calls: list[str] = []

    class _FakeRetriever:
        def _legacy_search_sync_wrapper(self, query, top_k=3, intent=None):
            del query, top_k, intent
            calls.append("legacy")
            return [{"path": "legacy"}]

        async def search_rag30(self, query, top_k=3, intent=None):
            del query, top_k, intent
            calls.append("pipeline")
            return [{"path": "pipeline"}]

    fake = _FakeRetriever()

    monkeypatch.setattr(settings, "UPGRADE3_RETRIEVER_PIPELINE_ENABLED", False, raising=False)
    result_disabled = pipeline_module.search_sync(fake, query="胸痛", top_k=1)
    assert isinstance(result_disabled, list)
    assert result_disabled[0]["path"] == "legacy"
    assert calls == ["legacy"]

    calls.clear()
    monkeypatch.setattr(settings, "UPGRADE3_RETRIEVER_PIPELINE_ENABLED", True, raising=False)
    result_enabled = pipeline_module.search_sync(fake, query="胸痛", top_k=1)
    assert isinstance(result_enabled, list)
    assert result_enabled[0]["path"] == "pipeline"
    assert calls == ["pipeline"]
