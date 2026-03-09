import hashlib
import os
import sys
from types import SimpleNamespace

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)

from app.core.settings import runtime_side_effects
from app.rag.cache import build_query_hash
from app.rag.query_normalizer import QueryNormalizer, normalize_query


def test_query_normalizer_bridge_matches_existing_behavior() -> None:
    raw = "  头痛!? Fever.  "
    expected = "头痛 fever"
    assert normalize_query(raw) == expected
    assert QueryNormalizer.normalize(raw) == expected


def test_cache_query_hash_bridge_uses_normalize_and_namespace() -> None:
    expected_raw_hash = hashlib.md5("头痛".encode()).hexdigest()
    expected_query_hash = hashlib.md5(f"v2|{expected_raw_hash}".encode()).hexdigest()
    assert build_query_hash("  头痛!!! ", cache_namespace="v2") == expected_query_hash


def test_runtime_side_effects_bridge_helpers(monkeypatch) -> None:
    openai_key = "sk-" + ("a" * 29)
    dashscope_key = "sk-" + ("b" * 29)
    pooled_key = "sk-" + ("c" * 29)
    rotation_key = "sk-" + ("d" * 29)

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    candidates = runtime_side_effects.build_key_candidates(
        openai_api_key=openai_key,
        dashscope_api_key=dashscope_key,
        dashscope_api_key_pool=f"{pooled_key},invalid",
        api_key_rotation_list=f"{rotation_key},sk-placeholder",
    )
    assert set(candidates) >= {openai_key, dashscope_key, pooled_key, rotation_key}
    assert runtime_side_effects.is_valid_key(openai_key) is True
    assert runtime_side_effects.mask_key(openai_key).startswith("sk-aaaaa")

    settings_obj = SimpleNamespace(
        LANGCHAIN_TRACING_V2="false",
        LANGCHAIN_ENDPOINT="https://api.smith.langchain.com",
        LANGCHAIN_API_KEY="langsmith-key",
        LANGCHAIN_PROJECT="sha",
        OPENAI_API_BASE="https://dashscope.aliyuncs.com/compatible-mode/v1",
        OPENAI_API_KEY=openai_key,
    )
    runtime_side_effects.export_runtime_env(settings_obj)
    assert os.environ["OPENAI_BASE_URL"] == settings_obj.OPENAI_API_BASE
    assert os.environ["OPENAI_API_KEY"] == openai_key
    assert os.environ["HF_ENDPOINT"] == "https://hf-mirror.com"
