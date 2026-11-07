import pytest

from app.config import Settings
from app.model_gateway import ModelGateway


@pytest.mark.asyncio
async def test_hash_embedding_is_deterministic():
    settings = Settings(OPENCODE_API_KEY="test", SILICONFLOW_API_KEY="", OLLAMA_ENABLED=False)
    gateway = ModelGateway(settings)
    try:
        first = await gateway.embed(["高血压患者能吃阿司匹林吗"])
        second = await gateway.embed(["高血压患者能吃阿司匹林吗"])
        assert len(first[0]) == settings.EMBEDDING_DIM
        assert first == second
    finally:
        await gateway.close()


@pytest.mark.asyncio
async def test_rerank_fallback_order():
    settings = Settings(OPENCODE_API_KEY="test", SILICONFLOW_API_KEY="", OLLAMA_ENABLED=False)
    gateway = ModelGateway(settings)
    try:
        ranked = await gateway.rerank("头痛", ["文档A", "文档B"])
        assert ranked[0][0] == 0
        assert ranked[0][1] > ranked[1][1]
    finally:
        await gateway.close()
