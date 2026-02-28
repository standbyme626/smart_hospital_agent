from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.config import settings
from app.rag.retriever import get_retriever

router = APIRouter()


class RagSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="用户查询")
    top_k: int = Field(3, ge=1, le=10, description="返回文档数量")
    use_rerank: bool = Field(True, description="是否启用重排")
    rerank_threshold: float = Field(0.15, ge=0.0, le=1.0, description="重排阈值")
    include_debug: bool = Field(False, description="是否返回调试指标")
    enable_intent_router: bool = Field(False, description="是否启用意图路由（默认关闭）")
    enable_hyde: bool = Field(False, description="是否启用HyDE增强（默认关闭）")
    pure_mode: bool = Field(False, description="是否启用纯检索模式（禁用LLM增强链路）")


class RagSearchResponse(BaseModel):
    query: str
    top_k: int
    use_rerank: bool
    rerank_threshold: float
    count: int
    results: List[Dict[str, Any]]
    debug: Optional[Dict[str, Any]] = None


@router.get("/status")
async def rag_status() -> Dict[str, Any]:
    retriever = get_retriever()
    return {
        "status": "ok",
        "bm25_ready": bool(getattr(retriever.bm25_indexer, "is_ready", False)),
        "reranker_loaded": bool(getattr(retriever, "reranker", None)),
    }


@router.post("/search", response_model=RagSearchResponse)
async def rag_search(request: RagSearchRequest) -> RagSearchResponse:
    retriever = get_retriever()
    threshold = request.rerank_threshold if request.use_rerank else 0.0
    pure_mode = bool(getattr(settings, "RAG_PURE_RETRIEVAL_MODE", False) or request.pure_mode)

    try:
        response = await retriever.search_rag30(
            query=request.query.strip(),
            top_k=request.top_k,
            intent="INFO",
            return_debug=request.include_debug,
            skip_summarize=True,
            use_rerank=request.use_rerank,
            rerank_threshold=request.rerank_threshold if request.use_rerank else None,
            # Keep standalone RAG endpoint modular: off by default, manually enabled by caller.
            skip_intent_router=not request.enable_intent_router,
            skip_hyde=not request.enable_hyde,
            pure_mode=pure_mode,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"RAG 查询失败: {exc}") from exc

    results: List[Dict[str, Any]] = []
    debug: Optional[Dict[str, Any]] = None
    if request.include_debug and isinstance(response, tuple):
        results, debug = response
    elif isinstance(response, list):
        results = response
    else:
        raise HTTPException(status_code=500, detail="RAG 返回结构异常")

    return RagSearchResponse(
        query=request.query,
        top_k=request.top_k,
        use_rerank=request.use_rerank,
        rerank_threshold=threshold,
        count=len(results),
        results=results,
        debug=debug,
    )
