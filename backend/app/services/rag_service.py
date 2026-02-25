import asyncio
import json
import time
import structlog
from typing import List, Dict, Any, Optional
import redis.asyncio as redis
from pymilvus import Collection, connections

from app.core.config import settings
from app.services.embedding import EmbeddingService
from app.rag.modules.bm25 import BM25Indexer
from app.rag.reranker import QwenReranker

logger = structlog.get_logger(__name__)

class AsyncRAGService:
    """
    [PROJECT_DNA] Core RAG Service (Pure Async)
    Replaces legacy MedicalRetriever. 
    enforces:
    1. Native Async Driver usage (Redis)
    2. Hybrid Search (Vector + BM25 + RRF)
    3. Strict Reranking
    """
    
    def __init__(self):
        self.embedding_service = EmbeddingService()
        # Note: BM25 is CPU-bound, so we use it via thread poolExecutor in a managed way
        self.bm25_indexer = BM25Indexer(cache_dir=settings.PROJECT_ROOT + "/cache")
        # Load Reranker
        self.reranker = None
        if settings.RERANKER_MODEL_PATH:
             self.reranker = QwenReranker(settings.RERANKER_MODEL_PATH)
        
        # Async Redis Client
        self.redis = redis.from_url(
            settings.REDIS_URL, 
            encoding="utf-8", 
            decode_responses=True,
            max_connections=settings.REDIS_MAX_CONNECTIONS
        )
        
        # Milvus connection (Global)
        self._ensure_milvus_connection()
        self.collection = Collection("huatuo_knowledge") if connections.has_connection("default") else None

    def _ensure_milvus_connection(self):
        try:
            if not connections.has_connection("default"):
                connections.connect(
                    alias="default", 
                    host=settings.MILVUS_HOST, 
                    port=settings.MILVUS_PORT
                )
        except Exception as e:
            logger.error("milvus_connect_failed", error=str(e))

    async def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Main Search Entry Point
        """
        start_time = time.time()
        
        # 1. Parallel: Embedding + BM25
        # BM25 is CPU bound, run in thread
        bm25_future = asyncio.to_thread(self.bm25_indexer.search, query, top_k=50)
        
        # Embedding is I/O dependent (GPU/API), assume service handles it
        # If EmbeddingService is blocking, wrap it. It seems to be blocking in current codebase.
        query_vector = await asyncio.to_thread(self.embedding_service.get_embedding, query)
        
        # Milvus Search (Sync SDK -> Thread)
        # Sadly pymilvus standard SDK is sync. We wrap strict I/O here.
        milvus_future = asyncio.to_thread(
            self._search_milvus_sync, query_vector, top_k=50
        )
        
        bm25_res, vec_res = await asyncio.gather(bm25_future, milvus_future)
        
        # 2. RRF Fusion
        fusion_res = self._rrf_fusion(vec_res, bm25_res, top_k=20)
        
        # 3. Rerank
        if self.reranker and fusion_res:
            final_res = await asyncio.to_thread(self.reranker.rerank, query, fusion_res)
            # Cut to top_k
            final_res = final_res[:top_k]
        else:
            final_res = fusion_res[:top_k]
            
        logger.info("rag_search_complete", duration=time.time()-start_time, count=len(final_res))
        return final_res

    def _search_milvus_sync(self, vector: List[float], top_k: int) -> List[Dict]:
        if not self.collection: return []
        
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        results = self.collection.search(
            data=[vector], 
            anns_field="embedding", 
            param=search_params, 
            limit=top_k,
            output_fields=["content", "department"]
        )
        # Format
        hits = []
        for hit in results[0]:
            hits.append({
                "content": hit.entity.get("content"),
                "score": hit.score,
                "id": hit.id,
                "source": "vector"
            })
        return hits

    def _rrf_fusion(self, vec_res: List[Dict], bm25_res: List[Dict], top_k: int, k=60) -> List[Dict]:
        scores = {}
        for rank, item in enumerate(vec_res):
            doc_id = item['id']
            if doc_id not in scores:
                scores[doc_id] = {"item": item, "score": 0}
            scores[doc_id]["score"] += 1 / (k + rank + 1)
            
        for rank, item in enumerate(bm25_res):
            doc_id = item.get('id', item.get('content')) # BM25 might not have ID
            if doc_id not in scores:
                scores[doc_id] = {"item": item, "score": 0}
            scores[doc_id]["score"] += 1 / (k + rank + 1)
            
        sorted_items = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
        return [x["item"] for x in sorted_items[:top_k]]

# Singleton
_rag_service = None
def get_rag_service():
    global _rag_service
    if not _rag_service:
        _rag_service = AsyncRAGService()
    return _rag_service
