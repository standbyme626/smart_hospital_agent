import time
import structlog
import hashlib
import zstandard as zstd
from typing import List, Dict, Any, Optional
from pymilvus import Collection, connections, utility, FieldSchema, CollectionSchema, DataType
from app.core.config import settings
from app.services.embedding import EmbeddingService
from app.core.cache.redis_pool import get_redis_client
from app.core.monitoring.metrics import CACHE_HITS, CACHE_MISSES

logger = structlog.get_logger(__name__)

class SemanticCacheManager:
    """
    语义缓存管理器 (Semantic Cache 2.0 - Milvus + Redis).
    利用向量相似度加速重复或相似医疗问题的响应。
    """
    def __init__(self, collection_name: str = "semantic_cache", threshold: float = 0.96):
        self.collection_name = collection_name
        self.threshold = threshold
        self.collection: Optional[Collection] = None
        self.embedding_service = EmbeddingService()
        self._next_connect_retry_ts: float = 0.0
        self._connect_retry_interval_s: float = 30.0
        # [Phase 5.5] 动态获取维度
        self.dim = self.embedding_service.dim if hasattr(self.embedding_service, 'dim') else 1024
        # [Phase 6.5] 初始化压缩器
        self.compressor = zstd.ZstdCompressor(level=3)
        self.decompressor = zstd.ZstdDecompressor()

    def compress(self, data: str) -> bytes:
        """压缩 JSON 字符串"""
        return self.compressor.compress(data.encode('utf-8'))

    def decompress(self, data: bytes) -> str:
        """解压为 JSON 字符串"""
        return self.decompressor.decompress(data).decode('utf-8')

    def connect(self):
        """获取 Milvus 集合 (依赖全局 Infrastructure)"""
        now = time.time()
        if now < self._next_connect_retry_ts and self.collection is None:
            logger.warning(
                "semantic_cache_connect_backoff_active",
                retry_after_s=round(self._next_connect_retry_ts - now, 2),
                collection=self.collection_name,
            )
            return

        try:
            # [Phase 5.5] 依赖全局连接
            if not connections.has_connection("default"):
                connections.connect(
                    alias="default",
                    host=settings.MILVUS_HOST,
                    port=settings.MILVUS_PORT,
                    timeout=1.5,
                )
            
            if not utility.has_collection(self.collection_name, timeout=1.5):
                self._create_collection()
            
            self.collection = Collection(self.collection_name)
            self.collection.load(timeout=1.5)
            self._next_connect_retry_ts = 0.0
            logger.info("semantic_cache_collection_loaded", collection=self.collection_name)
        except Exception as e:
            self._next_connect_retry_ts = time.time() + self._connect_retry_interval_s
            logger.error("semantic_cache_load_failed", error=str(e))

    def _create_collection(self):
        """创建语义缓存集合架构"""
        # [Fix] 动态获取维度，确保与 EmbeddingService 一致
        if hasattr(self.embedding_service, 'dim'):
            self.dim = self.embedding_service.dim
        else:
            # 尝试通过预热获取维度
            try:
                test_emb = self.embedding_service.get_embedding("test")
                self.dim = len(test_emb)
            except:
                self.dim = 1024 # Fallback

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="query_hash", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="query_text", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="query_vector", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
            FieldSchema(name="updated_at", dtype=DataType.DOUBLE)
        ]
        schema = CollectionSchema(fields, description="Semantic cache for medical queries")
        
        # 创建集合
        collection = Collection(self.collection_name, schema)
        
        # 创建索引 (HNSW 适合快速查找)
        index_params = {
            "metric_type": "IP", # 内积相似度，因为 Embedding 已经归一化
            "index_type": "HNSW",
            "params": {"M": 8, "efConstruction": 64}
        }
        collection.create_index(field_name="query_vector", index_params=index_params)
        logger.info("semantic_cache_collection_created", collection=self.collection_name)

    async def get_cache(self, query: str, query_vector: Optional[List[float]] = None) -> Optional[str]:
        """
        通过语义搜索查找缓存 Key
        Returns: cache_key (MD5 hash) or None
        """
        if self.collection is None:
            self.connect()
        
        if self.collection is None:
            return None

        start = time.time()
        if query_vector is None:
            import asyncio
            query_vector = await asyncio.to_thread(self.embedding_service.get_embedding, query)
        
        search_params = {"metric_type": "IP", "params": {"ef": 32}}
        
        try:
            results = self.collection.search(
                data=[query_vector],
                anns_field="query_vector",
                param=search_params,
                limit=1,
                output_fields=["query_hash", "query_text"]
            )
            
            if results and len(results[0]) > 0:
                hit = results[0][0]
                score = hit.score
                if score >= self.threshold:
                    logger.info("semantic_cache_hit", 
                                score=f"{score:.4f}", 
                                original=hit.entity.get("query_text"), 
                                current=query[:30])
                    CACHE_HITS.inc()
                    return hit.entity.get("query_hash")
            
            logger.info("semantic_cache_miss", query=query[:30])
            CACHE_MISSES.inc()
            return None
        except Exception as e:
            logger.error("semantic_cache_search_failed", error=str(e))
            return None
        finally:
            # 记录耗时 (毫秒)
            duration = (time.time() - start) * 1000
            if duration > 100: # 超过 100ms 记录警告
                logger.warning("semantic_cache_search_slow", duration_ms=f"{duration:.2f}")

    def update_cache(self, query: str, query_hash: str, query_vector: Optional[List[float]] = None):
        """
        异步更新语义缓存 (由外部调用)
        """
        if self.collection is None:
            self.connect()
            
        if self.collection is None:
            return

        try:
            if query_vector is None:
                query_vector = self.embedding_service.get_embedding(query)
            
            now = time.time()
            data = [
                [query_hash],
                [query[:512]], # 截断
                [query_vector],
                [now]
            ]
            self.collection.insert(data)
            
            # [Phase 6.1] 自动清理过期数据 (Simple LRU logic: delete items older than 7 days)
            # 在实际生产中，这应该由定时任务完成，这里通过概率触发减少性能开销
            import random
            if random.random() < 0.01: # 1% 概率触发清理
                seven_days_ago = now - (7 * 24 * 3600)
                expr = f"updated_at < {seven_days_ago}"
                self.collection.delete(expr)
                logger.info("semantic_cache_cleanup_triggered", expr=expr)

            logger.debug("semantic_cache_inserted", query_hash=query_hash)
        except Exception as e:
            logger.error("semantic_cache_update_failed", error=str(e))
