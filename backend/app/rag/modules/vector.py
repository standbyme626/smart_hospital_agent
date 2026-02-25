import time
import structlog
import asyncio
from typing import List, Dict, Any, Optional
from pymilvus import Collection, connections, utility
from app.core.config import settings
from app.services.embedding import EmbeddingService

logger = structlog.get_logger(__name__)

class VectorStoreManager:
    """
    向量数据库组件 (Vector Store Module - Milvus).
    负责管理 Milvus 连接、向量检索及文档入库。
    
    [Refactor Phase 2] Async Wrapper Added
    """
    def __init__(self, collection_name: str = "huatuo_knowledge"):
        self.collection_name = collection_name
        self.collection: Optional[Collection] = None
        self.embedding_service = EmbeddingService()

    def connect(self):
        """获取 Milvus 集合 (依赖全局 Infrastructure)"""
        try:
            # [Phase 5.5] 依赖全局连接，不再重复 connect
            if not connections.has_connection("default"):
                logger.warning("milvus_not_connected_global_fallback", host=settings.MILVUS_HOST)
                connections.connect(
                    alias="default",
                    host=settings.MILVUS_HOST,
                    port=settings.MILVUS_PORT
                )
            
            if utility.has_collection(self.collection_name):
                self.collection = Collection(self.collection_name)
                self.collection.load()
                logger.info("milvus_collection_loaded", collection=self.collection_name)
            else:
                 logger.warning("milvus_collection_missing", collection=self.collection_name)
            
        except Exception as e:
            logger.error("milvus_collection_load_failed_continuing_without_vector_store", error=str(e))
            self.collection = None
            # raise # [Fix] Do not raise, allow degradation

    async def asearch(self, query: str, top_k: int = 5, filter_expr: Optional[str] = None, filter_ids: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        """[Async] 执行向量相似度检索"""
        return await asyncio.to_thread(self.search, query, top_k, filter_expr, filter_ids)

    def search(self, query: str, top_k: int = 5, filter_expr: Optional[str] = None, filter_ids: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        """执行向量相似度检索 (包含向量化)"""
        query_vector = self.embedding_service.get_embedding(query)
        return self.search_by_vector(query_vector, top_k, filter_expr, filter_ids)

    async def asearch_by_vector(self, query_vector: list, top_k: int = 5, filter_expr: Optional[str] = None, filter_ids: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        """[Async] 直接使用向量执行检索"""
        return await asyncio.to_thread(self.search_by_vector, query_vector, top_k, filter_expr, filter_ids)

    def search_by_vector(self, query_vector: list, top_k: int = 5, filter_expr: Optional[str] = None, filter_ids: Optional[List[int]] = None) -> List[Dict[str, Any]]:

        """直接使用向量执行检索 (用于并行优化)"""
        if self.collection is None:
            self.connect()
            
        if self.collection is None:
            logger.error("vector_search_skipped", reason="collection_not_loaded")
            return []
        
        start = time.time()
        
        vector_field = "embedding"
        # ... 保持原逻辑 ...
        available_fields = []
        id_field = "milvus_id"
        if self.collection:
            available_fields = [f.name for f in self.collection.schema.fields]
            if "embedding" not in available_fields and "vector" in available_fields:
                vector_field = "vector"
            if "milvus_id" in available_fields:
                id_field = "milvus_id"
            elif "id" in available_fields:
                id_field = "id"
            else:
                # fallback to primary key field if available
                for field in self.collection.schema.fields:
                    if getattr(field, "is_primary", False):
                        id_field = field.name
                        break

        # 构造表达式
        expr = filter_expr
        if filter_ids:
            ids_str = ", ".join(str(i) for i in filter_ids)
            id_expr = f"{id_field} in [{ids_str}]"
            if expr:
                expr = f"({expr}) and ({id_expr})"
            else:
                expr = id_expr
        
        logger.info("[DEBUG] Milvus Search Params", 
                    vector_field=vector_field, 
                    id_field=id_field,
                    available_fields=available_fields,
                    expr=expr,
                    top_k=top_k)

        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        try:
            output_fields = ["content"]
            if "department" in available_fields:
                output_fields.append("department")
            results = self.collection.search(
                data=[query_vector],
                anns_field=vector_field, 
                param=search_params,
                limit=top_k,
                expr=expr,
                output_fields=output_fields
            )
        except Exception as e:
            logger.error("[DEBUG] Milvus Search Failed", error=str(e), schema=available_fields)
            raise
        
        hits = []
        for hit in results[0]:
            hits.append({
                "content": hit.entity.get("content"),
                "department": hit.entity.get("department"), # 新增
                "title": hit.entity.get("content"),
                "score": float(hit.score),
                "source": "vector",
                "id": hit.id
            })
        
        logger.info("vector_search_complete", count=len(hits), duration=f"{time.time()-start:.2f}s")
        return hits

    def add_documents(self, documents: List[Dict[str, Any]]) -> int:
        """
        向向量库添加文档 (Add Documents to Knowledge Base).
        
        Args:
            documents: List of dicts, each containing 'content' and optional metadata ('department', etc.)
        
        Returns:
            int: Number of documents inserted.
        """
        if not documents:
            return 0
            
        if self.collection is None:
            self.connect()
            
        texts = [doc["content"] for doc in documents]
        embeddings = self.embedding_service.batch_get_embeddings(texts)
        
        # 生成唯一 ID (使用当前时间戳微秒级 + 序列号)
        import time
        base_id = int(time.time() * 1000000)
        ids = [base_id + i for i in range(len(documents))]
        
        # 准备数据，对齐 Schema: [milvus_id, content, embedding, department]
        departments = [doc.get("department", "General") for doc in documents]
        
        # 截断 department 防止溢出
        departments = [d[:1000] for d in departments]
        
        entities = [
            ids,
            texts,
            embeddings,
            departments
        ]
        
        try:
            self.collection.insert(entities)
            self.collection.flush()
            logger.info("knowledge_ingest_success", count=len(documents))
            return len(documents)
        except Exception as e:
            logger.error("knowledge_ingest_failed", error=str(e))
            raise

    def batch_add(self, texts: List[str], metadatas: List[Dict[str, Any]]):
        """Deprecated: Use add_documents instead"""
        docs = []
        for t, m in zip(texts, metadatas):
            d = {"content": t}
            d.update(m)
            docs.append(d)
        self.add_documents(docs)

    def check_health(self) -> Dict[str, Any]:
        """[Startup] 健康检查"""
        status = {"status": "UNKNOWN", "details": ""}
        try:
            if not self.collection:
                self.connect()
            
            if self.collection:
                cnt = self.collection.num_entities
                status["status"] = "CONNECTED"
                status["details"] = f"Collection: {self.collection_name} | Count: {cnt}"
            else:
                status["status"] = "ERROR"
                status["details"] = "Collection not loaded"
        except Exception as e:
            status["status"] = "ERROR"
            status["details"] = str(e)
        return status

    def get_all_texts(self, limit: int = 2000) -> List[Dict[str, Any]]:
        """获取所有文本用于构建 BM25 索引 (Limit 2000 for performance)"""
        if self.collection is None:
            self.connect()
        
        try:
            available_fields = [f.name for f in self.collection.schema.fields] if self.collection else []
            id_field = "milvus_id" if "milvus_id" in available_fields else ("id" if "id" in available_fields else None)
            output_fields = ["content"]
            if "department" in available_fields:
                output_fields.append("department")
            if id_field:
                output_fields.append(id_field)

            expr = f"{id_field} >= 0" if id_field else "1 == 1"

            # Milvus Query with limit
            res = self.collection.query(
                expr=expr,
                limit=limit, 
                output_fields=output_fields
            )
            # 统一字段名映射，确保上层调用透明
            for item in res:
                if "content" in item:
                    item["text"] = item.pop("content")
                if "milvus_id" in item:
                    item["id"] = item.pop("milvus_id")
                elif "id" in item:
                    item["id"] = item.pop("id")
            return res
        except Exception as e:
            logger.error("vector_get_all_failed", error=str(e))
            return []
