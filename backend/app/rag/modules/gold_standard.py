import time
import structlog
from typing import List, Dict, Any, Optional
from pymilvus import Collection, connections, utility, FieldSchema, CollectionSchema, DataType
from app.core.config import settings
from app.services.embedding import EmbeddingService

logger = structlog.get_logger(__name__)

class GoldStandardManager:
    """
    金标准知识库管理器 (Gold Standard Knowledge Base).
    用于存储经过人工 (HITL) 审核和修正的高质量问答对，用于后续的 Few-shot Learning 或 RAG 增强。
    """
    def __init__(self, collection_name: str = "gold_standard"):
        self.collection_name = collection_name
        self.collection: Optional[Collection] = None
        self.embedding_service = EmbeddingService()
        self.dim = 1024 # Default Qwen-Embedding dim

    def connect(self):
        """连接并初始化集合"""
        try:
            if not connections.has_connection("default"):
                connections.connect(
                    alias="default",
                    host=settings.MILVUS_HOST,
                    port=settings.MILVUS_PORT
                )
            
            if utility.has_collection(self.collection_name):
                self.collection = Collection(self.collection_name)
                self.collection.load()
            else:
                self._create_collection()
                
            logger.info("gold_standard_collection_loaded", collection=self.collection_name)
        except Exception as e:
            logger.error("gold_standard_load_failed", error=str(e))

    def _create_collection(self):
        """创建集合 Schema"""
        logger.info("creating_gold_standard_collection")
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="question", dtype=DataType.VARCHAR, max_length=2048),
            FieldSchema(name="answer", dtype=DataType.VARCHAR, max_length=8192),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
            FieldSchema(name="modified_by", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="timestamp", dtype=DataType.INT64)
        ]
        schema = CollectionSchema(fields, "Gold Standard Medical QA Pairs from HITL")
        self.collection = Collection(self.collection_name, schema)
        
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        self.collection.create_index(field_name="vector", index_params=index_params)
        self.collection.load()

    def add_gold_sample(self, question: str, answer: str, modified_by: str = "expert_review"):
        """添加一条金标准样本"""
        if not question or not answer:
            return

        try:
            if self.collection is None:
                self.connect()
            
            # 1. Generate Embedding
            vector = self.embedding_service.get_embedding(question)
            
            # 2. Insert
            data = [
                [question],
                [answer],
                [vector],
                [modified_by],
                [int(time.time())]
            ]
            self.collection.insert(data)
            self.collection.flush() # Ensure visibility
            logger.info("gold_standard_added", question_preview=question[:20], modified_by=modified_by)
            
        except Exception as e:
            logger.error("gold_standard_add_failed", error=str(e))

    def search_similar(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """检索相似的金标准案例 (用于 Few-shot)"""
        try:
            if self.collection is None:
                self.connect()
                
            vector = self.embedding_service.get_embedding(query)
            
            results = self.collection.search(
                data=[vector],
                anns_field="vector",
                param={"metric_type": "COSINE", "params": {"nprobe": 10}},
                limit=top_k,
                output_fields=["question", "answer"]
            )
            
            hits = []
            for hits_i in results:
                for hit in hits_i:
                    hits.append({
                        "question": hit.entity.get("question"),
                        "answer": hit.entity.get("answer"),
                        "score": hit.score
                    })
            return hits
            
        except Exception as e:
            logger.error("gold_standard_search_failed", error=str(e))
            return []
