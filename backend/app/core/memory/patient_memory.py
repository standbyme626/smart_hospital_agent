
import time
from pymilvus import (
    connections, Collection, FieldSchema, CollectionSchema, 
    DataType, utility
)
from app.core.config import settings
from app.services.embedding import EmbeddingService

class PatientMemoryRetriever:
    """
    患者长期记忆检索器 (Patient Long-term Memory Retriever)
    基于 Milvus 存储患者的历史诊疗摘要，支持语义检索。
    """
    
    def __init__(self):
        self.milvus_host = settings.MILVUS_HOST
        self.milvus_port = settings.MILVUS_PORT
        self.collection_name = "patient_memories"
        self.embedding_service = EmbeddingService()
        self.collection = None
        
        try:
            self.connect_milvus()
            if connections.has_connection("default"):
                self._ensure_collection()
                self.collection = Collection(self.collection_name)
                self.collection.load()
        except Exception as e:
            print(f"[PatientMemory] Init failed: {e}. Running in degraded mode (No Memory).")

    def connect_milvus(self):
        try:
            if not connections.has_connection("default"):
                connections.connect("default", host=self.milvus_host, port=self.milvus_port)
        except Exception as e:
            print(f"[PatientMemory] Failed to connect to Milvus: {e}")

    def _ensure_collection(self):
        """确保集合存在，不存在则创建"""
        try:
            if utility.has_collection(self.collection_name):
                return

            print(f"[PatientMemory] Creating collection {self.collection_name}...")
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="patient_id", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=4096),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1024),
                FieldSchema(name="session_id", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="timestamp", dtype=DataType.INT64)
            ]
            schema = CollectionSchema(fields, "Patient Long-term Memory Storage")
            collection = Collection(self.collection_name, schema)
            
            # 创建索引
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
            collection.create_index("vector", index_params)
            
            # 为 patient_id 创建标量索引以加速过滤
            # Milvus will create an inverted index by default for scalar fields if we call create_index
            collection.create_index("patient_id", index_name="idx_patient_id")
            print(f"[PatientMemory] Collection {self.collection_name} created.")
        except Exception as e:
            print(f"[PatientMemory] Ensure collection failed: {e}")

    def add_memory(self, patient_id: str, session_id: str, content: str):
        """添加一条记忆"""
        if self.collection is None:
             print("[PatientMemory] Skipped add_memory (No Collection)")
             return

        try:
            vector = self.embedding_service.get_embedding(content)
            timestamp = int(time.time())
            
            entities = [
                [patient_id],   # patient_id
                [content],      # content
                [vector],       # vector
                [session_id],   # session_id
                [timestamp]     # timestamp
            ]
            
            self.collection.insert(entities)
            self.collection.flush() # Ensure data is visible
            print(f"[PatientMemory] Added memory for patient {patient_id}")
        except Exception as e:
            print(f"[PatientMemory] Add memory failed: {e}")

    def search_memory(self, patient_id: str, query: str, top_k: int = 5) -> list:
        """检索患者相关记忆"""
        if self.collection is None:
             print("[PatientMemory] Skipped search_memory (No Collection)")
             return []

        try:
            vector = self.embedding_service.get_embedding(query)
            
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 10}
            }
            
            # 使用 expr 过滤特定患者
            expr = f"patient_id == '{patient_id}'"
            
            results = self.collection.search(
                data=[vector],
                anns_field="vector",
                param=search_params,
                limit=top_k,
                expr=expr,
                output_fields=["content", "session_id", "timestamp"]
            )
            return results
        except Exception as e:
            print(f"[PatientMemory] Search memory failed: {e}")
            return []
        
        memories = []
        for hits in results:
            for hit in hits:
                memories.append({
                    "content": hit.entity.get("content"),
                    "session_id": hit.entity.get("session_id"),
                    "score": hit.distance,
                    "timestamp": hit.entity.get("timestamp")
                })
        
        return memories
