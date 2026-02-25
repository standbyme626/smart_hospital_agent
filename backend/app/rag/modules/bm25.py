import os
import time
import asyncio
import pickle
import hashlib
import structlog
import multiprocessing
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
import jieba
import psycopg2
from app.core.config import settings

logger = structlog.get_logger(__name__)

def _tokenize_batch(texts: List[str]) -> List[List[str]]:
    """Helper for parallel tokenization"""
    return [list(jieba.cut(t)) for t in texts]

class BM25Indexer:
    """
    词法搜索组件 (Lexical Search Module - BM25).
    负责构建、保存、加载倒排索引及执行关键词搜索。
    """
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        self.bm25_instance: Optional[BM25Okapi] = None
        self.corpus: List[str] = []
        self.ids: List[int] = [] 
        self.departments: List[str] = [] # 新增: 存储对应的科室
        self.is_ready = False

    def get_checksum(self, texts: List[str]) -> str:
        content = "".join(texts[:1000]) # 采样摘要
        return hashlib.md5(content.encode()).hexdigest()

    def force_rebuild(self):
        """强制全量从 SQL 重建索引 (支持 54w 数据)"""
        logger.info("bm25_force_rebuild_start")
        start_time = time.time()
        
        try:
            # 1. Fetch data from SQL (medical_chunks)
            # Use psycopg2 for sync, direct access
            db_url = settings.DATABASE_URL.replace('+asyncpg', '')
            conn = psycopg2.connect(db_url)
            cur = conn.cursor() # Use server-side cursor if needed, but 54w fits in RAM (54w * 500 chars ~ 270MB)
            
            # Select milvus_id and content (or title from metadata if we want title-only indexing for speed)
            # Pipeline stores: Title in Milvus, Full content in SQL.
            # BM25 usually works better on Full content, but speed/memory is a concern.
            # Given requirement: "关键词漏失", full content is better.
            # However, Milvus vector stores "Title" (or enhanced title).
            # If we want RRF to match, we need consistency.
            # Let's use what Milvus has? No, instruction says "from PostgreSQL medical_chunks".
            # Let's index 'content' from SQL.
            
            logger.info("bm25_fetching_sql")
            rows = []
            query_variants = [
                # New schema
                "SELECT milvus_id, content, COALESCE(metadata->>'department', 'Unknown') FROM medical_chunks WHERE milvus_id IS NOT NULL",
                # Legacy schema in init.sql
                "SELECT id, content, COALESCE(department, 'Unknown') FROM medical_chunks WHERE id IS NOT NULL",
                # Mixed schema fallback
                "SELECT id, content, COALESCE(metadata->>'department', COALESCE(department, 'Unknown')) FROM medical_chunks WHERE id IS NOT NULL",
            ]
            for q in query_variants:
                try:
                    cur.execute(q)
                    rows = cur.fetchall()
                    if rows:
                        break
                except Exception as q_err:
                    logger.warning("bm25_sql_variant_failed", error=str(q_err), query=q)
            cur.close()
            conn.close()
            
            if not rows:
                logger.warning("bm25_rebuild_no_data")
                return

            self.ids = [r[0] for r in rows]
            self.corpus = [r[1] for r in rows]
            self.departments = [r[2] for r in rows]
            
            logger.info("bm25_tokenizing", count=len(self.corpus))
            
            # 2. Parallel Tokenization
            num_processes = min(multiprocessing.cpu_count(), 8)
            chunk_size = len(self.corpus) // num_processes + 1
            chunks = [self.corpus[i:i + chunk_size] for i in range(0, len(self.corpus), chunk_size)]

            tokenized_corpus = []
            try:
                with multiprocessing.Pool(processes=num_processes) as pool:
                    tokenized_chunks = pool.map(_tokenize_batch, chunks)
                for chunk in tokenized_chunks:
                    tokenized_corpus.extend(chunk)
            except Exception as mp_err:
                # 在受限环境（如容器/sandbox）中可能无法创建子进程，降级串行执行
                logger.warning("bm25_parallel_tokenize_failed_fallback_serial", error=str(mp_err))
                tokenized_corpus = [list(jieba.cut(doc)) for doc in self.corpus]
                
            # 3. Build BM25
            logger.info("bm25_indexing")
            self.bm25_instance = BM25Okapi(tokenized_corpus)
            
            # 4. Save Cache
            self._save_cache()
            self.is_ready = True
            
            logger.info("bm25_rebuild_complete", duration=f"{time.time()-start_time:.2f}s", doc_count=len(self.corpus))
            
        except Exception as e:
            logger.error("bm25_rebuild_failed", error=str(e))

    def _save_cache(self):
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            cache_path = os.path.join(self.cache_dir, "bm25_index.pkl")
            with open(cache_path, "wb") as f:
                pickle.dump({
                    "instance": self.bm25_instance, 
                    "corpus": self.corpus,
                    "ids": self.ids,
                    "departments": self.departments
                }, f)
        except Exception as e:
            logger.error("bm25_cache_save_failed", error=str(e))

    def build_from_milvus(self, milvus_results: List[Dict[str, Any]]):
        """从 Milvus 原始数据构建索引 (仅首次或数据变更时执行)"""
        # ... (Same logic, simplified to call _save_cache)
        logger.info("bm25_building_start", count=len(milvus_results))
        start = time.time()
        
        self.corpus = []
        self.ids = [] 
        
        for r in milvus_results:
            text = r.get('text') or r.get('content') or ''
            doc_id = r.get('id')
            
            if text:
                self.corpus.append(text)
                self.ids.append(doc_id if doc_id is not None else -1)
        
        if not self.corpus:
            logger.warning("bm25_build_skipped_empty_corpus")
            return

        tokenized_corpus = [list(jieba.cut(doc)) for doc in self.corpus]
        self.bm25_instance = BM25Okapi(tokenized_corpus)
        
        self._save_cache()
        self.is_ready = True
        logger.info("bm25_building_success", duration=f"{time.time()-start:.2f}s")

    async def load_index_async(self) -> bool:
        """
        [Refactor Phase 2] 异步懒加载索引
        Non-blocking initialization for high-performance startup.
        """
        if self.is_ready:
            return True
            
        logger.info("bm25_async_load_start")
        try:
            # 1. 尝试从缓存加载 (Run in thread pool)
            success = await asyncio.to_thread(self.load_cache_sync)
            if success:
                logger.info("bm25_async_load_success_from_cache")
                return True
                
            # 2. 如果缓存不存在，尝试重建 (Run in thread pool)
            # 注意：这在生产环境可能会比较慢，建议通过构建脚本预热
            logger.warning("bm25_cache_miss_triggering_rebuild")
            await asyncio.to_thread(self.force_rebuild)
            
            if self.is_ready:
                logger.info("bm25_async_rebuild_success")
                return True
            else:
                logger.error("bm25_async_rebuild_failed")
                return False
                
        except Exception as e:
            logger.error("bm25_async_load_error", error=str(e))
            return False

    def load_cache_sync(self) -> bool:
        """同步加载缓存 (阻塞式)"""
        if self.load_cache():
            return True
        
        # 自动触发重建 (Instruction 1)
        logger.warning("bm25_cache_missing_triggering_rebuild")
        self.force_rebuild()
        return self.is_ready
        
    def load_cache(self) -> bool:
        cache_path = os.path.join(self.cache_dir, "bm25_index.pkl")
        if not os.path.exists(cache_path):
            return False
        
        try:
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
                self.bm25_instance = data["instance"]
                self.corpus = data["corpus"]
                self.ids = data.get("ids", [-1] * len(self.corpus)) 
                self.departments = data.get("departments", ["Unknown"] * len(self.corpus))
            self.is_ready = True
            logger.info("bm25_cache_loaded")
            return True
        except Exception as e:
            logger.error("bm25_cache_load_failed", error=str(e))
            return False

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.is_ready or not self.bm25_instance:
            logger.warning("bm25_not_ready")
            return []
        
        tokenized_query = list(jieba.cut(query))
        scores = self.bm25_instance.get_scores(tokenized_query)
        top_n = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for i in top_n:
            results.append({
                "content": self.corpus[i],
                "score": float(scores[i]),
                "source": "bm25",
                "id": self.ids[i],
                "department": self.departments[i]
            })
        return results
