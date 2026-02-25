import os
import time
import threading
import asyncio
import hashlib
import functools
import structlog
import redis.asyncio as aioredis
import json
import re
from typing import List, Dict, Any, Optional

from app.core.config import settings, PROJECT_ROOT
from app.core.cache.redis_pool import get_redis_client, get_redis_binary_client
# from app.core.monitoring.metrics import track_rag_query, record_retrieval_metrics
from app.rag.modules.bm25 import BM25Indexer
from app.rag.modules.vector import VectorStoreManager
from app.rag.modules.semantic_cache import SemanticCacheManager
from app.rag.router import MedicalRouter
from app.rag.reranker import QwenReranker
from app.rag.sql_prefilter import SQLPreFilter
from app.rag.ddinter_checker import DDInterChecker
from app.core.models.local_slm import local_slm
from app.db.session import AsyncSessionLocal
from sqlalchemy import text
from langsmith import traceable, get_current_run_tree

from app.core.middleware.instrumentation import RAG_RETRIEVAL_LATENCY, RAG_CACHE_HIT, RAG_CACHE_MISS

logger = structlog.get_logger(__name__)

class MedicalRetriever:
    """
    医疗知识检索器 (RAG 3.0 Pipeline Coordinator).
    该类作为核心编排器，协调向量检索、词法检索、重排序及安全防护。
    
    架构特性:
    - 解耦设计: 词法与向量搜索分离至 modules 目录。
    - 异步友好: 耗时操作封装在 asyncio.to_thread 中。
    - 防御性编程: 包含层级安全拦截。
    - [New] 本地摘要化: 使用 Qwen3-0.6B 压缩长文本。
    """
    def __init__(self):
        # 1. 核心检索模块
        self.vector_store = VectorStoreManager()
        self.bm25_indexer = BM25Indexer(cache_dir=os.path.join(PROJECT_ROOT, "cache"))
        self.semantic_cache = SemanticCacheManager()
        
        # 2. 外部插件服务
        self.router = MedicalRouter()
        
        # 检查 Reranker 模型是否存在
        rerank_path = settings.RERANKER_MODEL_PATH
        if os.path.exists(rerank_path):
            self.reranker = QwenReranker(rerank_path)
        else:
            logger.warning("reranker_model_missing", path=rerank_path)
            self.reranker = None
            
        self.sql_filter = SQLPreFilter()
        self.ddinter_checker = DDInterChecker()
        
        # 3. 缓存设施 (Redis) - [V6.5.0] 切换为二进制客户端以支持 ZSTD 压缩
        try:
            self.redis_client = get_redis_binary_client()
            logger.info("retriever_cache_ready_binary_zstd_enabled")
        except Exception as e:
            logger.error("retriever_cache_init_failed", error=str(e))
            self.redis_client = None

        # 4. [Task 1.1] 强制同步初始化词法索引 (Blocking)
        # 禁止在索引未就绪时执行检索
        # [Fix] Deferred to async initialize() to avoid blocking or loop conflicts
        # self._init_lexical_index()
        logger.info("medical_retriever_instantiated")

    async def initialize(self):
        """异步初始化 (必须在 Server Startup 时调用)"""
        logger.info("medical_retriever_async_init_start")
        # Initialize BM25 in a thread to avoid blocking loop
        # [Refactor Phase 2] Use proper async load
        await self.bm25_indexer.load_index_async()
        logger.info("medical_retriever_async_init_complete")

    def _init_lexical_index_sync(self):
        """同步加载/构建索引 (Internal Sync Implementation)"""
        try:
            if not self.bm25_indexer.load_cache_sync():
                logger.warning("bm25_cache_miss_sync_building_start")
                try:
                    self.bm25_indexer.force_rebuild()
                    if self.bm25_indexer.is_ready:
                        logger.info("bm25_sync_build_success")
                    else:
                        logger.warning("bm25_sync_build_failed_check_logs")
                except Exception as build_err:
                    logger.error("bm25_sync_build_failed", error=str(build_err))
        except Exception as e:
            logger.error("bm25_init_error", error=str(e))
    
    # Deprecated: _init_lexical_index was renamed to _init_lexical_index_sync and logic simplified
    # to avoid checking for loop since it's now guaranteed to run in thread via initialize()


    def _normalize_query(self, query: str) -> str:
        """归一化查询语句用于生成缓存 Key (工程铁律)"""
        if not query:
            return ""
        # strip + lower + 去标点
        query = query.strip().lower()
        query = re.sub(r'[^\w\s\u4e00-\u9fa5]', '', query) # 保留中文字符、数字和字母
        return query

    async def analyze_intent(self, query: str) -> Dict[str, Any]:
        """
        [New] Intent Analysis using 0.6B Model.
        Classifies query into:
        - GREETING: Chit-chat, no retrieval needed.
        - MEDICAL: Medical query, needs retrieval.
        - OTHER: Out of scope.
        """
        try:
            # 简化 Prompt，减少模型出错几率
            prompt = f"分类该医疗咨询意图：GREETING, MEDICAL, OTHER\n咨询内容: \"{query}\"\n仅输出 JSON: {{\"category\": \"...\"}}"
            
            # [V8.1] 直接使用异步批量接口
            results = await local_slm.generate_batch_async(
                [prompt], 
                system_prompt="You are a medical intent classifier. Output JSON only.",
                thinking_mode=False
            )
            
            if not results:
                return {"category": "MEDICAL"}
                
            res = results[0]
            
            # [Fix] 更鲁棒的解析逻辑
            import re
            # 寻找 JSON 块
            match = re.search(r'(\{.*\})', res, re.DOTALL)
            if match:
                json_str = match.group(1)
                data = json.loads(json_str)
            else:
                # 备选逻辑：关键词匹配
                res_upper = res.upper()
                if "GREETING" in res_upper: return {"category": "GREETING"}
                if "OTHER" in res_upper: return {"category": "OTHER"}
                return {"category": "MEDICAL"}

            return data
        except Exception as e:
            logger.error("intent_analysis_failed", error=str(e), raw_res=locals().get('res', ''))
            return {"category": "MEDICAL"}

    @traceable(run_type="retriever", name="RAG30_Pipeline")
    async def search_rag30(self, query: str, top_k: int = 3, intent: str = None, return_debug: bool = False, skip_summarize: bool = False) -> Any:
        """
        [ASYNC] RAG 3.0 四层检索流水线 (支持 Semantic Cache 2.0)
        """
        try:
            logger.info("rag_search_start", query=query[:30])
            start_time = time.time()

            # [Task 1.4] 语义缓存检查 (Semantic Cache Check)
            cache_key = None
            norm_query = self._normalize_query(query)
            query_hash = hashlib.md5(norm_query.encode()).hexdigest()
            
            # 首先计算 embedding，语义缓存和后续检索都需要它
            emb_start = time.time()
            query_vector = await asyncio.to_thread(self.vector_store.embedding_service.get_embedding, query)
            embedding_latency = time.time() - emb_start
            
            print(f"[DEBUG] RAG Query: {query}")
            print(f"[DEBUG] Embedding Latency: {embedding_latency:.4f}s")

            if self.redis_client:
                # 尝试语义命中
                semantic_hit_hash = await self.semantic_cache.get_cache(query, query_vector=query_vector)
                # 如果语义命中，使用命中的 hash；否则使用当前的 hash
                effective_hash = semantic_hit_hash if semantic_hit_hash else query_hash
                cache_key = f"rag_cache:{effective_hash}"
                
                cached_data = await self.redis_client.get(cache_key)
                if cached_data:
                    logger.info("rag_cache_hit", query=query[:30], type="semantic" if semantic_hit_hash else "exact")
                    
                    # [Metrics] Cache Hit
                    RAG_CACHE_HIT.labels(cache_type="semantic" if semantic_hit_hash else "exact").inc()

                    # [Phase 6.5] ZSTD Decompress
                    decompressed = self.semantic_cache.decompress(cached_data)
                    results = json.loads(decompressed)
                    if return_debug:
                        return (results, {"cache_hit": True, "cache_type": "semantic" if semantic_hit_hash else "exact"})
                    return results
            
            # [Metrics] Cache Miss
            RAG_CACHE_MISS.labels(cache_type="semantic").inc()

            # [Task 2] Track Latencies
            metrics = {
                "retrieval_latency": 0.0,
                "rerank_latency": 0.0,
                "embedding_latency": embedding_latency, 
                "milvus_latency": 0.0,    # 新增: 追踪 Milvus 耗时
                "bm25_latency": 0.0,      # 新增: 追踪 BM25 耗时
                "metadata_filter_hit": False,
                "vector_score": 0.0,
                "rerank_score": 0.0,
                "context_length": 0,      # 新增: 追踪上下文长度
                "raw_results": [] 
            }
            
            # L1: 安全拦截 (Guardrail)
            if self.ddinter_checker and not self.ddinter_checker.check_query_safety(query):
                logger.warning("query_blocked_by_guardrail")
                res = [{"content": "您的查询可能涉及高风险用药建议，已被安全护栏拦截。", "score": 1.0, "source": "guardrail"}]
                return (res, metrics) if return_debug else res

            # [Task 1] Intent Analysis (0.6B Router)
            # Analyze intent before heavy retrieval
            intent_start = time.time()
            intent_result = await self.analyze_intent(query)
            metrics["intent_latency"] = time.time() - intent_start
            
            category = intent_result.get("category", "MEDICAL")
            if category == "GREETING":
                logger.info("intent_router_hit_greeting", query=query)
                res = [{"content": "您好！我是您的智能医疗助手。请问有什么可以帮您？", "score": 1.0, "source": "greeting"}]
                return (res, metrics) if return_debug else res
            elif category == "OTHER":
                 # Fallback to medical search anyway but with lower priority? Or just search.
                 # For now, treat as Medical to be safe, but maybe log it.
                 pass

            # Smart Pre-filtering: 提取科室 (异步优化)
            dept_ids, target_dept = await self._extract_department_filter_async(query)
            if dept_ids:
                metrics["metadata_filter_hit"] = True
            
            # L2: 混合召回 (Hybrid Search)
            # 优化：根据用户建议，召回阶段进行更细粒度的并行和监控
            retrieval_start = time.time()
            
            # [Optimization] 1. 异步 HyDE
            # HyDE Generation (0.6B)
            # Generate a hypothetical answer to boost recall for short queries
            hyde_doc = None
            if len(query) < 20: # Only apply HyDE for short/ambiguous queries
                try:
                    hyde_doc = await self._generate_hyde_doc(query)
                except Exception as e:
                    logger.warning("hyde_generation_failed", error=str(e))

            # Vectorize HyDE Doc (if exists) and Fuse
            final_vector = query_vector
            if hyde_doc:
                try:
                    hyde_vector = await asyncio.to_thread(self.vector_store.embedding_service.get_embedding, hyde_doc)
                    # Weighted Fusion: 70% Original + 30% HyDE
                    if isinstance(query_vector, list) and isinstance(hyde_vector, list):
                        final_vector = [0.7 * q + 0.3 * h for q, h in zip(query_vector, hyde_vector)]
                        logger.info("hyde_vector_fusion_applied")
                except Exception as e:
                    logger.warning("hyde_vectorization_failed", error=str(e))
            
            # [Optimization] 2. 并行执行 Milvus 向量搜索和 BM25 关键词搜索
            # Tiered Reranking L1: Broad Recall (Top 100)
            recall_window = 100 
            
            # Use final_vector for search
            # [Refactor Phase 2] Parallel Execution Flow Optimization
            # Ensure BM25 is ready (Lazy Check)
            if not self.bm25_indexer.is_ready:
                # If not ready, trigger async load but don't block heavily if possible
                # For now, we await it to ensure correctness
                await self.bm25_indexer.load_index_async()

            milvus_task = asyncio.to_thread(self.vector_store.search_by_vector, final_vector, top_k=recall_window, filter_ids=dept_ids)
            # BM25 is now CPU-bound but fast, run in thread to keep loop free
            bm25_task = asyncio.to_thread(self.bm25_indexer.search, query, top_k=recall_window)
            
            # 我们在此时并行启动两者
            milvus_start = time.time()
            v_results, b_results = await asyncio.gather(milvus_task, bm25_task)
            milvus_duration = time.time() - milvus_start
            
            # [Metrics] Latency Tracking
            RAG_RETRIEVAL_LATENCY.labels(stage="milvus_parallel").observe(milvus_duration)
            
            # RRF Fusion -> Tiered Reranking L2: Coarse Rank (Top 30)
            # Use RRF scores to filter down to 30 candidates for the heavy Cross-Encoder
            rerank_candidate_k = 30
            results = self._rrf_fusion(v_results, b_results, top_k=rerank_candidate_k, target_dept=target_dept)
            metrics["retrieval_latency"] = time.time() - retrieval_start
            
            if results:
                metrics["vector_score"] = results[0].get("score", 0.0)
                
                # Fetch content for reranking
                # ... (Logic to fetch content from SQL if needed)
                # ...

                metrics["raw_results"] = [r.copy() for r in results]
            
            # [Task 1.2] SQL 回查 (Critical Fix)
            # ... (保持原逻辑) ...
            if self.sql_filter and results:
                try:
                    milvus_ids = [r['id'] for r in results if r.get('id') is not None]
                    if milvus_ids:
                        async def _fetch_content_map_by_ids(ids_list):
                            if not ids_list:
                                return {}
                            try:
                                async with AsyncSessionLocal() as session:
                                    params = {f"id{i}": int(id_val) for i, id_val in enumerate(ids_list)}
                                    placeholders = ", ".join([f":id{i}" for i in range(len(ids_list))])
                                    query_variants = [
                                        f"SELECT milvus_id, content FROM medical_chunks WHERE milvus_id IN ({placeholders})",
                                        f"SELECT id, content FROM medical_chunks WHERE id IN ({placeholders})",
                                        f"SELECT CAST(chunk_id AS BIGINT), content FROM medical_chunks WHERE chunk_id IN ({placeholders})",
                                    ]
                                    for query in query_variants:
                                        try:
                                            stmt = text(query)
                                            result = await session.execute(stmt, params)
                                            rows = result.all()
                                            if rows:
                                                return {int(row[0]): row[1] for row in rows if row[0] is not None}
                                        except Exception as q_err:
                                            logger.warning("sql_fetch_variant_failed", error=str(q_err), query=query)
                                    return {}
                            except Exception as db_err:
                                logger.error("sql_fetch_error", error=str(db_err))
                                return {}

                        content_map = await _fetch_content_map_by_ids(milvus_ids)
                        for r in results:
                            m_id = r.get('id')
                            if m_id in content_map:
                                r['content'] = content_map[m_id]
                except Exception as e:
                    logger.error("sql_content_fetch_failed", error=str(e))

            # L3: 语义精排 (Adaptive Reranking)
            MAX_FINAL_K = top_k
            if self.reranker and len(results) > 0:
                # [Optimization] Adaptive Reranking
                skip_rerank = False
                if len(results) == 1:
                    skip_rerank = True
                elif results[0].get("score", 0) > 0.9 and (results[0].get("score", 0) - results[1].get("score", 0)) > 0.2:
                    skip_rerank = True
                    logger.info("adaptive_rerank_skipped", top_score=results[0].get("score"))

                if not skip_rerank:
                    rerank_start = time.time()
                    
                    # [Refactor Phase 2] Smart Truncation before Reranking
                    # Prevent OOM by truncating long documents
                    MAX_DOC_TOKENS = 512
                    truncated_results = []
                    for r in results:
                        content = r.get("content", "")
                        # Simple char approximation: 1 token approx 1.5 chars for Chinese
                        # 512 tokens ~ 800 chars
                        if len(content) > 800:
                            # Keep first 500 chars + last 300 chars (heuristic)
                            # Or just first 800
                            r["content_truncated"] = content[:800]
                        else:
                            r["content_truncated"] = content
                        truncated_results.append(r)
                    
                    # Pass truncated content to reranker, but keep original content in dict
                    # Note: QwenReranker.rerank likely uses 'content' key. 
                    # We need to temporarily swap content or modify reranker.
                    # Assuming reranker uses 'content'.
                    
                    # Swap content
                    for r in truncated_results:
                        r["_original_content"] = r["content"]
                        r["content"] = r["content_truncated"]
                        
                    # 传入所有召回结果进行精排，让 Reranker 决定哪些更相关
                    # Now sending safe length content
                    results = await asyncio.to_thread(self.reranker.rerank, query, truncated_results)
                    
                    # Restore content
                    for r in results:
                        if "_original_content" in r:
                            r["content"] = r["_original_content"]
                            del r["_original_content"]
                            del r["content_truncated"]
                            
                    rerank_duration = time.time() - rerank_start
                    metrics["rerank_latency"] = rerank_duration
                    
                    # [Metrics] Latency Tracking
                    RAG_RETRIEVAL_LATENCY.labels(stage="reranker").observe(rerank_duration)
                else:
                    metrics["rerank_latency"] = 0.0

                # [Optimization] SQL Meta Backfill
                # ... (保持原逻辑) ...
                missing_dept_ids = [r['id'] for r in results if r.get('department') in [None, 'Unknown', 'None', '', 'null']]
                if missing_dept_ids:
                    try:
                        async with AsyncSessionLocal() as session:
                            params = {f"id{i}": int(id_val) for i, id_val in enumerate(missing_dept_ids)}
                            placeholders = ", ".join([f":id{i}" for i in range(len(missing_dept_ids))])
                            query_variants = [
                                f"SELECT milvus_id, metadata->>'department' FROM medical_chunks WHERE milvus_id IN ({placeholders})",
                                f"SELECT id, department FROM medical_chunks WHERE id IN ({placeholders})",
                                f"SELECT id, metadata->>'department' FROM medical_chunks WHERE id IN ({placeholders})",
                            ]
                            db_metas = {}
                            for query in query_variants:
                                try:
                                    stmt = text(query)
                                    db_res = await session.execute(stmt, params)
                                    rows = db_res.all()
                                    if rows:
                                        db_metas = {int(row[0]): row[1] for row in rows if row[0] is not None}
                                        break
                                except Exception as q_err:
                                    logger.warning("sql_meta_variant_failed", error=str(q_err), query=query)
                        for r in results:
                            m_id = r.get('id')
                            if m_id in db_metas and db_metas[m_id]:
                                r['department'] = db_metas[m_id]
                    except Exception as e:
                        logger.error("sql_meta_backfill_failed", error=str(e))
                
                if results:
                    metrics["rerank_score"] = results[0].get("score", 0.0)

                # [Task 2.1] 动态阈值与阻断策略 (Dynamic Thresholding)
                BASE_REL_THRESHOLD = 0.15  # [Optimization] 进一步降低阈值以减少不必要的重试延迟
                GAP_THRESHOLD = 0.05
                
                if not results or results[0]['score'] < BASE_REL_THRESHOLD:
                    logger.warning("rerank_blocked_by_threshold_triggering_fallback", top_score=results[0]['score'] if results else 0)
                    
                    # [Task 2.2] 失败回退机制: Query Rewriting
                    # 仅在非调试模式且第一次检索失败时尝试一次重写
                    if not intent == "fallback_retry":
                        rewritten_query = await self._rewrite_query_async(query)
                        if rewritten_query and rewritten_query != query:
                            logger.info("rag_fallback_retry_start", rewritten_query=rewritten_query)
                            return await self.search_rag30(rewritten_query, top_k=top_k, intent="fallback_retry", return_debug=return_debug, skip_summarize=skip_summarize)
                    
                    return ([], metrics) if return_debug else []

                # 动态确定保留数量：如果后续结果与前一个分差很小，则保留
                final_count = 1
                for j in range(1, min(len(results), MAX_FINAL_K)):
                    if results[j]['score'] >= BASE_REL_THRESHOLD and \
                       (results[j-1]['score'] - results[j]['score']) < GAP_THRESHOLD:
                        final_count += 1
                    else:
                        break
                
                # 至少保留 top_k (如果分数达标)
                final_count = max(final_count, min(len(results), top_k))
                results = results[:final_count]
                logger.info("dynamic_k_selected", selected_k=final_count, top_score=results[0]['score'])
                
                print(f"[DEBUG] RAG Found {len(results)} docs. Top Score: {results[0]['score']:.4f}")
                for i, r in enumerate(results):
                    print(f"[DEBUG] Doc {i+1}: {r.get('content', '')[:100]}...")

            # DDI Warning Injection
            # ... (保持原逻辑) ...
            if self.ddinter_checker:
                # [V6.6.2 Fix] Use async method to avoid blocking event loop
                warnings = await self.ddinter_checker.scan_query_for_warnings_async(query)
                if warnings:
                    warning_msg = "\n".join(warnings)
                    results.insert(0, {
                        "content": f"【用药安全警示】\n{warning_msg}",
                        "score": 1.0,
                        "source": "guardrail_warning",
                        "id": -1
                    })

            # [Task 3.2] 上下文长度统计 (Context Monitoring)
            total_context_len = sum(len(r.get('content', '')) for r in results)
            metrics["context_length"] = total_context_len
            
            # [Task 1.3] Local Summarization
            final_results = results[:MAX_FINAL_K] # 这里的 final_count 已经限制了结果
            
            if final_results and not skip_summarize:
                # [Safety Guard] 检测最高分是否低于 0.3
                top_score = final_results[0].get('score', 0.0)
                low_confidence = top_score < 0.3
                
                # 2. 识别需要摘要的内容 (跳过警告信息或已处理内容)
                to_summarize_indices = []
                texts_to_summarize = []
                for i, r in enumerate(final_results):
                    # 仅对来自检索源且未被摘要的正文进行处理
                    if r.get('source') in ['reranked', 'vector', 'bm25'] and len(r.get('content', '')) > 200:
                        to_summarize_indices.append(i)
                        texts_to_summarize.append(r['content'])
                
                if texts_to_summarize:
                    # 3. 调用 summarize_batch 实现真正的 GPU 批量并行推理
                    try:
                        summaries = await self.summarize_batch(texts_to_summarize, query, low_confidence=low_confidence)
                        for idx, summary in zip(to_summarize_indices, summaries):
                            final_results[idx]['content'] = summary
                            final_results[idx]['source'] = f"{final_results[idx].get('source')}_summarized"
                            
                            # [Safety Guard] 强制注入兜底声明 (工程硬核拦截)
                            if low_confidence and "以上信息仅供参考" not in final_results[idx]['content']:
                                final_results[idx]['content'] += "\n\n以上信息仅供参考，具体诊疗请咨询线下医生。"
                    except Exception as sum_err:
                        logger.error("summarization_failed", error=str(sum_err))
                elif low_confidence:
                    # 如果没有触发摘要（文本太短），但也属于低置信度，手动为第一条结果追加声明
                    if final_results and "以上信息仅供参考" not in final_results[0]['content']:
                         final_results[0]['content'] += "\n\n以上信息仅供参考，具体诊疗请咨询线下医生。"

            duration = time.time() - start_time
            logger.info("rag_search_end", duration=f"{duration:.2f}s", result_count=len(final_results))
            
            # [Task 2] LangSmith Metadata Logging
            run = get_current_run_tree()
            if run:
                run.metadata.update(metrics)
            
            # 性能监控上报 (修正参数匹配: latency_ms, success)
            # track_rag_query(duration * 1000, True)
            
            # [Task 1.4] 写入缓存 (Cache Write) with ZSTD
            if self.redis_client and cache_key and final_results:
                try:
                    # [Phase 6.5] ZSTD Compress
                    compressed = self.semantic_cache.compress(json.dumps(final_results, ensure_ascii=False))
                    await self.redis_client.setex(cache_key, 86400, compressed)
                    # 更新语义缓存索引 (Milvus)
                    await asyncio.to_thread(self.semantic_cache.update_cache, query, effective_hash, query_vector=query_vector)
                    logger.info("rag_cache_write_success", key=cache_key)
                except Exception as cache_err:
                    logger.error("rag_cache_write_failed", error=str(cache_err))

            # Record total latency
            RAG_RETRIEVAL_LATENCY.labels(stage="total").observe(time.time() - start_time)

            return (final_results, metrics) if return_debug else final_results
        except Exception as outer_e:
            import traceback
            logger.error("rag_pipeline_critical_error", error=str(outer_e), stack=traceback.format_exc())
            return ([], {}) if return_debug else []

    async def _extract_department_filter_async(self, query: str) -> tuple[Optional[List[int]], Optional[str]]:
        """[ASYNC] 从 Query 中提取科室，并返回 (ID 列表, 科室名称)"""
        if not self.sql_filter:
            return None, None
            
        # 委托给 SQLPreFilter 提取科室
        filters = self.sql_filter.extract_filters(query)
        target_dept = filters.get("department")
        
        if target_dept:
            # 调用 SQLPreFilter 获取对应科室的 ID 列表 (异步)
            ids = await self.sql_filter.get_ids_by_department_async(target_dept)
            return ids, target_dept
        return None, None

    async def _rewrite_query_async(self, query: str) -> str:
        """[ASYNC] 使用本地轻量级模型进行查询重写"""
        try:
            prompt = f"你是一个医学助手。请将用户的口语化咨询转化为3个核心检索关键词，用空格隔开。只需回答关键词。\n用户提问：{query}\n关键词："
            # [Optimization] Use generate_batch for consistency with 1.7B API
            results = await local_slm.generate_batch_async(
                [prompt],
                system_prompt="You are a medical query rewriter."
            )
            rewritten = results[0].strip() if results else query
            # 简单清洗，去掉可能生成的序号或引导词
            rewritten = re.sub(r'^\d+\.\s*', '', rewritten)
            return rewritten if len(rewritten) > 2 else query
        except Exception as e:
            logger.error("query_rewrite_failed", error=str(e))
            return query

    def _extract_department_filter(self, query: str) -> tuple[Optional[List[int]], Optional[str]]:
        """从 Query 中提取科室，并返回 (ID 列表, 科室名称)"""
        if not self.sql_filter:
            return None, None
            
        # 委托给 SQLPreFilter 提取科室
        filters = self.sql_filter.extract_filters(query)
        target_dept = filters.get("department")
        
        if target_dept:
            # 调用 SQLPreFilter 获取对应科室的 ID 列表
            return self.sql_filter.get_ids_by_department(target_dept), target_dept
        return None, None

    def _heavy_lifting_search(self, query: str, top_k: int, intent: str, dept_ids: Optional[List[int]] = None, target_dept: str = None) -> List[Dict[str, Any]]:
        """Legacy 接口，不再被 search_rag30 直接调用"""
        v_results = self.vector_store.search(query, top_k=top_k * 2, filter_ids=dept_ids)
        b_results = self.bm25_indexer.search(query, top_k=top_k * 2)
        return self._rrf_fusion(v_results, b_results, top_k=top_k, target_dept=target_dept)

    def _rrf_fusion(self, vector_res: List[Dict], bm25_res: List[Dict], top_k: int, rrf_k: int = 60, target_dept: str = None) -> List[Dict]:
        """高质量融合算法 - 支持 ID 追踪"""
        # 使用 ID 作为唯一标识符进行融合，而不是 content (因为 content 可能会变)
        # 结果映射: id -> {score: float, content: str, id: int, source: str, department: str}
        fused_scores = {}
        
        # 处理向量结果
        for rank, res in enumerate(vector_res):
            doc_id = res.get('id', res.get('content'))
            if doc_id not in fused_scores:
                fused_scores[doc_id] = {
                    "score": 0.0,
                    "content": res.get('content'),
                    "id": res.get('id'),
                    "source": "vector",
                    "department": res.get('department') # 保留原始值，不默认 Unknown
                }
            fused_scores[doc_id]["score"] += 1.0 / (rrf_k + rank + 1)

        # 处理 BM25 结果
        for rank, res in enumerate(bm25_res):
            doc_id = res.get('id', res.get('content'))
            if doc_id not in fused_scores:
                fused_scores[doc_id] = {
                    "score": 0.0,
                    "content": res.get('content'),
                    "id": res.get('id'),
                    "source": "bm25",
                    "department": res.get('department')
                }
            else:
                # 补全科室信息
                if not fused_scores[doc_id].get("department") and res.get("department"):
                    fused_scores[doc_id]["department"] = res.get("department")
            
            fused_scores[doc_id]["score"] += 1.0 / (rrf_k + rank + 1)

        # [优化] 如果文档科室与建议科室匹配，给予加成
        if target_dept:
            for doc_id in fused_scores:
                doc = fused_scores[doc_id]
                doc_dept = doc.get("department", "")
                if doc_dept and (target_dept in doc_dept or doc_dept in target_dept):
                    doc["score"] *= 1.2  # 20% 的相关性加成
            
        # 排序并确保 department 字段存在（最后兜底）
        sorted_docs = sorted(fused_scores.values(), key=lambda x: x["score"], reverse=True)[:top_k]
        for doc in sorted_docs:
            if not doc.get("department"):
                doc["department"] = "Unknown"
        
        # 格式化输出
        return sorted_docs

    @traceable(run_type="retriever", name="RAG30_Sync_Wrapper")
    def search_sync(self, query: str, top_k: int = 3, intent: str = None) -> List[Dict[str, Any]]:
        """专供同步环境使用的 Wrapper"""
        try:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # 如果 loop 正在运行（例如在 async 任务中调用同步工具）
                    # 这是一个危险的操作，通常应该使用 asyncio.run_coroutine_threadsafe
                    # 但在单线程事件循环中，这会死锁。
                    # 不过 CrewAI 的 Agent 通常在单独的线程池中运行同步工具。
                    import threading
                    from concurrent.futures import Future
                    
                    def _run_coroutine(coro, loop, future):
                        asyncio.set_event_loop(loop)
                        try:
                            res = loop.run_until_complete(coro)
                            future.set_result(res)
                        except Exception as e:
                            future.set_exception(e)

                    # 检查是否已经在主线程/运行 loop 的线程中
                    # 如果在，则只能尝试 run_in_executor 或直接运行（如果 loop 没启动）
                    # 这里我们尝试最安全的方式：创建一个新的临时 loop (如果当前 loop 报错)
                    return loop.run_until_complete(self.search_rag30(query, top_k, intent))
                else:
                    return loop.run_until_complete(self.search_rag30(query, top_k, intent))
            except RuntimeError:
                # 针对 "There is no current event loop in thread" 或 "Event loop is closed"
                return asyncio.run(self.search_rag30(query, top_k, intent))
        except Exception as e:
            logger.error("search_sync_failed", error=str(e))
            # 最后的兜底：如果异步环境实在搞不定，返回空
            return []

    def search(self, query: str, top_k: int = 3, intent: str = None) -> List[Dict[str, Any]]:
        """Legacy 兼容接口"""
        return self.search_sync(query, top_k, intent)

    async def summarize_medical_text(self, text: str, query: str) -> str:
        """
        [Legacy Wrapper] 单条摘要 (保持兼容性)
        """
        return (await self.summarize_batch([text], query))[0]

    async def _generate_hyde_doc(self, query: str) -> str:
        """
        [New] HyDE (Hypothetical Document Embeddings) Generator.
        Uses 0.6B model to hallucinate a plausible medical answer to capture semantic meaning.
        """
        prompt = f"""You are an expert doctor. Briefly answer the patient's question.
Question: "{query}"
Answer:"""
        
        results = await local_slm.generate_batch_async(
            [prompt],
            system_prompt="You are a helpful medical assistant. Be concise.",
            thinking_mode=False
        )
        return results[0] if results else None

    async def summarize_batch(self, texts: List[str], query: str, low_confidence: bool = False) -> List[str]:
        """
        [Upgrade] Enhanced Summarization with Local 0.6B Model (Optimized for Latency)
        利用 0.6B 的长文本能力和推理能力处理检索回来的 RAG 文档。
        
        优化策略:
        1. Structured Prompt: 强制输出结构化信息，减少废话。
        2. Soft-Skip: 移除硬性长度校验，接受高置信度的短回答。
        3. No Retry: 移除重试逻辑，避免延迟倍增。
        """
        print(f"[DEBUG] Node MedicalRetriever.summarize_batch Start (Batch Size: {len(texts)}, Low Confidence: {low_confidence})")
        if not texts:
            return []
            
        # Feature Extraction Prompt Template
        prompts = []
        for t in texts:
            # 如果是低置信度，在 Prompt 中显式要求注入免责声明
            safety_instruction = ""
            if low_confidence:
                safety_instruction = "Note: The source text relevance is low. You MUST end with '以上信息仅供参考，具体诊疗请咨询线下医生'。"

            # 使用结构化 Prompt 引导模型
            p = f"""You are an expert medical researcher. Summarize the medical document based on the user's query.
Query: "{query}"

Document:
{t[:4000]}

Instructions:
1. Extract key symptoms, treatments, and warnings.
2. Be concise but comprehensive.
3. If the document is irrelevant, output "Irrelevant".
4. Output format: JSON with keys "summary" (string) and "relevance" (high/medium/low).
{safety_instruction}

JSON Output:"""
            prompts.append(p)

        try:
            # Single pass generation with strict timeout
            # Using thinking_mode=False for speed, or True if we really need reasoning. 
            # Given latency concerns, let's try False first or True with very short thinking? 
            # The user has 1.7B, which is fast. Let's keep thinking_mode=True but rely on the prompt to be concise.
            # Actually, for summarization, thinking might be overkill and cause the "too short" issue if it over-thinks.
            # Let's use thinking_mode=False for pure summarization tasks to speed up P95.
            # [Optimization] Increase timeout for local batch summary to 30s
            results = await asyncio.wait_for(
                local_slm.generate_batch_async(
                    prompts, 
                    system_prompt="You are a medical summarizer. Output valid JSON only.",
                    thinking_mode=False 
                ), 
                timeout=30.0
            )
            
            final_summaries = []
            for i, res in enumerate(results):
                try:
                    # 尝试解析 JSON
                    # 简单的清洗，去掉可能的 Markdown 代码块
                    clean_res = re.sub(r'```json|```', '', res).strip()
                    data = json.loads(clean_res)
                    summary = data.get("summary", "")
                    
                    # Soft Check
                    if len(summary) < 10 or data.get("relevance") == "low":
                         # 如果模型觉得不相关，或者生成失败，回退到截断原文
                         final_summaries.append(texts[i][:300] + "...")
                    else:
                        final_summaries.append(summary)
                except json.JSONDecodeError:
                    # 如果不是 JSON，尝试直接使用文本 (Fallback)
                    if len(res) > 20:
                        final_summaries.append(res)
                    else:
                        final_summaries.append(texts[i][:300] + "...")
                        
            return final_summaries
            
        except asyncio.TimeoutError:
            logger.error("local_batch_summary_timeout", duration="30s")
            return [t[:300] + "..." for t in texts]
            
        except Exception as e:
            logger.error("local_batch_summary_failed", error=str(e))
            return [t[:300] + "..." for t in texts]

# 单例管理
_retriever = None

def get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = MedicalRetriever()
    return _retriever
