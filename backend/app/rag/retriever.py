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
from app.core.constants import DEPARTMENT_LIST, SYMPTOM_KEYWORD_MAP
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
        self._department_alias_patterns = self._build_department_alias_patterns()
        self._department_keyword_index = self._build_department_keyword_index()
        
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

    def _build_overlap_terms(self, text: str) -> set[str]:
        """构建轻量词项集合，用于语义缓存命中后的相关性校验。"""
        normalized = self._normalize_query(text)
        if not normalized:
            return set()

        terms: set[str] = set()
        for token in re.findall(r"[a-z0-9]+|[\u4e00-\u9fa5]+", normalized):
            if not token:
                continue
            # 中文采用 2-gram，减少单字误判（如“头疼”与“脚趾头痛”）
            if re.fullmatch(r"[\u4e00-\u9fa5]+", token):
                if len(token) == 1:
                    terms.add(token)
                else:
                    for i in range(len(token) - 1):
                        terms.add(token[i : i + 2])
                continue

            # 英文/数字词按原词保留
            if len(token) >= 2:
                terms.add(token)

        return terms

    def _default_dept_post_debug(self) -> Dict[str, Any]:
        return {
            "dept_normalize_enabled": bool(getattr(settings, "RAG_DEPT_NORMALIZE_ON_RESULT", True)),
            "dept_normalize_changed_count": 0,
            "dept_normalize_unknown_count": 0,
            "dept_normalize_reason": "not_checked",
            "dept_gate_enabled": bool(getattr(settings, "RAG_DEPT_CONSISTENCY_GATE_ON", True)),
            "dept_gate_applied": False,
            "dept_gate_reordered": False,
            "dept_gate_reason": "not_checked",
            "dept_gate_top1_support": 0,
            "dept_gate_max_support": 0,
            "dept_gate_support_candidate_idx": None,
            "dept_gate_support_candidate_gap": None,
            "dept_gate_support_hits": [],
            "dept_gate_effective_max_gap": None,
            "dept_top1_before": None,
            "dept_top1_after": None,
        }

    def _build_department_alias_patterns(self) -> List[tuple[int, str, str]]:
        """构建科室别名模式，priority 越小优先级越高。"""
        patterns: List[tuple[int, str, str]] = []

        def add_alias(alias: str, canonical: str, priority: int) -> None:
            alias_clean = str(alias or "").strip().lower()
            canonical_clean = str(canonical or "").strip()
            if alias_clean and canonical_clean:
                patterns.append((priority, alias_clean, canonical_clean))

        manual_aliases = {
            "心血管内科": "心内科",
            "cardiology": "心内科",
            "精神科": "精神内科",
            "心理科": "精神内科",
            "心理内科": "精神内科",
            "psychiatry": "精神内科",
            "mental health": "精神内科",
            "普通外科": "普外科",
            "general surgery": "普外科",
            "泌尿外科": "泌尿科",
            "urology": "泌尿科",
            "中医科": "中医",
            "tcm": "中医",
            "耳鼻咽喉科": "耳鼻喉科",
            "ent": "耳鼻喉科",
            "全科": "内科",
            "通用": "内科",
            "general": "内科",
            "unknown": "Unknown",
            "none": "Unknown",
            "null": "Unknown",
            "指标解读": "Unknown",
            "植物科": "Unknown",
        }
        for alias, canonical in manual_aliases.items():
            add_alias(alias, canonical, priority=0)

        for dept in DEPARTMENT_LIST:
            priority = 2 if dept in {"内科", "外科"} else 1
            add_alias(dept, dept, priority=priority)

        patterns.sort(key=lambda item: (item[0], -len(item[1])))
        return patterns

    def _normalize_department_name(self, raw_department: Any) -> tuple[str, str]:
        raw = str(raw_department or "").strip()
        if not raw:
            return "Unknown", "empty"

        normalized = raw.lower().replace("（", "(").replace("）", ")")
        for _, alias, canonical in self._department_alias_patterns:
            if alias and alias in normalized:
                return canonical, f"alias:{alias}"

        for dept in DEPARTMENT_LIST:
            if dept in raw:
                return dept, f"token:{dept}"

        if len(raw) > 18 or "归类" in raw:
            return "Unknown", "long_unmapped_text"

        return raw, "passthrough"

    def _build_department_keyword_index(self) -> Dict[str, set[str]]:
        """由症状关键词反推科室关键词，用于轻量一致性重排。"""
        index: Dict[str, set[str]] = {}
        for keyword, dept in SYMPTOM_KEYWORD_MAP.items():
            canonical, _ = self._normalize_department_name(dept)
            if canonical != "Unknown":
                index.setdefault(canonical, set()).add(str(keyword).strip().lower())

        extra_keywords = {
            "耳鼻喉科": {"耳鸣", "耳聋", "耳痛", "耳朵", "嗡嗡响", "鼻塞", "鼻出血", "流鼻血", "鼻窦炎", "擤鼻涕", "咽痛", "喉咙"},
            "心内科": {"胸痛", "胸闷", "心悸", "心慌", "心率", "血压"},
            "神经内科": {"头痛", "头疼", "头晕", "眩晕", "抽搐", "抽筋", "癫痫", "口吐白沫", "麻木", "晕倒", "偏头痛"},
            "消化内科": {"胃痛", "胃疼", "腹痛", "腹泻", "恶心", "呕吐", "便秘", "腹胀", "反酸", "肝病", "肝炎", "肝硬化", "喝酒", "饮酒", "酒后", "食欲不振", "胃口不好", "吃不下"},
            "呼吸内科": {"气短", "咳痰", "咽炎", "呼吸", "哮喘"},
            "妇科": {"白带", "阴道", "经期", "月经", "宫颈"},
            "妇产科": {"孕", "妊娠", "产检", "分娩"},
            "泌尿科": {"尿频", "尿急", "尿痛", "血尿"},
            "骨科": {"关节", "腰痛", "腿痛", "骨折"},
            "皮肤科": {"皮疹", "红斑", "湿疹", "皮炎"},
            "儿科": {"小儿", "儿童", "婴儿", "宝宝", "孩子", "小孩", "儿子", "女儿", "周岁", "个月", "多动症", "不吃饭"},
            "精神内科": {"焦虑", "抑郁", "强迫", "强迫症", "失眠", "情绪", "烦躁", "惊恐", "恐惧", "精神", "心理"},
            "外科": {"肛瘘", "肛门", "脓水", "血管损伤", "损伤", "创伤"},
            "内科": {"内分泌失调", "乏力", "无力"},
        }
        for dept, keywords in extra_keywords.items():
            canonical, _ = self._normalize_department_name(dept)
            if canonical == "Unknown":
                continue
            index.setdefault(canonical, set()).update({k.strip().lower() for k in keywords if k})

        return index

    def _department_keyword_support(self, query: str, canonical_dept: str) -> tuple[int, List[str]]:
        keywords = self._department_keyword_index.get(canonical_dept, set())
        if not query or not keywords:
            return 0, []
        hits = sorted({kw for kw in keywords if kw and kw in query}, key=len, reverse=True)
        return len(hits), hits[:5]

    def _extract_query_department_mentions(self, norm_query: str) -> List[str]:
        hits: List[tuple[int, int, str]] = []
        seen = set()
        for _, alias, canonical in self._department_alias_patterns:
            if not alias or canonical == "Unknown" or canonical in seen:
                continue
            pos = norm_query.find(alias)
            if pos < 0:
                continue
            hits.append((pos, -len(alias), canonical))
            seen.add(canonical)
        hits.sort(key=lambda x: (x[0], x[1]))
        ordered = [x[2] for x in hits]

        # 纯检索中很多 query 不直接说科室名，补充高置信症状词反推科室。
        keyword_candidates: List[tuple[int, int, str]] = []
        for dept, keywords in self._department_keyword_index.items():
            if dept in seen:
                continue
            kw_hits = [kw for kw in keywords if kw and kw in norm_query]
            if not kw_hits:
                continue
            hit_count = len(set(kw_hits))
            max_kw_len = max((len(kw) for kw in kw_hits), default=0)
            # 高置信规则：>=2 个词命中，或命中 >=3 字关键词（如“内分泌失调”“口吐白沫”）。
            if hit_count >= 2 or max_kw_len >= 3:
                keyword_candidates.append((-(hit_count + max_kw_len), -max_kw_len, dept))
                seen.add(dept)

        keyword_candidates.sort()
        ordered.extend([x[2] for x in keyword_candidates])
        return ordered

    def _default_pure_dept_penalty_debug(self, execution_path: str = "not_executed") -> Dict[str, Any]:
        return {
            "pure_dept_penalty_applied": False,
            "pure_dept_penalty_reordered": False,
            "pure_dept_penalty_reason": "not_checked",
            "pure_dept_penalty_query_mentions": [],
            "pure_dept_penalty_top1_before": None,
            "pure_dept_penalty_top1_after": None,
            "pure_dept_penalty_exec_path": execution_path,
        }

    def _default_pure_candidate_comp_debug(self) -> Dict[str, Any]:
        return {
            "pure_dept_candidate_comp_applied": False,
            "pure_dept_candidate_comp_reason": "not_checked",
            "pure_dept_candidate_comp_mentions": [],
            "pure_dept_candidate_comp_mention_source": "none",
            "pure_dept_candidate_comp_reserved_count": 0,
            "pure_dept_candidate_comp_injected_count": 0,
        }

    def _extract_support_department_mentions(self, norm_query: str, top_k: int = 3) -> List[str]:
        if not norm_query:
            return []

        scored: List[tuple[int, int, int, str]] = []
        for dept in self._department_keyword_index.keys():
            support, hits = self._department_keyword_support(norm_query, dept)
            if support <= 0:
                continue
            max_kw_len = max((len(hit) for hit in hits), default=0)
            # 保守信号阈值：优先多命中；单命中仅接受较长症状词，且避免泛科室误触发。
            if support >= 2:
                pass
            elif support == 1 and max_kw_len >= 3 and dept not in {"内科", "外科"}:
                pass
            elif max_kw_len >= 4:
                pass
            else:
                continue
            generic_bias = 0 if dept in {"内科", "外科"} else 1
            scored.append((support, max_kw_len, generic_bias, dept))

        scored.sort(reverse=True)
        ordered: List[str] = []
        seen = set()
        for _, _, _, dept in scored:
            if dept in seen:
                continue
            seen.add(dept)
            ordered.append(dept)
            if len(ordered) >= max(1, int(top_k)):
                break
        return ordered

    def _select_pure_rerank_candidates(
        self,
        query: str,
        fused_results: List[Dict[str, Any]],
        rerank_candidate_k: int,
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        debug = self._default_pure_candidate_comp_debug()
        if not isinstance(fused_results, list) or not fused_results:
            debug["pure_dept_candidate_comp_reason"] = "empty_fused_results"
            return fused_results, debug

        safe_k = max(1, int(rerank_candidate_k))
        if len(fused_results) <= safe_k:
            debug["pure_dept_candidate_comp_reason"] = "insufficient_pool"
            return fused_results[:safe_k], debug

        norm_query = self._normalize_query(query)
        if not norm_query:
            debug["pure_dept_candidate_comp_reason"] = "empty_query"
            return fused_results[:safe_k], debug

        mentions = self._extract_query_department_mentions(norm_query)
        if not mentions:
            fallback_mentions = self._extract_support_department_mentions(norm_query, top_k=3)
            if fallback_mentions:
                mentions = fallback_mentions
                debug["pure_dept_candidate_comp_mention_source"] = "support_fallback"
            else:
                debug["pure_dept_candidate_comp_reason"] = "no_query_department_mention"
                return fused_results[:safe_k], debug
        else:
            debug["pure_dept_candidate_comp_mention_source"] = "query_direct"
        debug["pure_dept_candidate_comp_mentions"] = mentions[:5]

        max_reserved = max(1, safe_k // 2)
        max_reserved = min(max_reserved, len(mentions), 3)
        reserved_indices: List[int] = []
        for dept in mentions:
            best_idx = None
            for idx, doc in enumerate(fused_results):
                if idx in reserved_indices:
                    continue
                canonical_dept, _ = self._normalize_department_name(doc.get("department"))
                if canonical_dept == dept:
                    best_idx = idx
                    break
            if best_idx is not None:
                reserved_indices.append(best_idx)
            if len(reserved_indices) >= max_reserved:
                break

        if not reserved_indices:
            debug["pure_dept_candidate_comp_reason"] = "no_matching_dept_docs"
            return fused_results[:safe_k], debug

        reserved_set = set(reserved_indices)
        ranked = sorted(
            enumerate(fused_results),
            key=lambda item: (
                1 if item[0] in reserved_set else 0,
                float(item[1].get("score", 0.0)),
            ),
            reverse=True,
        )
        selected_indices = [int(idx) for idx, _ in ranked[:safe_k]]
        selected = [fused_results[idx] for idx in selected_indices]

        injected = sum(1 for idx in selected_indices if idx >= safe_k)
        debug["pure_dept_candidate_comp_reserved_count"] = len(reserved_indices)
        debug["pure_dept_candidate_comp_injected_count"] = injected
        if injected > 0:
            debug["pure_dept_candidate_comp_applied"] = True
            debug["pure_dept_candidate_comp_reason"] = "applied"
        else:
            debug["pure_dept_candidate_comp_reason"] = "reserved_already_in_topk"

        return selected, debug

    def _apply_pure_department_mismatch_penalty(
        self,
        query: str,
        results: List[Dict[str, Any]],
        execution_path: str = "retrieval",
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        debug = self._default_pure_dept_penalty_debug(execution_path=execution_path)
        if not isinstance(results, list) or len(results) < 2:
            debug["pure_dept_penalty_reason"] = "insufficient_candidates"
            return results, debug

        norm_query = self._normalize_query(query)
        if not norm_query:
            debug["pure_dept_penalty_reason"] = "empty_query"
            return results, debug

        query_mentions = self._extract_query_department_mentions(norm_query)
        mention_set = set(query_mentions)
        debug["pure_dept_penalty_query_mentions"] = query_mentions[:5]
        if not mention_set:
            debug["pure_dept_penalty_reason"] = "no_query_department_mention"
            return results, debug
        if mention_set.issubset({"内科", "外科"}):
            debug["pure_dept_penalty_reason"] = "generic_query_department_only"
            return results, debug

        debug["pure_dept_penalty_top1_before"] = str(results[0].get("department", "") or "")

        hard_penalty = 0.08
        soft_penalty = 0.03
        scored: List[Dict[str, Any]] = []
        penalty_count = 0
        for idx, doc in enumerate(results):
            canonical_dept, _ = self._normalize_department_name(doc.get("department"))
            doc["department"] = canonical_dept
            base_score = float(doc.get("score", 0.0))
            support, _ = self._department_keyword_support(norm_query, canonical_dept)
            penalty = 0.0
            if canonical_dept not in mention_set:
                penalty = hard_penalty if support <= 0 else soft_penalty
            adjusted_score = max(0.0, base_score - penalty)
            if penalty > 0:
                penalty_count += 1
                doc["score_raw"] = base_score
                doc["score"] = round(adjusted_score, 6)
            scored.append(
                {
                    "idx": idx,
                    "base_score": base_score,
                    "adjusted_score": adjusted_score,
                    "penalty": penalty,
                }
            )

        if penalty_count == 0:
            debug["pure_dept_penalty_reason"] = "no_mismatch_to_penalize"
            return results, debug

        debug["pure_dept_penalty_applied"] = True
        reordered = sorted(
            scored,
            key=lambda item: (float(item["adjusted_score"]), float(item["base_score"])),
            reverse=True,
        )
        new_order = [int(item["idx"]) for item in reordered]
        if new_order != list(range(len(results))):
            results = [results[i] for i in new_order]
            debug["pure_dept_penalty_reordered"] = True
            debug["pure_dept_penalty_reason"] = "applied"
        else:
            debug["pure_dept_penalty_reason"] = "applied_no_order_change"

        debug["pure_dept_penalty_top1_after"] = str(results[0].get("department", "") or "")
        return results, debug

    def _apply_department_normalization(
        self,
        results: List[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        enabled = bool(getattr(settings, "RAG_DEPT_NORMALIZE_ON_RESULT", True))
        debug: Dict[str, Any] = {
            "dept_normalize_enabled": enabled,
            "dept_normalize_changed_count": 0,
            "dept_normalize_unknown_count": 0,
            "dept_normalize_reason": "not_checked",
        }
        if not enabled:
            debug["dept_normalize_reason"] = "disabled"
            return results, debug

        changed = 0
        unknown_count = 0
        for doc in results:
            raw_dept = str(doc.get("department", "") or "").strip()
            canonical_dept, _ = self._normalize_department_name(raw_dept)
            if raw_dept and raw_dept != canonical_dept:
                doc["department_raw"] = raw_dept
                changed += 1
            doc["department"] = canonical_dept
            if canonical_dept == "Unknown":
                unknown_count += 1

        debug["dept_normalize_reason"] = "applied"
        debug["dept_normalize_changed_count"] = changed
        debug["dept_normalize_unknown_count"] = unknown_count
        return results, debug

    def _apply_department_consistency_gate(
        self,
        query: str,
        results: List[Dict[str, Any]],
        pure_mode: bool = False,
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        enabled = bool(getattr(settings, "RAG_DEPT_CONSISTENCY_GATE_ON", True))
        debug: Dict[str, Any] = {
            "dept_gate_enabled": enabled,
            "dept_gate_applied": False,
            "dept_gate_reordered": False,
            "dept_gate_reason": "not_checked",
            "dept_gate_top1_support": 0,
            "dept_gate_max_support": 0,
            "dept_gate_support_candidate_idx": None,
            "dept_gate_support_candidate_gap": None,
            "dept_gate_support_hits": [],
            "dept_gate_effective_max_gap": None,
        }
        if not enabled:
            debug["dept_gate_reason"] = "disabled"
            return results, debug
        if len(results) < 2:
            debug["dept_gate_reason"] = "insufficient_candidates"
            return results, debug

        norm_query = self._normalize_query(query)
        if not norm_query:
            debug["dept_gate_reason"] = "empty_query"
            return results, debug

        bonus = max(0.0, float(getattr(settings, "RAG_DEPT_CONSISTENCY_BONUS", 0.05)))
        if pure_mode:
            # pure 检索无改写/回退，门控需更积极地修正“top3中有正确科室但 top1 错配”。
            bonus = max(bonus, 0.12)
        max_gap = max(0.0, float(getattr(settings, "RAG_DEPT_CONSISTENCY_MAX_SCORE_GAP", 0.12)))
        min_hits = max(1, int(getattr(settings, "RAG_DEPT_CONSISTENCY_MIN_KEYWORD_HITS", 1)))
        require_top1_unsupported = bool(getattr(settings, "RAG_DEPT_CONSISTENCY_REQUIRE_TOP1_UNSUPPORTED", True))

        scored: List[Dict[str, Any]] = []
        query_mentioned_departments = set()
        for _, alias, canonical in self._department_alias_patterns:
            if alias and canonical != "Unknown" and alias in norm_query:
                query_mentioned_departments.add(canonical)
        for idx, doc in enumerate(results):
            canonical_dept = str(doc.get("department") or "Unknown")
            support, hits = self._department_keyword_support(norm_query, canonical_dept)
            base_score = float(doc.get("score", 0.0))
            adjusted_score = base_score + min(3, support) * bonus
            if canonical_dept == "Unknown":
                adjusted_score -= bonus
            mentioned_in_query = canonical_dept in query_mentioned_departments
            if pure_mode and mentioned_in_query:
                adjusted_score += bonus * 0.8
            scored.append(
                {
                    "idx": idx,
                    "dept": canonical_dept,
                    "base_score": base_score,
                    "adjusted_score": adjusted_score,
                    "support": support,
                    "hits": hits,
                    "mentioned_in_query": mentioned_in_query,
                }
            )

        if not scored:
            debug["dept_gate_reason"] = "empty_scores"
            return results, debug

        top1 = scored[0]
        debug["dept_gate_top1_support"] = int(top1["support"])
        debug["dept_gate_max_support"] = int(max(item["support"] for item in scored))

        if require_top1_unsupported and int(top1["support"]) >= min_hits:
            debug["dept_gate_reason"] = "top1_supported"
            return results, debug

        supported_candidates = [item for item in scored[1:] if int(item["support"]) >= min_hits]
        if pure_mode and not supported_candidates:
            mentioned_candidates = [item for item in scored[1:] if bool(item.get("mentioned_in_query"))]
            if mentioned_candidates:
                supported_candidates = mentioned_candidates
                debug["dept_gate_reason"] = "pure_query_mention_support"
        if not supported_candidates:
            debug["dept_gate_reason"] = "no_supported_candidate"
            return results, debug

        best_supported = max(
            supported_candidates,
            key=lambda item: (int(item["support"]), float(item["base_score"])),
        )
        score_gap = float(top1["base_score"]) - float(best_supported["base_score"])
        support_delta = int(best_supported["support"]) - int(top1["support"])
        strong_supported_signal = int(best_supported["support"]) >= max(min_hits + 1, 2) and support_delta >= 1
        effective_max_gap = max_gap
        if strong_supported_signal:
            # 对“候选科室关键词信号明显更强”的场景放宽分差，修复 top3 命中但 top1 错配。
            effective_max_gap = max(effective_max_gap, 0.6 if pure_mode else 0.35)
        if pure_mode and support_delta >= 1 and int(best_supported["support"]) >= min_hits:
            # pure 模式下，候选信号只要明显优于 top1，则允许更宽分差以减少误阻断。
            effective_max_gap = max(effective_max_gap, 0.85)
        if pure_mode and int(top1["support"]) == 0 and int(best_supported["support"]) >= 3:
            # pure 模式下，当 top1 无任何症状支撑且候选支撑很强，允许更大分差触发重排。
            effective_max_gap = max(effective_max_gap, 1.0)
        if pure_mode and int(top1["support"]) == 0 and bool(best_supported.get("mentioned_in_query")):
            # 如果候选科室被 query 明确点名，也放宽阈值，降低 score_gap_too_large 误阻断。
            effective_max_gap = max(effective_max_gap, 1.1)
        debug["dept_gate_support_candidate_idx"] = int(best_supported["idx"])
        debug["dept_gate_support_candidate_gap"] = round(score_gap, 6)
        debug["dept_gate_support_hits"] = best_supported["hits"]
        debug["dept_gate_effective_max_gap"] = round(effective_max_gap, 6)

        if score_gap > effective_max_gap:
            debug["dept_gate_reason"] = "score_gap_too_large"
            return results, debug

        reordered = sorted(
            scored,
            key=lambda item: (float(item["adjusted_score"]), float(item["base_score"])),
            reverse=True,
        )
        new_order = [int(item["idx"]) for item in reordered]
        if new_order == list(range(len(results))):
            debug["dept_gate_reason"] = "no_order_change"
            return results, debug

        new_results = [results[i] for i in new_order]
        debug["dept_gate_applied"] = True
        debug["dept_gate_reordered"] = True
        debug["dept_gate_reason"] = "applied"
        return new_results, debug

    def _postprocess_department_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
        pure_mode: bool = False,
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        debug = self._default_dept_post_debug()
        if not isinstance(results, list) or not results:
            debug["dept_normalize_reason"] = "empty_results"
            debug["dept_gate_reason"] = "empty_results"
            return results, debug

        debug["dept_top1_before"] = str(results[0].get("department", "") or "")

        results, norm_debug = self._apply_department_normalization(results)
        debug.update(norm_debug)

        results, gate_debug = self._apply_department_consistency_gate(query, results, pure_mode=pure_mode)
        debug.update(gate_debug)

        debug["dept_top1_after"] = str(results[0].get("department", "") or "") if results else None
        return results, debug

    def _build_cache_namespace(self) -> str:
        """构建缓存命名空间标签（版本化隔离）。"""
        raw = str(getattr(settings, "RAG_CACHE_NAMESPACE_VERSION", "v1") or "v1").strip()
        ns = re.sub(r"[^a-zA-Z0-9_.:-]", "_", raw)[:48]
        return ns or "v1"

    def _build_reject_memory_key(
        self,
        cache_namespace: str,
        query_hash: str,
        semantic_hit_hash: str,
        cache_profile_hash: str,
    ) -> str:
        return f"rag_cache_reject:{cache_namespace}:{query_hash}:{semantic_hit_hash}:{cache_profile_hash}"

    def _evaluate_cache_write_gate(self, final_results: List[Dict[str, Any]]) -> tuple[bool, Dict[str, Any]]:
        """缓存写入门：阻止低质量结果污染语义缓存。"""
        gate_on = bool(getattr(settings, "RAG_CACHE_WRITE_GATE_ON", True))
        min_top = float(getattr(settings, "RAG_CACHE_WRITE_MIN_TOP_SCORE", 0.35))
        min_gap = float(getattr(settings, "RAG_CACHE_WRITE_MIN_SCORE_GAP", 0.03))
        require_known_dept = bool(getattr(settings, "RAG_CACHE_WRITE_REQUIRE_NON_UNKNOWN_DEPT", False))
        debug = {
            "cache_write_gate_enabled": gate_on,
            "cache_write_allowed": False,
            "cache_write_reason": "init",
            "cache_write_min_top_score": min_top,
            "cache_write_min_score_gap": min_gap,
            "cache_write_top_score": 0.0,
            "cache_write_score_gap": 0.0,
        }

        if not final_results:
            debug["cache_write_reason"] = "empty_results"
            return False, debug

        if not gate_on:
            debug["cache_write_allowed"] = True
            debug["cache_write_reason"] = "write_gate_disabled"
            return True, debug

        top_score = float(final_results[0].get("score", 0.0))
        second_score = float(final_results[1].get("score", 0.0)) if len(final_results) > 1 else 0.0
        score_gap = top_score - second_score if len(final_results) > 1 else top_score
        debug["cache_write_top_score"] = top_score
        debug["cache_write_score_gap"] = score_gap

        if top_score < min_top:
            debug["cache_write_reason"] = "top_score_below_threshold"
            return False, debug

        if score_gap < min_gap:
            debug["cache_write_reason"] = "top_score_gap_below_threshold"
            return False, debug

        if require_known_dept:
            dept = str(final_results[0].get("department", "") or "").strip().lower()
            if not dept or dept in {"unknown", "none", "null"}:
                debug["cache_write_reason"] = "unknown_department"
                return False, debug

        debug["cache_write_allowed"] = True
        debug["cache_write_reason"] = "passed"
        return True, debug

    async def _verify_semantic_cache_hit(
        self,
        query: str,
        cached_results: Any,
    ) -> tuple[bool, Dict[str, Any]]:
        """
        语义缓存命中后二次校验。
        失败则应降级为 cache miss，继续走正常召回/重排链路。
        """
        min_term_overlap = max(0, int(getattr(settings, "RAG_CACHE_VERIFY_MIN_TERM_OVERLAP", 1)))
        min_rerank_score = max(0.0, float(getattr(settings, "RAG_CACHE_VERIFY_MIN_RERANK_SCORE", 0.25)))
        max_doc_chars = max(100, int(getattr(settings, "RAG_CACHE_VERIFY_MAX_DOC_CHARS", 600)))

        verify_debug: Dict[str, Any] = {
            "cache_verify_min_term_overlap": min_term_overlap,
            "cache_verify_term_overlap": 0,
            "cache_verify_min_rerank_score": min_rerank_score,
            "cache_verify_rerank_score": None,
            "cache_verify_reason": "init",
        }

        if not isinstance(cached_results, list) or not cached_results:
            verify_debug["cache_verify_reason"] = "empty_cache_payload"
            return False, verify_debug

        top_item = cached_results[0] if isinstance(cached_results[0], dict) else {}
        top_content = str(top_item.get("content", "")) if isinstance(top_item, dict) else str(cached_results[0])
        if not top_content.strip():
            verify_debug["cache_verify_reason"] = "empty_top_content"
            return False, verify_debug

        clipped_content = top_content[:max_doc_chars]
        query_terms = self._build_overlap_terms(query)
        doc_terms = self._build_overlap_terms(clipped_content)
        term_overlap = len(query_terms.intersection(doc_terms))
        verify_debug["cache_verify_term_overlap"] = term_overlap

        if term_overlap < min_term_overlap:
            verify_debug["cache_verify_reason"] = "term_overlap_below_threshold"
            return False, verify_debug

        if self.reranker:
            try:
                rerank_input = [
                    {
                        "id": top_item.get("id"),
                        "score": float(top_item.get("score", 0.0)),
                        "source": top_item.get("source", "cache_verify"),
                        "department": top_item.get("department"),
                        "content": clipped_content,
                    }
                ]
                rerank_results = await asyncio.to_thread(self.reranker.rerank, query, rerank_input)
                rerank_score = float(rerank_results[0].get("score", 0.0)) if rerank_results else 0.0
                verify_debug["cache_verify_rerank_score"] = rerank_score
                if rerank_score < min_rerank_score:
                    verify_debug["cache_verify_reason"] = "rerank_score_below_threshold"
                    return False, verify_debug
            except Exception as rerank_verify_err:
                verify_debug["cache_verify_reason"] = "rerank_verify_error"
                logger.warning(
                    "rag_semantic_cache_verify_rerank_failed",
                    error=str(rerank_verify_err),
                )
                return False, verify_debug

        verify_debug["cache_verify_reason"] = "passed"
        return True, verify_debug

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
    async def search_rag30(
        self,
        query: str,
        top_k: int = 3,
        intent: str = None,
        return_debug: bool = False,
        skip_summarize: bool = False,
        use_rerank: Optional[bool] = None,
        rerank_threshold: Optional[float] = None,
        skip_intent_router: bool = False,
        skip_hyde: bool = False,
        pure_mode: Optional[bool] = None,
    ) -> Any:
        """
        [ASYNC] RAG 3.0 四层检索流水线 (支持 Semantic Cache 2.0)
        """
        try:
            logger.info(
                "rag_search_start",
                query=query[:30],
                top_k=top_k,
                use_rerank=use_rerank,
                rerank_threshold=rerank_threshold,
                skip_intent_router=skip_intent_router,
                skip_hyde=skip_hyde,
                pure_mode=pure_mode,
            )
            pure_retrieval_mode = bool(getattr(settings, "RAG_PURE_RETRIEVAL_MODE", False))
            if pure_mode is True:
                pure_retrieval_mode = True

            if pure_retrieval_mode and bool(getattr(settings, "RAG_DISABLE_INTENT_ROUTER_WHEN_PURE", True)):
                skip_intent_router = True
            if pure_retrieval_mode and bool(getattr(settings, "RAG_DISABLE_HYDE_WHEN_PURE", True)):
                skip_hyde = True
            if pure_retrieval_mode and bool(getattr(settings, "RAG_DISABLE_SUMMARIZE_WHEN_PURE", True)):
                skip_summarize = True

            disable_query_rewrite_when_pure = bool(
                pure_retrieval_mode and getattr(settings, "RAG_DISABLE_QUERY_REWRITE_WHEN_PURE", True)
            )
            disable_low_score_fallback_rewrite_when_pure = bool(
                pure_retrieval_mode and getattr(settings, "RAG_DISABLE_LOW_SCORE_FALLBACK_REWRITE_WHEN_PURE", True)
            )

            start_time = time.time()
            cache_namespace = self._build_cache_namespace()
            cache_verify_debug: Dict[str, Any] = {
                "cache_namespace": cache_namespace,
                "semantic_cache_enabled": bool(getattr(settings, "RAG_SEMANTIC_CACHE_ENABLED", True)),
                "cache_verify_enabled": False,
                "cache_verify_passed": None,
                "cache_verify_reason": "not_checked",
                "cache_verify_min_term_overlap": int(getattr(settings, "RAG_CACHE_VERIFY_MIN_TERM_OVERLAP", 1)),
                "cache_verify_term_overlap": 0,
                "cache_verify_min_rerank_score": float(getattr(settings, "RAG_CACHE_VERIFY_MIN_RERANK_SCORE", 0.25)),
                "cache_verify_rerank_score": None,
                "cache_reject_memory_hit": False,
                "cache_reject_memory_key": None,
            }
            dept_post_debug = self._default_dept_post_debug()

            # [Task 1.4] 语义缓存检查 (Semantic Cache Check)
            cache_key = None
            norm_query = self._normalize_query(query)
            raw_query_hash = hashlib.md5(norm_query.encode()).hexdigest()
            query_hash = hashlib.md5(f"{cache_namespace}|{raw_query_hash}".encode()).hexdigest()
            # Cache profile must include retrieval knobs, otherwise different presets share stale cache.
            cache_profile = json.dumps(
                {
                    "top_k": int(top_k),
                    "use_rerank": use_rerank,
                    "rerank_threshold": rerank_threshold,
                    "skip_intent_router": skip_intent_router,
                    "skip_hyde": skip_hyde,
                    "pure_retrieval_mode": pure_retrieval_mode,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            cache_profile_hash = hashlib.md5(cache_profile.encode()).hexdigest()
            
            # 首先计算 embedding，语义缓存和后续检索都需要它
            emb_start = time.time()
            query_vector = await asyncio.to_thread(self.vector_store.embedding_service.get_embedding, query)
            embedding_latency = time.time() - emb_start
            
            print(f"[DEBUG] RAG Query: {query}")
            print(f"[DEBUG] Embedding Latency: {embedding_latency:.4f}s")

            if self.redis_client:
                # 尝试语义命中
                semantic_enabled = bool(getattr(settings, "RAG_SEMANTIC_CACHE_ENABLED", True))
                if semantic_enabled:
                    try:
                        semantic_hit_hash = await asyncio.wait_for(
                            self.semantic_cache.get_cache(query, query_vector=query_vector),
                            timeout=1.5,
                        )
                    except asyncio.TimeoutError:
                        logger.warning("semantic_cache_lookup_timeout_degrade_to_retrieval", timeout_s=1.5)
                        semantic_hit_hash = None
                    except Exception as semantic_err:
                        logger.warning("semantic_cache_lookup_failed_degrade_to_retrieval", error=str(semantic_err))
                        semantic_hit_hash = None
                else:
                    semantic_hit_hash = None
                    cache_verify_debug["cache_verify_reason"] = "semantic_cache_disabled"

                if semantic_hit_hash and semantic_hit_hash != query_hash:
                    reject_memory_key = self._build_reject_memory_key(
                        cache_namespace=cache_namespace,
                        query_hash=query_hash,
                        semantic_hit_hash=semantic_hit_hash,
                        cache_profile_hash=cache_profile_hash,
                    )
                    cache_verify_debug["cache_reject_memory_key"] = reject_memory_key
                    try:
                        reject_memory_hit = bool(
                            await asyncio.wait_for(self.redis_client.exists(reject_memory_key), timeout=0.5)
                        )
                    except Exception as reject_memory_err:
                        logger.warning("rag_reject_memory_check_failed", error=str(reject_memory_err))
                        reject_memory_hit = False

                    if reject_memory_hit:
                        cache_verify_debug["cache_reject_memory_hit"] = True
                        cache_verify_debug["cache_verify_reason"] = "semantic_reject_memory_blocked"
                        logger.info(
                            "rag_semantic_cache_hit_blocked_by_reject_memory",
                            query=query[:30],
                            reject_key=reject_memory_key,
                        )
                        semantic_hit_hash = None

                # 如果语义命中，使用命中的 hash；否则使用当前的 hash
                effective_hash = semantic_hit_hash if semantic_hit_hash else query_hash
                read_cache_key = f"rag_cache:{cache_namespace}:{effective_hash}:{cache_profile_hash}"

                try:
                    cached_data = await asyncio.wait_for(self.redis_client.get(read_cache_key), timeout=1.0)
                except Exception as cache_read_err:
                    logger.warning("rag_cache_read_failed_degrade_to_retrieval", error=str(cache_read_err))
                    cached_data = None

                if not cached_data and semantic_hit_hash and semantic_hit_hash != query_hash:
                    exact_cache_key = f"rag_cache:{cache_namespace}:{query_hash}:{cache_profile_hash}"
                    try:
                        exact_cached_data = await asyncio.wait_for(
                            self.redis_client.get(exact_cache_key),
                            timeout=0.8,
                        )
                    except Exception as exact_cache_err:
                        logger.warning("rag_exact_cache_fallback_read_failed", error=str(exact_cache_err))
                        exact_cached_data = None
                    if exact_cached_data:
                        cached_data = exact_cached_data
                        cache_verify_debug["cache_verify_reason"] = "semantic_key_miss_exact_fallback_hit"
                        logger.info("rag_exact_cache_fallback_hit", query=query[:30], stage="semantic_key_miss")

                if cached_data:
                    cache_type = "semantic" if semantic_hit_hash else "exact"
                    try:
                        # [Phase 6.5] ZSTD Decompress
                        decompressed = self.semantic_cache.decompress(cached_data)
                        results = json.loads(decompressed)
                    except Exception as cache_decode_err:
                        logger.warning("rag_cache_payload_decode_failed_degrade_to_retrieval", error=str(cache_decode_err))
                        results = []
                        cached_data = None

                    if cached_data and semantic_hit_hash:
                        if getattr(settings, "RAG_CACHE_VERIFY_ON_HIT", True):
                            cache_verify_debug["cache_verify_enabled"] = True
                            verified, verify_fields = await self._verify_semantic_cache_hit(query, results)
                            cache_verify_debug.update(verify_fields)
                            cache_verify_debug["cache_verify_passed"] = bool(verified)
                            if verified:
                                logger.info(
                                    "rag_semantic_cache_hit_verified",
                                    query=query[:30],
                                    reason=cache_verify_debug.get("cache_verify_reason"),
                                    term_overlap=cache_verify_debug.get("cache_verify_term_overlap"),
                                    rerank_score=cache_verify_debug.get("cache_verify_rerank_score"),
                                )
                            else:
                                logger.warning(
                                    "rag_semantic_cache_hit_rejected",
                                    query=query[:30],
                                    reason=cache_verify_debug.get("cache_verify_reason"),
                                    term_overlap=cache_verify_debug.get("cache_verify_term_overlap"),
                                    min_term_overlap=cache_verify_debug.get("cache_verify_min_term_overlap"),
                                    rerank_score=cache_verify_debug.get("cache_verify_rerank_score"),
                                    min_rerank_score=cache_verify_debug.get("cache_verify_min_rerank_score"),
                                )
                                reject_memory_key = self._build_reject_memory_key(
                                    cache_namespace=cache_namespace,
                                    query_hash=query_hash,
                                    semantic_hit_hash=semantic_hit_hash,
                                    cache_profile_hash=cache_profile_hash,
                                )
                                cache_verify_debug["cache_reject_memory_key"] = reject_memory_key
                                try:
                                    reject_reason = str(cache_verify_debug.get("cache_verify_reason", "semantic_rejected"))
                                    reject_ttl = max(60, int(getattr(settings, "RAG_CACHE_REJECT_TTL_SECONDS", 21600)))
                                    await asyncio.wait_for(
                                        self.redis_client.setex(reject_memory_key, reject_ttl, reject_reason),
                                        timeout=0.8,
                                    )
                                except Exception as reject_write_err:
                                    logger.warning("rag_reject_memory_write_failed", error=str(reject_write_err))
                                cached_data = None
                                # 语义命中被拒后，尝试同配置 exact cache 兜底，减少高频 query 性能回退。
                                if semantic_hit_hash != query_hash:
                                    exact_cache_key = f"rag_cache:{cache_namespace}:{query_hash}:{cache_profile_hash}"
                                    try:
                                        exact_cached_data = await asyncio.wait_for(
                                            self.redis_client.get(exact_cache_key),
                                            timeout=0.8,
                                        )
                                    except Exception as exact_cache_err:
                                        logger.warning(
                                            "rag_exact_cache_fallback_read_failed",
                                            error=str(exact_cache_err),
                                        )
                                        exact_cached_data = None

                                    if exact_cached_data:
                                        try:
                                            exact_results = json.loads(self.semantic_cache.decompress(exact_cached_data))
                                            cached_data = exact_cached_data
                                            results = exact_results
                                            cache_type = "exact"
                                            cache_verify_debug["cache_verify_reason"] = "semantic_rejected_exact_fallback_hit"
                                            logger.info("rag_exact_cache_fallback_hit", query=query[:30])
                                        except Exception as exact_decode_err:
                                            logger.warning(
                                                "rag_exact_cache_fallback_decode_failed",
                                                error=str(exact_decode_err),
                                            )
                                            cached_data = None
                        else:
                            cache_verify_debug["cache_verify_reason"] = "verify_disabled"
                    elif cached_data:
                        cache_verify_debug["cache_verify_reason"] = "exact_cache_hit_skip_verify"

                    if cached_data:
                        pure_penalty_debug = self._default_pure_dept_penalty_debug()
                        if pure_retrieval_mode and results:
                            results, pure_penalty_debug = self._apply_pure_department_mismatch_penalty(
                                query,
                                results,
                                execution_path="cache_hit",
                            )
                        elif not pure_retrieval_mode:
                            pure_penalty_debug["pure_dept_penalty_reason"] = "pure_mode_disabled"

                        results, dept_post_debug = self._postprocess_department_results(
                            query,
                            results,
                            pure_mode=pure_retrieval_mode,
                        )
                        dept_post_debug.update(pure_penalty_debug)
                        logger.info("rag_cache_hit", query=query[:30], type=cache_type)

                        # [Metrics] Cache Hit
                        RAG_CACHE_HIT.labels(cache_type=cache_type).inc()

                        if return_debug:
                            return (
                                results,
                                {
                                    "cache_hit": True,
                                    "cache_type": cache_type,
                                    "cache_profile_hash": cache_profile_hash,
                                    **cache_verify_debug,
                                    **dept_post_debug,
                                },
                            )
                        return results

                # 写入一律写到当前 query 的 namespaced hash，避免跨版本语义索引串键。
                cache_key = f"rag_cache:{cache_namespace}:{query_hash}:{cache_profile_hash}"
            
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
                "raw_results": [],
                "cache_hit": False,
                "cache_type": "miss",
                "pure_retrieval_mode": pure_retrieval_mode,
                "pure_disable_query_rewrite": disable_query_rewrite_when_pure,
                "pure_disable_low_score_fallback_rewrite": disable_low_score_fallback_rewrite_when_pure,
                "pure_skip_summarize": skip_summarize,
                **self._default_pure_dept_penalty_debug(),
                **self._default_pure_candidate_comp_debug(),
            }
            metrics.update(cache_verify_debug)
            metrics.update(dept_post_debug)
            
            # L1: 安全拦截 (Guardrail)
            if self.ddinter_checker and not self.ddinter_checker.check_query_safety(query):
                logger.warning("query_blocked_by_guardrail")
                res = [{"content": "您的查询可能涉及高风险用药建议，已被安全护栏拦截。", "score": 1.0, "source": "guardrail"}]
                return (res, metrics) if return_debug else res

            # [Task 1] Intent Analysis (0.6B Router)
            # Allow API callers to bypass this stage for pure retrieval latency.
            if not skip_intent_router:
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
            else:
                metrics["intent_latency"] = 0.0

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
            if not skip_hyde and len(query) < 20: # Only apply HyDE for short/ambiguous queries
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
            # Tiered Reranking L1: Broad Recall
            recall_window = max(int(getattr(settings, "RAG_RECALL_WINDOW", 100)), int(top_k))
            
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
            
            # RRF Fusion -> Tiered Reranking L2: Coarse Rank
            # Cap rerank candidates to reduce cold-query latency on 8GB GPUs.
            rerank_candidate_k = max(int(top_k), int(getattr(settings, "RAG_RERANK_CANDIDATE_K", 12)))
            rerank_candidate_k = min(rerank_candidate_k, recall_window)
            fusion_pool_k = recall_window if pure_retrieval_mode else rerank_candidate_k
            fused_pool = self._rrf_fusion(v_results, b_results, top_k=fusion_pool_k, target_dept=target_dept)
            pure_candidate_comp_debug = self._default_pure_candidate_comp_debug()
            if pure_retrieval_mode and fused_pool:
                results, pure_candidate_comp_debug = self._select_pure_rerank_candidates(
                    query=query,
                    fused_results=fused_pool,
                    rerank_candidate_k=rerank_candidate_k,
                )
            else:
                results = fused_pool[:rerank_candidate_k]
            metrics["retrieval_latency"] = time.time() - retrieval_start
            metrics["recall_window"] = recall_window
            metrics["rerank_candidate_k"] = rerank_candidate_k
            metrics.update(pure_candidate_comp_debug)
            
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
            if self.reranker and len(results) > 0 and use_rerank is not False:
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

                if pure_retrieval_mode and results:
                    results, pure_penalty_debug = self._apply_pure_department_mismatch_penalty(
                        query,
                        results,
                        execution_path="retrieval",
                    )
                    metrics.update(pure_penalty_debug)
                
                if results:
                    metrics["rerank_score"] = results[0].get("score", 0.0)

                # [Task 2.1] 动态阈值与阻断策略 (Dynamic Thresholding)
                BASE_REL_THRESHOLD = 0.15  # [Optimization] 进一步降低阈值以减少不必要的重试延迟
                if rerank_threshold is not None:
                    try:
                        BASE_REL_THRESHOLD = max(0.0, min(1.0, float(rerank_threshold)))
                    except (TypeError, ValueError):
                        logger.warning("invalid_rerank_threshold_fallback_default", rerank_threshold=rerank_threshold)
                effective_rel_threshold = BASE_REL_THRESHOLD
                if pure_retrieval_mode:
                    pure_factor = float(getattr(settings, "RAG_PURE_RERANK_THRESHOLD_FACTOR", 0.75))
                    pure_factor = max(0.0, min(1.0, pure_factor))
                    effective_rel_threshold = BASE_REL_THRESHOLD * pure_factor
                    metrics["pure_rerank_threshold_factor"] = pure_factor
                    metrics["pure_effective_rerank_threshold"] = effective_rel_threshold
                GAP_THRESHOLD = 0.05
                
                if not results or results[0]['score'] < effective_rel_threshold:
                    logger.warning("rerank_blocked_by_threshold_triggering_fallback", top_score=results[0]['score'] if results else 0)
                    
                    # [Task 2.2] 失败回退机制: Query Rewriting
                    # 仅在非调试模式且第一次检索失败时尝试一次重写
                    if not intent == "fallback_retry" and not disable_low_score_fallback_rewrite_when_pure:
                        rewritten_query = await self._rewrite_query_async(query)
                        if rewritten_query and rewritten_query != query:
                            logger.info("rag_fallback_retry_start", rewritten_query=rewritten_query)
                            return await self.search_rag30(
                                rewritten_query,
                                top_k=top_k,
                                intent="fallback_retry",
                                return_debug=return_debug,
                                skip_summarize=skip_summarize,
                                use_rerank=use_rerank,
                                rerank_threshold=rerank_threshold,
                                skip_intent_router=skip_intent_router,
                                skip_hyde=skip_hyde,
                                pure_mode=pure_retrieval_mode,
                            )
                    elif disable_low_score_fallback_rewrite_when_pure:
                        logger.info("rag_fallback_retry_skipped_pure_mode", query=query[:60])
                    
                    return ([], metrics) if return_debug else []

                # 动态确定保留数量：如果后续结果与前一个分差很小，则保留
                final_count = 1
                for j in range(1, min(len(results), MAX_FINAL_K)):
                    if results[j]['score'] >= effective_rel_threshold and \
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

            final_results, dept_post_debug = self._postprocess_department_results(
                query,
                final_results,
                pure_mode=pure_retrieval_mode,
            )
            metrics.update(dept_post_debug)

            duration = time.time() - start_time
            logger.info("rag_search_end", duration=f"{duration:.2f}s", result_count=len(final_results))
            
            # [Task 2] LangSmith Metadata Logging
            run = get_current_run_tree()
            if run:
                run.metadata.update(metrics)
            
            # 性能监控上报 (修正参数匹配: latency_ms, success)
            # track_rag_query(duration * 1000, True)
            
            # [Task 1.4] 写入缓存 (Cache Write) with ZSTD
            cache_write_allowed, cache_write_debug = self._evaluate_cache_write_gate(final_results)
            metrics.update(cache_write_debug)
            if self.redis_client and cache_key and final_results and cache_write_allowed:
                try:
                    # [Phase 6.5] ZSTD Compress
                    compressed = self.semantic_cache.compress(json.dumps(final_results, ensure_ascii=False))
                    cache_ttl = max(60, int(getattr(settings, "RAG_CACHE_TTL_SECONDS", 86400)))
                    await asyncio.wait_for(self.redis_client.setex(cache_key, cache_ttl, compressed), timeout=1.0)
                    semantic_enabled = bool(getattr(settings, "RAG_SEMANTIC_CACHE_ENABLED", True))
                    if semantic_enabled:
                        # 更新语义缓存索引 (Milvus)
                        await asyncio.wait_for(
                            asyncio.to_thread(self.semantic_cache.update_cache, query, query_hash, query_vector=query_vector),
                            timeout=1.5,
                        )
                    logger.info("rag_cache_write_success", key=cache_key)
                except Exception as cache_err:
                    logger.error("rag_cache_write_failed", error=str(cache_err))
            elif self.redis_client and cache_key and final_results:
                logger.info(
                    "rag_cache_write_skipped_by_gate",
                    reason=cache_write_debug.get("cache_write_reason"),
                    top_score=cache_write_debug.get("cache_write_top_score"),
                    score_gap=cache_write_debug.get("cache_write_score_gap"),
                )

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
        if bool(getattr(settings, "RAG_PURE_RETRIEVAL_MODE", False)) and bool(
            getattr(settings, "RAG_DISABLE_QUERY_REWRITE_WHEN_PURE", True)
        ):
            return query
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
