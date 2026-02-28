import re
import structlog
from typing import List, Optional
from collections import defaultdict
from app.db.session import AsyncSessionLocal
from sqlalchemy import text
import asyncio
from app.core.constants import DEPARTMENT_LIST, SYMPTOM_KEYWORD_MAP

logger = structlog.get_logger(__name__)

class SQLPreFilter:
    """
    RAG Level 1: SQL Pre-Filter
    从自然语言查询中提取结构化过滤条件 (Metadata Filters)
    """
    def __init__(self):
        pass

    def extract_filters(self, query: str) -> dict:
        """
        提取过滤条件，返回 Milvus/SQL 兼容的 filter expression
        """
        filters = {}
        norm_query = str(query or "").strip()
        if not norm_query:
            return filters
        
        # 1. 提取科室 (Department) - 增强映射逻辑
        departments = sorted(DEPARTMENT_LIST, key=len, reverse=True)
        
        # 语义关键词映射 (补齐隐含科室的词汇)
        keyword_map = SYMPTOM_KEYWORD_MAP

        # 1. 优先匹配明确科室名
        for dept in departments:
            if dept in norm_query:
                filters["department"] = dept
                break
        
        # 2. 其次匹配语义关键词（仅在高置信度下启用）
        # 目标：降低单个通用词触发误过滤，避免 pure 检索被错误科室缩窄召回范围。
        if "department" not in filters:
            dept_scores = defaultdict(float)
            dept_hit_keywords = defaultdict(set)

            ambiguous_keywords = {
                "感冒", "发烧", "发热", "咳嗽", "肺炎", "头晕", "鼻涕",
            }

            for kw, dept in keyword_map.items():
                kw = str(kw or "").strip()
                if not kw or kw not in norm_query:
                    continue
                # 单字关键词误触发率高，禁用其科室预过滤用途
                if len(kw) <= 1:
                    continue

                # 关键词加权：越长越可信；常见泛词降权
                weight = 1.0 if len(kw) == 2 else 2.0
                if kw in ambiguous_keywords:
                    weight *= 0.5

                dept_scores[dept] += weight
                dept_hit_keywords[dept].add(kw)

            if dept_scores:
                ranked = sorted(
                    dept_scores.items(),
                    key=lambda item: (
                        item[1],
                        len(dept_hit_keywords[item[0]]),
                        max((len(k) for k in dept_hit_keywords[item[0]]), default=0),
                    ),
                    reverse=True,
                )
                best_dept, best_score = ranked[0]
                second_score = ranked[1][1] if len(ranked) > 1 else 0.0
                best_hit_count = len(dept_hit_keywords[best_dept])
                best_max_kw_len = max((len(k) for k in dept_hit_keywords[best_dept]), default=0)

                # 高置信度启用规则：
                # - 分数足够高，且领先次优；或
                # - 至少两个独立关键词共同指向同一科室。
                high_confidence = (
                    (best_score >= 2.0 and (best_score - second_score) >= 0.5)
                    or (best_hit_count >= 2 and best_max_kw_len >= 2)
                )
                if high_confidence:
                    filters["department"] = best_dept

                
        # 2. 提取文档类型 (Type)
        if "指南" in norm_query:
            filters["doc_type"] = "guideline"
        elif "药品" in norm_query or "说明书" in norm_query:
            filters["doc_type"] = "drug_manual"
        elif "病历" in norm_query:
            filters["doc_type"] = "medical_record"
            
        return filters

    def generate_milvus_expr(self, filters: dict) -> str:
        """
        将字典转换为 Milvus 表达式字符串
        """
        if not filters:
            return ""
            
        exprs = []
        for k, v in filters.items():
            if k == "department":
                # 使用 LIKE 进行模糊匹配，支持“内科”匹配“呼吸内科”
                exprs.append(f"{k} like '%{v}%'")
            else:
                exprs.append(f"{k} == '{v}'")
            
        return " and ".join(exprs)

    def generate_filter_sql(self, filters: dict) -> str:
        """
        将字典转换为 PostgreSQL SQL 过滤子句
        """
        if not filters:
            return "1=1"
            
        conditions = []
        for k, v in filters.items():
            if k == "department":
                # 使用 LIKE 进行模糊匹配，支持“内科”匹配“呼吸内科”
                conditions.append(f"metadata->>'{k}' LIKE '%{v}%'")
            else:
                conditions.append(f"metadata->>'{k}' = '{v}'")
            
        return " AND ".join(conditions)

    def to_milvus_expr(self, filters: dict) -> str:
        # 保持向后兼容
        return self.generate_milvus_expr(filters)

    @staticmethod
    def _safe_int(value) -> Optional[int]:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    async def _query_department_ids_async(self, department: str) -> List[int]:
        """
        兼容两种 SQL Schema:
        1) medical_chunks(milvus_id, metadata jsonb)
        2) medical_chunks(id, department text)
        """
        candidates = [
            (
                text("SELECT milvus_id FROM medical_chunks WHERE (metadata->>'department' LIKE :dept) AND milvus_id IS NOT NULL"),
                lambda row: self._safe_int(row[0]),
            ),
            (
                text("SELECT id FROM medical_chunks WHERE department LIKE :dept"),
                lambda row: self._safe_int(row[0]),
            ),
            (
                text("SELECT id FROM medical_chunks WHERE (metadata->>'department' LIKE :dept)"),
                lambda row: self._safe_int(row[0]),
            ),
        ]
        async with AsyncSessionLocal() as session:
            for stmt, mapper in candidates:
                try:
                    result = await session.execute(stmt, {"dept": f"%{department}%"})
                    rows = result.all()
                    ids = [mapper(row) for row in rows]
                    ids = [i for i in ids if i is not None]
                    if ids:
                        return ids
                except Exception as e:
                    logger.warning("sql_prefilter_query_variant_failed", error=str(e), statement=str(stmt))
        return []

    async def get_ids_by_department_async(self, department: str) -> List[int]:
        """
        [ASYNC] 根据科室名称从 PostgreSQL 获取对应的 milvus_id 列表
        使用 AsyncSessionLocal 以复用连接池，减少 I/O 耗时
        """
        ids = []
        try:
            ids = await self._query_department_ids_async(department)
            logger.info("sql_prefilter_dept_match_async", department=department, count=len(ids))
        except Exception as e:
            logger.error("sql_prefilter_async_failed", error=str(e))
        
        return ids

    def get_ids_by_department(self, department: str) -> List[int]:
        """
        根据科室名称从 PostgreSQL 获取对应的 milvus_id 列表
        (Smart Pre-filtering Implementation)
        """
        import psycopg2
        from app.core.config import settings
        
        ids = []
        try:
            db_url = settings.DATABASE_URL.replace('+asyncpg', '')
            conn = psycopg2.connect(db_url)
            cur = conn.cursor()

            pattern = f"%{department}%"
            query_variants = [
                ("SELECT milvus_id FROM medical_chunks WHERE (metadata->>'department' LIKE %s) AND milvus_id IS NOT NULL",),
                ("SELECT id FROM medical_chunks WHERE department LIKE %s",),
                ("SELECT id FROM medical_chunks WHERE (metadata->>'department' LIKE %s)",),
            ]
            for query in query_variants:
                try:
                    cur.execute(query[0], (pattern,))
                    rows = cur.fetchall()
                    ids = [self._safe_int(row[0]) for row in rows]
                    ids = [i for i in ids if i is not None]
                    if ids:
                        break
                except Exception as q_err:
                    logger.warning("sql_prefilter_query_variant_failed_sync", error=str(q_err), query=query[0])
            
            cur.close()
            conn.close()
            
            logger.info("sql_prefilter_dept_match", department=department, count=len(ids))
        except Exception as e:
            logger.error("sql_prefilter_failed", error=str(e))
        
        return ids
