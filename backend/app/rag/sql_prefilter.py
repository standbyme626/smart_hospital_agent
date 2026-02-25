import re
import structlog
from typing import List, Optional
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
        
        # 1. 提取科室 (Department) - 增强映射逻辑
        departments = DEPARTMENT_LIST
        
        # 语义关键词映射 (补齐隐含科室的词汇)
        keyword_map = SYMPTOM_KEYWORD_MAP

        # 1. 优先匹配明确科室名
        for dept in departments:
            if dept in query:
                filters["department"] = dept
                break
        
        # 2. 其次匹配语义关键词 (采用最长匹配优先策略)
        if "department" not in filters:
            matches = []
            for kw, dept in keyword_map.items():
                if kw in query:
                    matches.append((kw, dept))
            
            if matches:
                # 按关键词长度降序排列，取最长的一个
                matches.sort(key=lambda x: len(x[0]), reverse=True)
                filters["department"] = matches[0][1]

                
        # 2. 提取文档类型 (Type)
        if "指南" in query:
            filters["doc_type"] = "guideline"
        elif "药品" in query or "说明书" in query:
            filters["doc_type"] = "drug_manual"
        elif "病历" in query:
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
