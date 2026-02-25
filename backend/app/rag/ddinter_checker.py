import structlog
from typing import List, Dict, Optional
import asyncio
from sqlalchemy import text, select, or_, and_
from app.db.session import AsyncSessionLocal
from app.db.models.ddinter import DDInterInteraction

from app.services.medical_rule_service import medical_rule_service

logger = structlog.get_logger(__name__)

class DDInterChecker:
    """
    药物相互作用检查器 (Drug-Drug Interaction Checker)
    基于 MedicalRuleService 和 DDInter 数据库规则进行安全拦截
    """
    _db_available = True # Circuit Breaker

    def __init__(self):
        # 委托给 MedicalRuleService，不再自行加载
        pass

    def _extract_drugs(self, query: str) -> List[str]:
        """从 Query 中提取潜在药物名 (简化版 NER)"""
        # 实际应使用 medical-ner 模型，这里使用 translation_map 的 key 进行匹配
        found_drugs = []
        translation_map = medical_rule_service.translation_map
        
        for cn_name in translation_map.keys():
            if len(cn_name) > 1 and cn_name in query:
                found_drugs.append(cn_name)
        
        # 补充常见药物
        common_drugs = medical_rule_service.get_common_drugs()
        for d in common_drugs:
            if d not in found_drugs and d in query:
                found_drugs.append(d)
                
        return list(set(found_drugs))

    def _translate_to_en(self, drug_name: str) -> str:
        """中文转英文 (DDInter 使用英文)"""
        return medical_rule_service.get_translation(drug_name)

    async def check_interaction_in_db_async(self, drug_a_en: str, drug_b_en: str) -> Optional[str]:
        """[ASYNC] 查询 SQL 数据库检查相互作用"""
        # 1. Check MedicalRuleService (Fast Path / Hardcoded Rules)
        risk_msg = medical_rule_service.check_interaction(drug_a_en, drug_b_en)
        if risk_msg:
            return risk_msg
        
        # Circuit Breaker Check
        if not self._db_available:
            return None

        try:
            # Use AsyncSessionLocal for non-blocking DB access
            async with AsyncSessionLocal() as session:
                # ORM Query
                stmt = select(DDInterInteraction).where(
                    or_(
                        and_(DDInterInteraction.drug_a == drug_a_en, DDInterInteraction.drug_b == drug_b_en),
                        and_(DDInterInteraction.drug_a == drug_b_en, DDInterInteraction.drug_b == drug_a_en)
                    )
                ).limit(1)
                
                result = await session.execute(stmt)
                interaction = result.scalar_one_or_none()
                
                if interaction:
                    return f"Recorded Interaction: {interaction.description} (Severity: {interaction.severity})"
                    
        except Exception as e:
            logger.warning("ddinter_db_async_check_failed", error=str(e))
            pass
            
        return None

    def check_interaction_in_db(self, drug_a_en: str, drug_b_en: str) -> Optional[str]:
        """[LEGACY] 查询 SQL 数据库检查相互作用"""
        # 1. Check MedicalRuleService (Fast Path / Hardcoded Rules)
        risk_msg = medical_rule_service.check_interaction(drug_a_en, drug_b_en)
        if risk_msg:
            return risk_msg
        
        # Legacy sync implementation (removed psycopg2 to avoid dependency)
        return None

    def check(self, drugs: list[str]) -> list[str]:

        """
        [SYNC] 检查药物列表是否存在已知的相互作用
        """
        # Sync version only checks hardcoded rules now to avoid blocking
        warnings = []
        n = len(drugs)
        if n < 2:
            return warnings
            
        for i in range(n):
            for j in range(i + 1, n):
                d1_cn = drugs[i]
                d2_cn = drugs[j]
                d1_en = self._translate_to_en(d1_cn)
                d2_en = self._translate_to_en(d2_cn)
                risk_msg = self.check_interaction_in_db(d1_en, d2_en)
                if risk_msg:
                    warnings.append(f"⚠️ 发现高危药物相互作用: [{d1_cn} + {d2_cn}] -> {risk_msg}")
        return list(set(warnings))

    async def check_async(self, drugs: list[str]) -> list[str]:
        """
        [ASYNC] 检查药物列表是否存在已知的相互作用
        """
        warnings = []
        n = len(drugs)
        if n < 2:
            return warnings
            
        # 2. 两两检查
        tasks = []
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                d1_cn = drugs[i]
                d2_cn = drugs[j]
                d1_en = self._translate_to_en(d1_cn)
                d2_en = self._translate_to_en(d2_cn)
                pairs.append((d1_cn, d2_cn))
                tasks.append(self.check_interaction_in_db_async(d1_en, d2_en))
        
        if not tasks:
            return []
            
        results = await asyncio.gather(*tasks)
        
        for idx, risk_msg in enumerate(results):
            if risk_msg:
                d1, d2 = pairs[idx]
                warnings.append(f"⚠️ 发现高危药物相互作用: [{d1} + {d2}] -> {risk_msg}")
                        
        return list(set(warnings))

    def check_query_safety(self, query: str) -> bool:
        """
        检查用户查询是否包含危险操作 (Level 4 Guardrail)
        """
        # 1. 拦截高危意图 (Delegate to Service)
        return medical_rule_service.check_query_safety(query)
        
    def scan_query_for_warnings(self, query: str) -> List[str]:
        """[SYNC] 扫描 Query 中的 DDI 风险"""
        drugs = self._extract_drugs(query)
        if len(drugs) >= 2:
            return self.check(drugs)
        return []

    async def scan_query_for_warnings_async(self, query: str) -> List[str]:
        """[ASYNC] 扫描 Query 中的 DDI 风险"""
        drugs = self._extract_drugs(query)
        if len(drugs) >= 2:
            return await self.check_async(drugs)
        return []
