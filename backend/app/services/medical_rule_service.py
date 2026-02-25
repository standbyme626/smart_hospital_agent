import asyncio
import structlog
from typing import List, Dict, Optional, Set
from sqlalchemy import select, or_, and_
from sqlalchemy.ext.asyncio import AsyncSession
from functools import lru_cache
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.db.session import AsyncSessionLocal
from app.db.models.medical_rules import DrugTranslation, DrugInteraction, SafetyGuardrail

logger = structlog.get_logger(__name__)

class MedicalRuleService:
    """
    医疗规则服务 (Medical Rule Service)
    负责从数据库加载和提供医疗业务规则，并提供内存缓存。
    """
    _instance = None
    
    # 内存缓存 (Memory Cache)
    _translation_cache: Dict[str, str] = {}
    _interaction_cache: List[Dict] = [] # List of {drugs: Set[str], desc: str, sev: str}
    _guardrail_cache: Set[str] = set()
    _common_drugs_cache: List[str] = []
    
    _cache_initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MedicalRuleService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        # 注意：__init__ 可能会被多次调用，但我们只希望初始化一次
        pass
        
    async def initialize(self):
        """Async initialization to load rules from DB"""
        if not self._cache_initialized:
            await self.refresh_rules()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry_error_callback=lambda retry_state: logger.error("Retry exhausted", last_exception=retry_state.outcome.exception())
    )
    async def refresh_rules(self):
        """从数据库全量重新加载规则到内存缓存"""
        logger.info("Refreshing medical rules from database...")
        try:
            async with AsyncSessionLocal() as session:
                # 1. Load Translations
                stmt_trans = select(DrugTranslation)
                result_trans = await session.execute(stmt_trans)
                translations = result_trans.scalars().all()
                
                new_trans_cache = {}
                common_drugs = []
                for t in translations:
                    new_trans_cache[t.cn_name] = t.en_name
                    # 假设目前所有翻译的 key 都是常见药物候选，或者我们可以加一个字段。
                    # 为了兼容之前的逻辑，这里简单地收集所有 cn_name
                    common_drugs.append(t.cn_name)
                
                self._translation_cache = new_trans_cache
                self._common_drugs_cache = common_drugs # 可以根据 category 过滤

                # 2. Load Interactions
                stmt_inter = select(DrugInteraction)
                result_inter = await session.execute(stmt_inter)
                interactions = result_inter.scalars().all()
                
                new_inter_cache = []
                for i in interactions:
                    new_inter_cache.append({
                        "drugs": {i.drug_a, i.drug_b},
                        "description": i.description,
                        "severity": i.severity
                    })
                self._interaction_cache = new_inter_cache

                # 3. Load Guardrails
                stmt_guard = select(SafetyGuardrail)
                result_guard = await session.execute(stmt_guard)
                guardrails = result_guard.scalars().all()
                
                self._guardrail_cache = {g.keyword for g in guardrails if g.action_type == "block"}
                
                self._cache_initialized = True
                logger.info("Medical rules refreshed", 
                            translations=len(self._translation_cache),
                            interactions=len(self._interaction_cache),
                            guardrails=len(self._guardrail_cache))
                            
        except Exception as e:
            logger.error("Failed to refresh medical rules", error=str(e))
            # Don't clear cache if DB fails, keep old cache

    def get_translation(self, cn_name: str) -> str:
        """Get English translation for a drug name (Cached)"""
        if not self._cache_initialized:
            logger.warning("MedicalRuleService accessed before initialization!")
            # Fallback or empty return? 
            # In sync context we can't await. Ideally app startup calls initialize()
        
        # 1. Exact match from cache
        if cn_name in self._translation_cache:
            return self._translation_cache[cn_name]
        
        # 2. Partial match (scan keys) - O(N) but N is small (<1000 rules usually)
        # If performance matters, use a Trie or separate mapping.
        for k, v in self._translation_cache.items():
            if k in cn_name:
                return v
        
        return cn_name

    def check_interaction(self, drug_a_en: str, drug_b_en: str) -> Optional[str]:
        """Check for high-risk interactions (Cached)"""
        pair = {drug_a_en, drug_b_en}
        
        for combo in self._interaction_cache:
            if pair == combo["drugs"]:
                return f"{combo['description']} (Severity: {combo['severity']})"
        
        return None

    def check_query_safety(self, query: str) -> bool:
        """Check if query contains forbidden keywords (Cached)"""
        for kw in self._guardrail_cache:
            if kw in query:
                return False
        return True

    def get_common_drugs(self) -> List[str]:
        return self._common_drugs_cache

    @property
    def translation_map(self) -> Dict[str, str]:
        """
        Legacy support for translation map access.
        Returns a copy of the translation cache.
        """
        return self._translation_cache.copy()

# Global instance
medical_rule_service = MedicalRuleService()
