from crewai.tools import BaseTool
from pydantic import PrivateAttr, Field
from app.rag.retriever import MedicalRetriever
from app.core.tools.medical_tools import get_retriever
from langsmith import traceable
import requests # For external drug interaction API

class SearchMedicalDB(BaseTool):
    """
    医学数据库搜索工具 (Medical Database Search Tool)
    
    用于搜索内部医学知识库，获取临床指南、药品信息和疾病诊疗方案。
    适用于辅助诊断、验证治疗方案或查询禁忌症。
    """
    name: str = "SearchMedicalDB"
    description: str = (
        "Search the internal medical knowledge base for clinical guidelines, "
        "drug information, and disease protocols. "
        "Useful for validating diagnoses and checking treatments. "
        "Input should be a specific medical query string."
    )
    # 使用 PrivateAttr 避免 Pydantic 验证和序列化问题
    _retriever: MedicalRetriever = PrivateAttr(default=None)
    
    def __init__(self, **kwargs):
        """
        初始化搜索工具 (Initialize Search Tool)
        如果未传入 retriever 实例，则自动初始化一个新的 MedicalRetriever。
        """
        super().__init__(**kwargs)
        # 延迟初始化检索器，或使用传入的实例
        # 这里如果未提供，则初始化它以确保可用
        if self._retriever is None:
            self._retriever = get_retriever()

    @traceable(name="SearchMedicalDB")
    def _run(self, query: str) -> str:
        """
        执行搜索操作 (Execute Search)
        
        Args:
            query: 具体的医学查询字符串 (e.g. "高血压的首选药物")
            
        Returns:
            str: 格式化后的检索结果，包含内容和来源
        """
        try:
            # 调用底层 RAG 检索器获取 Top-3 结果 (使用同步接口)
            # [Fix V5.4] Use search_sync to avoid async coroutine issues in CrewAI
            results = self._retriever.search_sync(query, top_k=3)
            return self._format_results(results)
        except Exception as e:
            return f"Error searching medical DB: {str(e)}"

    async def _arun(self, query: str) -> str:
        """
        [ASYNC] 执行搜索操作 (Execute Search Async)
        
        CrewAI 会优先尝试异步执行工具。
        """
        try:
            # 直接调用异步检索方法
            results = await self._retriever.search_rag30(query, top_k=3)
            return self._format_results(results)
        except Exception as e:
            return f"Error searching medical DB (async): {str(e)}"

    def _format_results(self, results) -> str:
        """格式化检索结果"""
        formatted_results = ""
        if not results:
             return "No relevant information found in medical database."
             
        for idx, res in enumerate(results):
            # 处理可能的元组返回 (results, metrics)
            actual_res = res if isinstance(res, dict) else res[0] if isinstance(res, (list, tuple)) else {}
            content = actual_res.get('content', str(actual_res))
            source = actual_res.get('source', 'Unknown')
            formatted_results += f"Result {idx+1}:\n{content}\n(Source: {source})\n\n"
        return formatted_results

from app.rag.ddinter_checker import DDInterChecker

class DrugInteractionCheck(BaseTool):
    """
    药物相互作用检查工具 (Drug Interaction Checker)
    
    专门用于查询药物之间的相互作用。基于 DDInter 数据库。
    """
    name: str = "DrugInteractionCheck"
    description: str = (
        "Check for drug-drug interactions using DDInter database. "
        "Input should be a comma-separated list of drug names (e.g., 'Aspirin, Warfarin')."
    )
    # 使用 PrivateAttr 避免序列化问题
    _checker: DDInterChecker = PrivateAttr(default=None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self._checker is None:
            self._checker = DDInterChecker()

    def _run(self, drugs: str) -> str:
        """
        [SYNC] 执行药物检查
        """
        try:
            # 清理输入
            drug_list = [d.strip() for d in drugs.split(',')]
            if len(drug_list) < 2:
                return "Need at least 2 drugs to check interactions."
            
            # 调用真实的 DDInterChecker
            warnings = self._checker.check(drug_list)
            
            if not warnings:
                return "✅ No known interactions found in DDInter database."
            
            return "⚠️ INTERACTIONS FOUND:\n" + "\n".join(warnings)
        except Exception as e:
            return f"Error checking interactions: {str(e)}"

    async def _arun(self, drugs: str) -> str:
        """
        [ASYNC] 执行药物检查 (Async)
        """
        try:
            drug_list = [d.strip() for d in drugs.split(',')]
            if len(drug_list) < 2:
                return "Need at least 2 drugs to check interactions."
                
            # 调用异步方法
            warnings = await self._checker.check_async(drug_list)
            
            if not warnings:
                return "✅ No known interactions found in DDInter database."
                
            return "⚠️ INTERACTIONS FOUND:\n" + "\n".join(warnings)
        except Exception as e:
             return f"Error checking interactions (async): {str(e)}"
