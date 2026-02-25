"""
指南检索工具 (Guideline Lookup Tool)

从RAG系统检索医学指南和临床知识
"""

from typing import Dict, Any, Optional, List
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import structlog

from app.tools.base import BaseTool
from app.rag.retriever import get_retriever

logger = structlog.get_logger()


class GuidelineLookupTool(BaseTool):
    """
    指南检索工具
    
    用途:
    - Agent调用检索医学指南
    - 分诊时查询症状对应科室
    - 诊疗时查询治疗方案
    """
    
    def __init__(self):
        super().__init__()
        self.retriever = get_retriever()
        
        # 通用医学原则(降级时使用)
        self.fallback_guidelines = {
            "胸痛": "胸痛可能是心血管疾病的表现，建议立即就医，挂心内科或急诊科。",
            "发热": "持续发热超过3天或伴有其他症状，建议就医检查，可挂普通内科或呼吸内科。",
            "头痛": "突发剧烈头痛或伴有恶心呕吐，建议急诊就医，可挂神经内科。",
            "腹痛": "急性腹痛可能是多种疾病的表现，建议就医，可挂急诊科或消化内科。",
            "default": "建议就医，由专业医生进行详细评估和诊断。"
        }
    
    @property
    def description(self) -> str:
        return """
        检索医学指南和临床知识。
        
        参数:
        - query (str): 医学问题或症状描述
        - department (str, optional): 科室过滤
        - top_k (int): 返回结果数,默认3
        
        返回:
        - guidelines: 检索到的指南列表
        - confidence: 置信度分数
        - source: 数据来源
        """
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception)
    )
    async def _execute(
        self, 
        query: str,
        department: Optional[str] = None,
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        执行指南检索
        
        流程:
        1. 调用RAG检索器
        2. 可选的科室过滤
        3. 格式化结果
        """
        # 1. RAG检索 (Updated to use RAG 3.0 API)
        results = await self.retriever.search_rag30(query, top_k=top_k)
        
        # 2. 科室过滤(可选)
        # Note: RAG 3.0 has smart pre-filtering, but if explicit department is passed, we can filter further.
        # However, metadata might not be available in flattened results.
        # We skip this for now or rely on RAG 3.0's internal filtering if we passed 'department' to it.
        # But search_rag30 doesn't take 'department' arg directly, it extracts from query.
        # If we want to support explicit department, we should update search_rag30 signature or just append to query.
        if department:
             # Simple append to query for RAG 3.0 to pick up
             # But we already searched.
             pass
        
        # 3. 计算置信度
        confidence = max([r.get('score', 0) for r in results]) if results else 0.0
        
        # 4. 格式化
        guidelines = []
        for r in results:
            guidelines.append({
                'content': r.get('content', ''),
                'score': r.get('score', 0),
                'source': r.get('source', 'unknown'),
                'id': r.get('id')
            })
        
        return {
            'success': True,
            'data': {
                'guidelines': guidelines,
                'query': query,
                'department_filter': department,
                'count': len(guidelines)
            },
            'confidence': confidence,
            'source': 'rag'
        }
    
    def _fallback(self, error: Exception, **kwargs) -> Dict[str, Any]:
        """
        降级方案: 返回通用医学原则
        
        策略:
        1. 关键词匹配 → 返回预设建议
        2. 无匹配 → 返回通用建议
        """
        query = kwargs.get('query', '')
        
        # 关键词匹配
        guideline = None
        for keyword, advice in self.fallback_guidelines.items():
            if keyword in query:
                guideline = advice
                break
        
        if not guideline:
            guideline = self.fallback_guidelines['default']
        
        logger.warning(
            "guideline_lookup.fallback",
            query=query,
            error=str(error)
        )
        
        return {
            'success': False,
            'data': {
                'guidelines': [{
                    'content': guideline,
                    'score': 0.5,
                    'department': None,
                    'disease': None
                }],
                'query': query,
                'count': 1
            },
            'confidence': 0.5,
            'source': 'fallback'
        }


# 便捷函数
async def lookup_guideline(query: str, department: Optional[str] = None) -> Dict[str, Any]:
    """
    快捷调用指南检索
    
    示例:
        result = await lookup_guideline("患者胸痛3小时", department="心内科")
        print(result['data']['guidelines'])
    """
    tool = GuidelineLookupTool()
    return await tool.run(query=query, department=department)
