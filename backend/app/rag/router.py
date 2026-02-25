from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.core.config import settings
from app.core.models.local_slm import local_slm
from app.core.llm.llm_factory import SmartRotatingLLM
from langsmith import traceable
import structlog

logger = structlog.get_logger(__name__)

class MedicalRouter:
    """
    RAG Router: 将用户查询路由到不同的处理分支
    - retrieval: 需要检索医学知识
    - direct: 直接回答 (闲聊/简单问题)
    - analysis: 需要复杂分析 (目前归为 retrieval)
    """
    def __init__(self):
        # [V6.5.2] Upgrade to SmartRotatingLLM
        self.llm = SmartRotatingLLM(
            model_name=settings.LLM_MODEL,
            prefer_local=False
        )
        self.prompt = ChatPromptTemplate.from_template(
            """你是一个医疗意图分类器。请判断用户输入的意图。
            
            可选类别:
            - retrieval: 用户询问医学知识、疾病症状、药品信息、治疗方案等 (例如: "感冒吃什么药", "糖尿病的症状").
            - direct: 用户进行打招呼、感谢、或者询问你是谁 (例如: "你好", "谢谢", "你是谁").
            - crisis: 用户表达自残、自杀倾向 (例如: "我想死").
            
            用户输入: {query}
            
            仅输出类别名称，不要输出其他内容。"""
        )
        self.chain = self.prompt | self.llm | StrOutputParser()

    def route(self, query: str) -> str:
        return self._route(query)

    @traceable(run_type="chain", name="MedicalRouter")
    def _route(self, query: str) -> str:
        try:
            category = self.chain.invoke({"query": query}).strip().lower()
            if category not in ["retrieval", "direct", "crisis"]:
                return "retrieval" # 默认回退
            return category
        except Exception:
            return "retrieval" # 故障安全


class SemanticRouter:
    """
    语义路由 (Semantic Router) - Local SLM Powered
    使用本地 Qwen3-0.6B 进行快速意图分类。
    V6.1 升级：支持 FAST/STANDARD/EXPERT 三级分诊
    """
    @traceable(run_type="chain", name="SemanticRouter")
    def route(self, query: str) -> dict:
        """
        语义路由 (Semantic Router) - Local SLM Powered
        使用本地模型进行【不思考】的极速意图分类。
        """
        print(f"[DEBUG] Node SemanticRouter.route Start (Query: {query})")
        
        categories = ["GREETING", "CRISIS", "VAGUE_SYMPTOM", "COMPLEX_SYMPTOM"]
        
        try:
            import asyncio
            # [V11.3] 直接调用分类接口，明确 reasoning=False (禁止思考)
            # 这样模型会直接输出类别，延迟从 0.8s 降至 0.1s
            category = asyncio.run(local_slm.constrained_classify(
                query, 
                categories, 
                reasoning=False 
            ))
            
            # 构造统一的返回格式，保持与旧版兼容
            # 意图映射表
            intent_map = {
                "GREETING": "fast",         # 闲聊 -> 快速通道
                "CRISIS": "expert",         # 危重 -> 专家/人工通道
                "VAGUE_SYMPTOM": "vague",   # 模糊 -> 追问/澄清
                "COMPLEX_SYMPTOM": "standard" # 复杂 -> 标准 RAG
            }
            
            path = intent_map.get(category, "standard")
            
            result = {
                "path": path,
                "complexity_score": 8 if path == "expert" else 3,
                "intent_raw": category,
                "reason": "Fast Track Classification (No-Think Mode)"
            }
            
            print(f"[DEBUG] SemanticRouter Result: {result}")
            return result

        except Exception as e:
            logger.error(f"❌ SemanticRouter Error: {e}")
            # 保持错误返回格式的一致性
            return {"path": "standard", "complexity_score": 5, "error": str(e), "intent_raw": "ERROR"}
