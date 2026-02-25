from typing import Dict, Any
import time
from langchain_core.runnables import RunnableConfig
import structlog

from app.core.graph.state import AgentState
from app.core.monitoring.tracing import monitor_node
from app.core.models.local_slm import local_slm
from app.core.models.slm_adapter import LocalSLMAdapter
from app.core.middleware.slm_middleware import HospitalSLMMiddleware
from langchain.agents import create_agent

logger = structlog.get_logger(__name__)

# --- Configuration & Initialization ---

SYSTEM_PROMPT = """你是一个急诊分诊助手。
任务：根据患者描述判断病情紧急程度。
类别：
- CRISIS (危急): 胸痛、呼吸困难、大出血、昏迷。
- GREETING (闲聊): 你好、再见。
- VAGUE_SYMPTOM (模糊): 头疼、不舒服（无细节）。
- COMPLEX_SYMPTOM (普通): 感冒、发烧、拉肚子（有细节）。

说明：
1. 你可以先在 <think> 标签中思考。
2. 思考结束后，必须换行输出：[TRIAGE] 类别: 原因

示例：
用户: 我胸口疼，喘不上气
回复:
<think>胸痛是危急症状</think>
[TRIAGE] CRISIS: 胸痛且呼吸困难

用户: 你好
回复:
<think>这是打招呼</think>
[TRIAGE] GREETING: 打招呼
"""

from app.core.llm.llm_factory import get_fast_llm
from langchain_core.messages import HumanMessage, SystemMessage
import json
import asyncio

logger = structlog.get_logger(__name__)

# --- Configuration & Initialization ---

SYSTEM_PROMPT = """你是一个极速医疗意图分类器。
你的任务是根据患者的主诉将其分类到以下类别之一。
禁止任何解释，禁止输出 <think> 标签，禁止输出多余文字。
必须仅返回指定的类别标签。

类别列表：
- CRISIS: 危急重症（胸痛、呼吸困难、大出血、昏迷、意识不清）
- GREETING: 打招呼、闲聊、礼貌用语
- VAGUE_SYMPTOM: 模糊主诉（如“我不舒服”、“头疼”但无细节）
- COMPLEX_SYMPTOM: 明确的医疗咨询或复杂症状（如“我感冒三天了，流鼻涕咳嗽”）

直接输出类别标签，严禁输出其他内容。"""

@monitor_node("classifier")
async def classifier_node(state: AgentState, config: RunnableConfig = None) -> Dict[str, Any]:
    """
    节点：智能意图分类器 (使用本地 0.6B 模型 + 约束解码)
    """
    print(f"[DEBUG] Node classifier Start")
    logger.info("node_start", node="classifier")
    start_time = time.time()
    
    # [Pain Point #33] Use EventContext
    event = state.get("event", {})
    if event and "raw_input" in event:
        user_input = event["raw_input"]
    else:
        user_input = state.get("symptoms", "")
        
    if not user_input or not user_input.strip():
        return {"intent": "GREETING", "reason": "Empty input"}
        
    if user_input.strip().lower() in ["hi", "hello", "你好", "您好"]:
        return {"intent": "GREETING", "reason": "Exact match greeting"}

    categories = ["CRISIS", "GREETING", "VAGUE_SYMPTOM", "COMPLEX_SYMPTOM"]
    
    try:
        # 1. 尝试使用本地 0.6B 模型的约束解码 (0.5s - 1.0s)
        # [Fix] Check config first
        from app.core.config import settings
        if not settings.ENABLE_LOCAL_FALLBACK:
             raise RuntimeError("Local SLM disabled")

        intent = await local_slm.constrained_classify(user_input, categories)
        duration = time.time() - start_time
        logger.info("classification_result", intent=intent, source="local_slm", duration=f"{duration:.2f}s")
        return {"intent": intent, "reason": "Local constrained classification"}
        
    except Exception as e:
        logger.warning("local_classification_failed", error=str(e))
        
        # 2. 本地失败后降级到云端极速模式 (Fallback)
        llm = get_fast_llm(temperature=0.0, prefer_local=False) 
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"用户主诉: {user_input}")
        ]

        try:
            response = await asyncio.wait_for(llm.ainvoke(messages), timeout=3.0)
            content = response.content.strip().upper()
            
            intent = "VAGUE_SYMPTOM"
            for candidate in categories:
                if candidate in content:
                    intent = candidate
                    break
                    
            logger.info("classification_result", intent=intent, source="cloud_fallback")
            return {"intent": intent, "reason": "Cloud fallback classification"}
            
        except Exception as cloud_err:
            logger.error("all_classification_failed", error=str(cloud_err))
            return {"intent": "VAGUE_SYMPTOM", "reason": "Final fallback"}
        logger.warning("classifier_timeout", limit="5s")
        return {"intent": "VAGUE_SYMPTOM", "reason": "Timeout fallback"}
    except Exception as e:
        logger.error("classifier_failed", error=str(e))
        return {"intent": "VAGUE_SYMPTOM", "reason": f"Error: {str(e)}"}

@monitor_node("fast_responder")
async def fast_responder_node(state: AgentState, config: RunnableConfig = None) -> Dict[str, Any]:
    """
    节点：快速响应 (Fast Responder)
    处理 GREETING 等无需调用复杂医疗模型的场景。
    """
    print(f"[DEBUG] Node fast_responder Start")
    logger.info("node_start", node="fast_responder")
    intent = state.get("intent", "")
    
    # 根据 Intent 返回预设话术
    if intent == "GREETING":
        return {"diagnosis_report": "您好！我是您的智能分诊助手。请告诉我您哪里不舒服？"}
    
    # 默认兜底
    return {"diagnosis_report": "请详细描述您的症状，以便我为您分诊。"}
