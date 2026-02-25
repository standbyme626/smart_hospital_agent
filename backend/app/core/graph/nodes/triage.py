import json
import structlog
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from app.core.config import settings
from app.core.llm.llm_factory import get_smart_llm
from app.core.graph.state import AgentState

logger = structlog.get_logger(__name__)

TRIAGE_SYSTEM_PROMPT = """你是由 Smart Hospital 部署的“医院全科导诊台”云端大模型。
你的职责是精准识别用户的意图，并以严格的 JSON 格式输出。

### 意图分类 (Category)
- **GREETING**: 纯寒暄、打招呼、无意义对话 (e.g. "你好", "在吗", "早安")。
- **MEDICAL_CONSULT**: 描述症状、询问病情、寻求用药建议 (e.g. "头痛怎么办", "孩子发烧了", "高血压药怎么吃")。
- **SERVICE_BOOKING**: 挂号、查医生、预约检查、支付 (e.g. "我要挂号", "预约张医生", "查一下CT费用").
- **OFF_TOPIC**: 政治、娱乐、编程等与医疗完全无关的话题 (e.g. "写个Python脚本", "特朗普是谁").

### 输出格式
必须仅输出以下 JSON，不要包含 Markdown 代码块：
{
  "intent": "GREETING | MEDICAL_CONSULT | SERVICE_BOOKING | OFF_TOPIC",
  "confidence": 0.0-1.0,
  "summary": "简要描述用户需求",
  "action_required": true
}
"""

async def triage_node(state: AgentState):
    """
    云端大模型接管意图分诊 (Cloud-LLM Based Triage)
    """
    logger.info("node_start", node="triage")
    
    # 0. Check Blocked Status (e.g. from PII Filter)
    if state.get("status") == "blocked":
        logger.info("triage_skipped_blocked")
        return {}
    
    # 1. 获取用户输入 (优先使用 EventContext)
    event = state.get("event", {})
    if event and "raw_input" in event:
        user_input = event["raw_input"]
    else:
        # Fallback to messages or user_input field
        user_input = state.get("user_input", "")
        if not user_input:
            messages = state.get("messages", [])
            if messages:
                user_input = messages[-1].content
    
    # 2. 调用 Cloud LLM
    llm = get_smart_llm(temperature=0.0)
    
    messages = [
        SystemMessage(content=TRIAGE_SYSTEM_PROMPT),
        HumanMessage(content=user_input or "Wait") # Handle empty input gracefully
    ]
    
    try:
        response = await llm.ainvoke(messages)
        content = response.content.strip()
        
        # 3. 解析 JSON
        # Robust parsing
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].strip()
            
        data = json.loads(content)
        
        intent = data.get("intent", "MEDICAL_CONSULT") 
        confidence = data.get("confidence", 0.0)
        summary = data.get("summary", "")
        
        logger.info("triage_result", intent=intent, confidence=confidence)
        
        return {
            "intent": intent,
            "triage_result": {
                "triage_id": "gen_uuid", 
                "recommended_department": "TBD",
                "confidence": confidence,
                "urgency_level": "routine",
                "reasoning": summary,
                "suggested_doctors": []
            }
        }
        
    except Exception as e:
        logger.error("triage_failed", error=str(e))
        # Fallback handling
        return {
            "intent": "GREETING", 
            "error": str(e),
            "triage_result": {
                "triage_id": "error",
                "recommended_department": "None",
                "confidence": 0.0,
                "urgency_level": "routine",
                "reasoning": "System Error Fallback",
                "suggested_doctors": []
            }
        }
