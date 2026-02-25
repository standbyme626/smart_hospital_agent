import structlog
from langchain_core.messages import AIMessage
from app.core.graph.state import AgentState

logger = structlog.get_logger(__name__)

async def fast_reply_node(state: AgentState):
    """
    快速回复节点 (Fast Reply)
    用于处理 GREETING 或 异常兜底。
    """
    logger.info("node_start", node="fast_reply")
    
    # 简单的寒暄回复，可以使用 LLM 也可以用模板。
    # 用户要求：由云端大模型以“温和的医院助手”身份回一句寒暄。
    # 为了性能，这里我们也可以用模板，或者调用 Fast LLM。
    # Given "Performance priority", and "Cloud LLM", I will use Fast LLM.
    
    from app.core.llm.llm_factory import get_fast_llm
    from langchain_core.messages import SystemMessage, HumanMessage
    from app.core.config import settings

    llm = get_fast_llm(temperature=0.5)

    event = state.get("event", {})
    user_input = event.get("raw_input", state.get("user_input", ""))
    
    messages = [
        SystemMessage(content="你是一名温和、专业的医院导诊助手。用户正在向你打招呼。请礼貌地回应，并询问有什么可以帮他（如挂号、咨询病情）。字数控制在50字以内。"),
        HumanMessage(content=user_input)
    ]
    
    try:
        response = await llm.ainvoke(messages)
        reply = response.content
    except Exception as e:
        reply = "您好，我是智能导诊助手，请问有什么可以帮您？"
        
    return {
        "messages": [AIMessage(content=reply)],
        "final_output": reply,
        "status": "completed"
    }
