import structlog
from langchain_core.messages import AIMessage
from app.core.graph.state import AgentState

logger = structlog.get_logger(__name__)

async def clarify_node(state: AgentState):
    """
    追问节点 (Clarify Node)
    当置信度不足时，请求用户补充信息。
    """
    logger.info("node_start", node="clarify")
    
    reply = "抱歉，我没有完全理解您的病情描述。请您详细描述一下您的主要症状、持续时间以及是否有其他不适，以便我为您推荐合适的科室。"
    
    return {
        "messages": [AIMessage(content=reply)],
        "final_output": reply,
        "status": "clarification_needed"
    }
