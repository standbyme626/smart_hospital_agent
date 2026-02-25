from typing import Dict, Any, List
import structlog
from langchain_core.messages import SystemMessage
from app.domain.states.agent_state import AgentState

logger = structlog.get_logger(__name__)

async def history_injector_node(state: AgentState) -> Dict[str, Any]:
    """
    [Phase 1] 历史注入器
    将 UserProfile 中的静态病历转换为 SystemMessage 并注入对话历史。
    """
    logger.info("node_start", node="history_injector")
    
    user_profile = state.get("user_profile")
    if not user_profile:
        logger.info("no_user_profile_skip")
        return {"next_node": "triage"}

    # 构建系统消息内容
    # 格式：【背景信息】患者既往史：...，过敏史：...。请在后续诊断中严格参考此信息。
    history_str = ", ".join(user_profile.medical_history) if user_profile.medical_history else "无"
    allergies_str = ", ".join(user_profile.allergies) if user_profile.allergies else "无"
    
    content = (
        f"【背景信息】\n"
        f"患者姓名：{user_profile.name} (ID: {user_profile.patient_id})\n"
        f"年龄：{user_profile.age}, 性别：{user_profile.gender}\n"
        f"既往史：{history_str}\n"
        f"过敏史：{allergies_str}\n"
        f"请在后续诊断中严格参考此信息，确保医疗安全。"
    )
    
    system_msg = SystemMessage(content=content)
    
    # 注入到 messages 列表 (利用 AgentState 的 reducer 机制，这里返回列表会被追加)
    # 注意：LangGraph 的 messages reducer 通常是 operator.add (追加)
    # 如果想插在最前面，可能需要特殊处理，但通常 SystemMessage 可以在任意位置，
    # 只要在 HumanMessage 之前即可被 LLM 关注到。
    # 这里我们简单地返回一个包含该 SystemMessage 的列表。
    
    logger.info("history_injected", patient_id=user_profile.patient_id)
    
    return {
        "messages": [system_msg],
        "next_node": "triage"
    }
