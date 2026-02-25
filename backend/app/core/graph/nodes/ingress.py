import structlog
from typing import Dict, Any
from app.domain.states.agent_state import AgentState
from app.core.services.auth import AuthService
from app.core.guard.guard_service import LiteGuardService
from langchain_core.messages import AIMessage

logger = structlog.get_logger(__name__)

async def ingress_node(state: AgentState) -> Dict[str, Any]:
    """
    Ingress Node: 身份感知闭环入口
    1. 身份提取
    2. 身份注入
    3. 安全检测
    4. 路由决策
    """
    print("[DEBUG] Node ingress Start")
    
    # 1. 身份提取 (Identity Extraction)
    # 优先从 event.payload 获取，其次尝试 state 现有信息
    event = state.get("event", {})
    payload = event.get("payload", {}) if event else {}
    
    user_id = payload.get("user_id")
    if not user_id:
        # 尝试从 metadata 或其他位置获取，或者默认为匿名
        user_id = state.get("patient_id") or "anonymous"
        
    # 2. 身份注入 (Identity Injection)
    auth_service = AuthService()
    user_profile = await auth_service.mock_get_user_profile(user_id)
    
    # 3. 安全检测 (Security Check)
    raw_input = ""
    if event and event.get("raw_input"):
        raw_input = event["raw_input"]
    elif state.get("user_input"):
        raw_input = state["user_input"]
        
    if raw_input:
        # 归一化处理 (Normalization) - User Rule
        raw_input_normalized = raw_input.strip().lower()
        if len(raw_input_normalized) < 1: # Basic check
             pass # Let it slide or handle as empty
             
        guard_service = LiteGuardService()
        # validate is async
        guard_result = await guard_service.validate(raw_input)
        
        if not guard_result["allowed"]:
            refusal_msg = guard_result.get("response", "请求被拒绝")
            logger.warning("ingress_blocked", user_id=user_id, input=raw_input, reason=refusal_msg)
            
            return {
                "user_profile": user_profile,
                "messages": [AIMessage(content=refusal_msg)],
                "next_node": "end",
                "status": "blocked",
                "patient_id": user_profile.patient_id
            }

    # 4. 路由决策 (Routing)
    # 如果通过安全检查，路由至 History Injector (Step 3: ingress -> history_injector -> triage)
    logger.info("ingress_success", user_id=user_id, identity_verified=user_profile.identity_verified)
    
    return {
        "user_profile": user_profile,
        "next_node": "history_injector",
        "status": "processing",
        "patient_id": user_profile.patient_id
    }
