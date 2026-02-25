import time
import structlog
from typing import Dict, Any
from app.core.graph.state import AgentState
from app.core.monitoring.tracing import monitor_node
from langchain_core.messages import AIMessage

logger = structlog.get_logger(__name__)

class SecurityGuard:
    """
    安全护栏节点 (Security Guard)
    
    [V12.0 Optimization]
    零延迟正则匹配 + 医疗危机快速识别。
    职责：拦截违规内容，标记医疗危机。
    """
    
    # 极端行为与违规拦截 (Security Guard - 保安模式)
    SECURITY_RISK_LIST = [
        "毒品", "制毒", "摇头丸", "购买处方药", "黑市", "炸药", "恐怖袭击"
    ]
    
    # 医疗危机识别 (Medical Crisis - 急诊模式)
    CRISIS_KEYWORDS_LIST = [
        "胸口疼", "胸痛", "喘不上气", "呼吸困难", "救命", "晕倒", "意识不清",
        "大出血", "半身不遂", "肢体麻木", "剧烈头痛", "吞咽困难", "吐血",
        "想死", "自杀", "suicide", "jump off", "不想活", "安眠药自杀", "割腕",
        "结束生命", "活不下去", "轻生"
    ]

    async def run(self, state: AgentState) -> Dict[str, Any]:
        print(f"[DEBUG] Node SecurityGuard Start")
        logger.info("node_start", node="guard")
        
        # 获取输入
        messages = state.get("messages", [])
        user_input = ""
        event = state.get("event", {})
        if isinstance(event, dict) and event.get("raw_input"):
            user_input = str(event.get("raw_input", ""))
        if not user_input and state.get("current_turn_input"):
            user_input = str(state.get("current_turn_input", ""))
        if not user_input and messages:
            from langchain_core.messages import HumanMessage
            for msg in reversed(messages):
                if isinstance(msg, HumanMessage) or (isinstance(msg, dict) and msg.get("type") == "human"):
                    user_input = msg.content if hasattr(msg, "content") else msg.get("content", "")
                    break
        
        if not user_input:
            user_input = state.get("symptoms", "")

        user_input_clean = user_input.strip().lower()
        
        # 1. 检查违规
        if any(k in user_input_clean for k in self.SECURITY_RISK_LIST):
            logger.warning("guard_security_blocked", input=user_input)
            msg = "检测到违规内容。本系统仅提供医疗咨询服务，不讨论违法或自残行为。"
            return {
                "status": "blocked",
                "final_output": msg,
                "messages": [AIMessage(content=msg)]
            }
            
        # 2. 检查医疗危机 (Fast-Pass)
        if any(k in user_input_clean for k in self.CRISIS_KEYWORDS_LIST):
            logger.info("guard_crisis_detected", input=user_input)
            return {
                "status": "crisis",
                "intent": "CRISIS"
            }

        # 3. 安全通过
        return {"status": "safe"}

# 实例化
security_guard = SecurityGuard()

@monitor_node("guard")
async def guard_node(state: AgentState):
    return await security_guard.run(state)
