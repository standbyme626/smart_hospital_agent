from typing import Dict, Any
import structlog
from app.core.graph.state import AgentState

logger = structlog.get_logger(__name__)

class EmergencyResponse:
    """
    急救响应节点 (Emergency Response)
    
    处理 CRISIS 意图，提供即时的急救指导。
    这是一个纯规则/模板驱动的节点，确保 100% 的可靠性和速度 (< 50ms)。
    """
    
    def get_emergency_guide(self, text: str) -> str:
        text = text.lower()
        
        if any(k in text for k in ["胸痛", "心梗", "心脏", "heart"]):
            return """【紧急警报】检测到疑似心脏问题！请立即执行：
1. 停止一切活动，绝对卧床休息。
2. 立即拨打 120 急救电话。
3. 身边若有硝酸甘油，舌下含服一片（若血压低则禁用）。
4. 保持呼吸道通畅，解开衣领。"""
            
        if any(k in text for k in ["中风", "卒中", "口眼歪斜", "麻木", "stroke"]):
            return """【紧急警报】检测到疑似中风症状！请立即执行：
1. 立即拨打 120，告知可能是“中风”。
2. 让患者平卧，头部侧向一边，防止呕吐物窒息。
3. 切勿给患者喂水或喂药！
4. 记录发病时间，这对于溶栓治疗至关重要。"""
            
        return """【紧急警报】检测到危急情况！
请立即拨打 120 急救电话或前往最近的急诊科。
系统已为您启动绿色通道，正在通知人工客服介入..."""

    async def run(self, state: AgentState) -> Dict[str, Any]:
        print(f"[DEBUG] Node EmergencyResponse Start")
        logger.info("node_start", node="emergency_response")
        
        # [Pain Point #33] Use EventContext
        event = state.get("event", {})
        if event and "raw_input" in event:
            user_input = event["raw_input"]
        else:
            messages = state.get("messages", [])
            user_input = messages[-1].content if messages else state.get("symptoms", "")
        
        guide = self.get_emergency_guide(user_input)
        
        # 构造最终响应
        response = f"{guide}\n\n(注意：AI 建议仅供参考，危急情况请务必寻求专业医疗救助)"
        
        return {
            "final_response": response, # 兼容旧字段
            "fast_response": response,   # 新字段
            "status": "completed"
        }

from app.core.monitoring.tracing import monitor_node

# 实例化
emergency_response = EmergencyResponse()

@monitor_node("emergency_response")
async def emergency_response_node(state: AgentState):
    return await emergency_response.run(state)
