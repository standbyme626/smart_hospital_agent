import structlog
import re
from typing import Dict, Any, List
from app.core.security.pii import PIIMasker
from app.db.sql_adapter import sql_adapter
from app.core.graph.state import AgentState

logger = structlog.get_logger(__name__)

class AuditService:
    """
    安全与 PII 深度审计服务 (Phase 4)
    负责 Egress 前的最终合规性检查与日志记录。
    """
    
    DISCLAIMER = "\n\n[AI 辅助提示：本建议由智慧医院 Agent 生成，不作为最终诊断依据，请以线下医生意见为准]"

    async def final_audit(self, state: AgentState) -> Dict[str, Any]:
        """
        最终审计节点逻辑
        1. PII 脱敏检查
        2. 越权处方拦截 (Regex)
        3. 强制追加免责声明
        4. 异步写回 HIS 审计日志
        """
        logger.info("final_audit_start")
        
        messages = state.get("messages", [])
        if not messages:
            return {}
            
        last_msg = messages[-1]
        content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
        
        # 1. PII Safety Check (Double Check)
        # 确保输出中不包含身份证号、手机号
        # PIIMasker.mask is synchronous
        safe_content = PIIMasker.mask(content)
        
        # 2. Compliance Check (Anti-Prescription)
        # 简单的正则拦截，防止 AI 直接开具处方药（合规要求）
        # 关键词：处方, Rx, 开药
        if re.search(r"(确诊|开具|处方药|Rx)", safe_content) and "建议" not in safe_content:
             logger.warning("potential_unauthorized_prescription", content=safe_content[:50])
             safe_content = safe_content.replace("确诊", "疑似").replace("开具", "建议使用")
        
        # 3. Append Disclaimer
        if self.DISCLAIMER.strip() not in safe_content:
            safe_content += self.DISCLAIMER
            
        # 4. Async Audit Log (Side-loading)
        user_profile = state.get("user_profile")
        patient_id = user_profile.patient_id if hasattr(user_profile, "patient_id") else "guest"
        
        await sql_adapter.log_audit(
            action="egress_audit", 
            details={
                "patient_id": patient_id, 
                "original_len": len(content), 
                "safe_len": len(safe_content),
                "intent": state.get("intent")
            }
        )
        
        # 更新最后一条消息
        # 我们需要返回更新后的消息列表或仅仅是修改
        # 在 LangGraph 中，通常是追加或替换。
        # 这里我们假设 UI 渲染的是 'safe_content'。
        # 最佳实践是在发送给用户前进行拦截。
        
        # 我们返回修改后的内容，以便后续流程（如果有）或 UI 使用
        if content != safe_content:
            # 如果内容被修改，记录日志
            logger.info("audit_content_modified", original_len=len(content), new_len=len(safe_content))
            
        return {"final_response": safe_content} 
        
audit_service = AuditService()
