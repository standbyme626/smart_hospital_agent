import time
import json
import re
import structlog
from typing import Dict, Any
from app.core.graph.state import AgentState
from app.core.monitoring.tracing import monitor_node
from app.core.models.vram_manager import vram_auto_clear
from app.core.config import settings
from app.core.llm.llm_factory import SmartRotatingLLM
from langchain_core.runnables.config import RunnableConfig

logger = structlog.get_logger(__name__)

async def run_auditor_node(state: AgentState, config: RunnableConfig = None) -> Dict[str, Any]:
    """运行审计节点"""
    try:
        llm = SmartRotatingLLM(temperature=settings.TEMPERATURE, max_tokens=1024, prefer_local=False)
        
        diag_out = state.get("diagnostician_output", "")
        pharm_out = state.get("pharmacist_output", "")
        
        audit_description = f"""
        你现在的身份是高级医疗审计专家 (Senior Medical Auditor)。
        请对以下专家组的会诊内容进行严格合规性审查。

        【诊断结论 (Diagnosis)】：
        {diag_out}
        
        【药剂建议 (Pharmacy)】：
        {pharm_out}
        
        【患者背景 (Patient Context)】：
        主诉症状：{state.get('symptoms', '')}
        既往史：{state.get('medical_history', '')}
        
        请执行以下审计指令：
        1. 逻辑校验：诊断是否完全对应主诉？
        2. 安全校验：药剂是否避开了既往史中的禁忌（如过敏、基础病）？
        3. 确诊限制：AI 严禁下达确诊结论 或开具处方，只能提供建议。

        请务必以 JSON 格式输出结果，不要包含 markdown 代码块：
        {{
            "passed": true/false,
            "risk_level": "low/medium/high",
            "reason": "简要说明理由 (中文)",
            "feedback": "如果未通过，请给出给专家组的修改建议 (中文); 如果通过，请整合诊断和药剂给出最终诊疗报告 (中文)"
        }}
        """
        
        response = await llm.ainvoke(audit_description)
        raw_output = response.content.strip()
        
        # Parse JSON
        clean_json = re.sub(r'```json\s*|\s*```', '', raw_output).strip()
        try:
            audit_result = json.loads(clean_json)
        except json.JSONDecodeError:
             logger.warning("auditor_json_parse_failed", raw=raw_output)
             audit_result = {
                 "passed": False, 
                 "risk_level": "high", 
                 "reason": "JSON Parsing Failed",
                 "feedback": "系统错误：审计输出格式异常，请重试。"
             }

        updates = {
            "audit_result": audit_result,
            "auditor_output": audit_result.get("feedback", raw_output)
        }
        
        if not audit_result.get("passed", False):
            # [Pain Point #20] Human-in-the-loop for High Risk
            # 如果审计未通过且风险等级为高，或者明确标记需要人工介入
            if audit_result.get("risk_level") == "high":
                logger.warning("high_risk_detected_interrupting", reason=audit_result.get("reason"))
                # 我们通过返回特殊状态码或标记来通知图进行中断
                # 注意：实际的中断逻辑需要在图定义中配置 interrupt_after=['safety_audit'] 或类似机制
                # 但在这里，我们可以将状态标记为 'requires_human_review'
                updates["status"] = "requires_human_review"
            
            updates["audit_retry_count"] = 1 # Incremented via operator.add in state
            updates["audit_feedback"] = audit_result.get("feedback", "")
            
        return updates
    except Exception as e:
        logger.error("auditor_failed", error=str(e))
        raise e

@monitor_node("safety_audit")
@vram_auto_clear(force=False)
async def safety_audit_node(state: AgentState, config: RunnableConfig = None) -> Dict[str, Any]:
    """安全审计节点"""
    print(f"[DEBUG] Node SafetyAudit Start")
    return await run_auditor_node(state, config)
