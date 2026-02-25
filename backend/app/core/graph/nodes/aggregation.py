import time
import json
import re
import structlog
from typing import Dict, Any
from langchain_core.messages import AIMessage
from langchain_core.runnables.config import RunnableConfig

from app.core.graph.state import AgentState
from app.core.monitoring.tracing import monitor_node
from app.core.models.vram_manager import vram_auto_clear
from app.core.llm.llm_factory import get_fast_llm

logger = structlog.get_logger(__name__)

@monitor_node("expert_aggregation")
@vram_auto_clear(force=False)
async def expert_aggregation_node(state: AgentState, config: RunnableConfig = None) -> Dict[str, Any]:
    """
    节点：专家意见聚合 (Expert Aggregation)
    收集并行专家组的结果并生成最终报告。
    """
    print(f"[DEBUG] Node expert_aggregation Start")
    logger.info("node_start", node="expert_aggregation")
    start = time.time()
    
    status = state.get("status")
    diag = state.get("diagnostician_output", "No diagnosis provided.")
    pharm = state.get("pharmacist_output", "No pharmacy review provided.")
    audit = state.get("auditor_output", "No audit provided.")
    
    # 处理降级状态
    if status == "downgraded":
        logger.info("expert_aggregation_downgrade_mode")
        clinical_report = diag
        user_response = f"【系统提示：当前处于诚实降级模式】\n\n您好，由于专家系统繁忙，已由本地助手为您提供初步建议：\n\n{diag}"
        return {
            "clinical_history": clinical_report,
            "clinical_report": clinical_report,
            "diagnosis_report": clinical_report,
            "final_output": user_response,
            "messages": [AIMessage(content=user_response)],
            "status": "approved"
        }

    # 构造聚合 Prompt
    from app.core.prompts.aggregation import get_expert_aggregation_prompt
    from app.core.utils.json_parser import extract_json_from_text
    
    # [Pain Point #15] Pass Evidence Chain
    evidence_list = state.get("evidence_chain", [])
    evidence_str = json.dumps(evidence_list, ensure_ascii=False, indent=2) if evidence_list else "无"

    prompt = get_expert_aggregation_prompt(diag, pharm, audit, evidence_str)
    
    try:
        llm = get_fast_llm(
            temperature=0.3, 
            task_type="medical_logic_aggregation", 
            prompt_text=prompt
        )
        response = await llm.ainvoke(prompt, config=config)
        content = response.content.strip()
        
        # 使用统一的 JSON 解析工具
        data = extract_json_from_text(content)
        
        if not data:
            # Fallback parsing (保留原有逻辑作为二重保障，或直接抛出)
            import re
            match = re.search(r"(\{.*\})", content, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(1))
                except:
                    pass
            
            if not data:
                # 最后的兜底：如果解析完全失败，将原始内容作为报告
                logger.error("aggregation_json_parse_failed", content=content[:100])
                data = {
                    "clinical_report": diag, # 降级为直接使用诊断意见
                    "user_response": content
                }

        clinical_report = data.get("clinical_report", "")
        # Ensure it's string
        if isinstance(clinical_report, dict) or isinstance(clinical_report, list):
             clinical_report = json.dumps(clinical_report, ensure_ascii=False, indent=2)

        user_response = data.get("user_response", "")
        if isinstance(user_response, dict) or isinstance(user_response, list):
             user_response = json.dumps(user_response, ensure_ascii=False, indent=2)
        
        if len(str(clinical_report)) < 10:
             logger.warning("generated_report_too_short")
             clinical_report = diag # Fallback

    except Exception as e:
        logger.error("expert_aggregation_failed", error=str(e))
        clinical_report = f"# 专家会诊报告\\n\\n## 诊断\\n{diag}\\n\\n## 用药\\n{pharm}\\n\\n## 审计\\n{audit}"
        user_response = "您的专家会诊流程已顺利完成，请您仔细阅读下方的详细医疗报告，如有疑问请继续咨询。"

    # Ensure clinical_report is not None
    if clinical_report is None:
        clinical_report = diag

    logger.info("node_end", node="expert_aggregation", duration=f"{time.time() - start:.2f}s")
    
    resp_meta = getattr(response, "response_metadata", {}) if 'response' in locals() else {}
        
    return {
        "clinical_history": clinical_report,
        "clinical_report": clinical_report,
        "diagnosis_report": clinical_report,
        "final_output": user_response,
        "messages": [AIMessage(content=user_response, response_metadata=resp_meta)],
        "status": "audited"
    }
