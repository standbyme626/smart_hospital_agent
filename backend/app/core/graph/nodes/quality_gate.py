import time
import structlog
from typing import Dict, Any
from app.core.graph.state import AgentState
from app.core.monitoring.tracing import monitor_node
from langchain_core.runnables.config import RunnableConfig

logger = structlog.get_logger(__name__)

@monitor_node("quality_gate")
async def quality_gate_node(state: AgentState, config: RunnableConfig = None) -> Dict[str, Any]:
    """节点：质量门禁 (Quality Gate)
    检查诊断报告是否通过审计或包含危险标识。
    """
    print(f"[DEBUG] Node quality_gate Start")
    logger.info("node_start", node="quality_gate")
    start = time.time()
    
    # 获取聚合后的报告
    report = str(state.get("clinical_report", "") or "")
    user_response = str(state.get("final_output", "") or "")
    status = str(state.get("status", "") or "")
    
    # 简单规则校验
    reasons = []
    hard_fail_keywords = ["FAIL", "Compliance Status: Fail", "合规状态：失败", "DANGEROUS"]
    if any(k in report for k in hard_fail_keywords):
        reasons.append("detected_failure_keywords")

    actionable_keywords = [
        "建议", "诊断", "复诊", "急诊", "就医", "观察", "检查", "用药", "处理", "咨询", "挂号",
        "recommend", "diagnosis", "treatment", "emergency"
    ]
    has_actionable = any(k in (report + user_response) for k in actionable_keywords)

    # 降级或危机场景下放宽长度阈值，避免有效答复被误拒
    min_report_len = 8 if status in {"downgraded", "crisis"} else 20
    min_user_resp_len = 6 if status in {"downgraded", "crisis"} else 10

    if len(report) < min_report_len:
        reasons.append("report_too_short")

    if len(user_response) < min_user_resp_len:
        reasons.append("user_response_too_short")

    # 仅因短答被拒时，如果内容可执行则放行
    short_only = set(reasons).issubset({"report_too_short", "user_response_too_short"})
    if short_only and has_actionable:
        reasons = []

    if reasons:
        logger.warning("quality_gate_rejected", reasons=reasons, status=status)
        return {"status": "rejected"}
    
    logger.info("node_end", node="quality_gate", status="approved", duration=f"{time.time() - start:.2f}s")
    return {"status": "approved"}
