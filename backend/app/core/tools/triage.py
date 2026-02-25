from typing import List, Optional
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_fixed

class TriageReport(BaseModel):
    """
    结构化分诊报告模型 (Structured Triage Report)
    定义了分诊 Agent 输出的数据结构。
    """
    department: str = Field(..., description="建议挂号科室 (Recommended department, e.g., 'Cardiology')")
    urgency: str = Field(..., description="紧急程度 (Urgency level: 'Normal', 'Urgent', 'Critical')")
    reasoning: str = Field(..., description="分诊理由简述 (Brief medical reasoning for the recommendation)")
    advice: List[str] = Field(..., description="患者建议列表 (List of immediate advice for the patient)")

@retry(stop=stop_after_attempt(2), wait=wait_fixed(0.1))
def submit_triage_report(
    department: str,
    urgency: str,
    reasoning: str,
    advice: List[str]
) -> str:
    """
    提交最终分诊报告 (Submit Triage Report)
    分诊 Agent 必须调用此工具来结束分诊流程。
    
    Args:
        department: 建议科室 (e.g., '心内科', '急诊科')
        urgency: 紧急程度 ('普通', '加急', '危急')
        reasoning: 分诊依据 (限制 100 字以内)
        advice: 行动建议列表 (e.g., '禁食水', '带这就诊卡')
    
    Returns:
        str: 提交确认消息
    """
    # 在真实系统中，这里会将报告保存到数据库或触发下游工作流。
    # 目前返回一个格式化的标记块，供前端解析或 Agent 确认。
    
    import json
    report = {
        "department": department,
        "urgency": urgency,
        "reasoning": reasoning,
        "advice": advice
    }
    
    # 返回特定标记，以便后端识别会话已完成
    return f"TRIAGE_REPORT_SUBMITTED: {json.dumps(report, ensure_ascii=False)}"
