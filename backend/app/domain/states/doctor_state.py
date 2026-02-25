from typing import Annotated, List, TypedDict, Optional, Dict, Any
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class DoctorState(TypedDict):
    """
    医生工作流的状态定义 (State Definition)
    
    Attributes:
        messages: 聊天记录列表，使用 add_messages reducer 进行追加更新。
        phase: 当前所处的诊疗阶段 ('diagnosis', 'prescription', 'scribe', 'end')。
        patient_id: 患者 ID。
        diagnosis_data: 确诊后的诊断数据。
        prescription_data: 开具后的处方数据。
        symptom_vector: 症状向量，包含部位、性质、程度等。
        audit_result: 审计结果 {"passed": bool, "risk_level": "low/med/high", "reason": "..."}。
        audit_retry_count: 审计重试次数。
    """
    messages: Annotated[List[BaseMessage], add_messages]
    phase: str  # 'diagnosis', 'prescription', 'scribe', 'end'
    patient_id: str
    diagnosis_data: Dict[str, Any]
    prescription_data: Dict[str, Any]
    symptom_vector: Dict[str, Any]
    audit_result: Dict[str, Any]
    audit_retry_count: int
