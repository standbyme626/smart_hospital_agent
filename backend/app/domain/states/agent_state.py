from typing import TypedDict, List, Dict, Any, Optional, Annotated
import operator
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages

def replace(old, new):
    return new

class EventContext(TypedDict):
    """
    [Pain Point #33] 统一事件上下文
    将输入解耦为事件类型与负载，支持多模态和非症状类输入。
    """
    event_type: str  # e.g., "SYMPTOM_DESCRIPTION", "IMAGE_UPLOAD", "SYSTEM_COMMAND", "GREETING"
    payload: Dict[str, Any]  # 结构化数据 (如提取出的症状、图片 URL、指令参数)
    raw_input: str   # 原始输入文本
    timestamp: float # 事件发生时间

class UserProfile(BaseModel):
    """
    [Phase 1] 用户静态身份信息
    来自 Auth/HIS 系统，作为 "Source of Truth"
    """
    patient_id: str
    name: str
    age: int
    gender: str
    medical_history: List[str] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)
    identity_verified: bool = False

class OrderContext(BaseModel):
    """
    [Phase 1] 交易/订单上下文
    用于追踪挂号、支付等事务状态
    """
    order_id: Optional[str] = None
    doctor_id: Optional[str] = None
    department: Optional[str] = None
    payment_status: str = "pending"
    mcp_transaction_id: Optional[str] = None

class TriageResult(TypedDict):
    """
    [Phase 1] 结构化分诊结果
    用于 UI 渲染卡片和 HIS 系统对接
    """
    triage_id: str
    recommended_department: str
    confidence: float
    urgency_level: str # "routine", "urgent", "emergency"
    reasoning: str
    suggested_doctors: List[str]

class AgentState(TypedDict, total=False):
    """
    智能医院 Agent 的核心状态定义 (V12.0)。
    
    该状态对象在 LangGraph 的各个节点间传递，作为唯一的真实数据源。
    
    字段说明:
    - messages: 对话历史记录，支持自动追加 (reducer=operator.add)。
    - event: [New] 当前处理的统一事件上下文。
    - user_profile: [New Phase 1] 用户静态身份信息。
    - order_context: [New Phase 1] 订单上下文。
    - triage_result: [New Phase 1] 结构化分诊结果。
    - user_input: [Deprecated] 当前轮次用户的原始输入文本 -> 迁移至 event['raw_input']。
    - intent: 意图识别结果 (如 CRISIS, GREETING, COMPLEX_SYMPTOM)。
    - persona: 患者画像信息 (年龄、性别、既往史等)，持续更新 (Dynamic Persona)。
    - clinical_report: 统一的医疗报告字段 (合并了原有的 diagnosis_report, triage_result)。
    - step: 当前所处的流程步骤名称。
    - next_step: 路由决定的下一个步骤名称。
    - next_node: [Phase 1 New] 路由标记。
    - error: 异常处理信息，若非空则表示上一节点发生错误。
    """
    
    # 消息历史 (LangGraph 自动处理追加)
    messages: Annotated[List[BaseMessage], add_messages]
    
    # [Pain Point #33] 统一事件驱动模型
    event: EventContext

    # [Phase 1 New] 身份与交易上下文
    user_profile: Annotated[Optional[UserProfile], replace]
    order_context: Annotated[Optional[OrderContext], replace]
    triage_result: Optional[TriageResult]
    
    # 输入上下文 (保留兼容，但在逻辑中优先使用 event)
    current_turn_input: Optional[str]
    retrieval_query: Optional[str]
    retrieval_query_variants: Optional[List[Dict[str, Any]]]
    retrieval_top_k: Optional[int]
    retrieval_plan: Optional[Dict[str, Any]]
    retrieval_index_scope: Optional[str]
    variant_hits_map: Optional[Dict[str, List[Dict[str, Any]]]]
    topk_source_ratio: Optional[Dict[str, Any]]
    fusion_method: Optional[str]
    request_id: Optional[str]
    debug_include_nodes: Optional[List[str]]
    debug_snapshots: Optional[Dict[str, Any]]
    user_input: str
    
    # 核心元数据
    intent: str
    persona: Dict[str, Any]
    
    # [Pain Point #3 Fixed] 统一医疗文档字段，避免字段碎片化
    clinical_report: str
    
    # [New] 跨阶段共享的关键状态
    confirmed_diagnosis: Optional[str] # 从 Diagnosis 传递到 Prescription
    
    # 流程控制
    step: str
    next_step: str
    next_node: Optional[str]
    status: str # [Pain Point #20] Added for HITL and general flow control
    error: Optional[str] # Error state
    
    # 审计状态
    audit_retry_count: Annotated[int, operator.add]
    audit_feedback: Optional[str]
    audit_result: Optional[Dict[str, Any]]

    # [Critical] Missing context fields required for Expert Crew
    symptoms: Optional[str]
    medical_history: Optional[str]
    medical_record_summary: Optional[str]
    departments: Optional[List[str]] # Dynamic Dispatcher selection

    # [Critical] Expert Outputs
    diagnostician_output: Optional[str]
    pharmacist_output: Optional[str]
    decision_action: Optional[str]
    decision_reason: Optional[str]
    confidence_score: Optional[float]
    grounded_flag: Optional[bool]

    # [Refactor V12] Subgraph Shared State
    patient_id: Optional[str]
    draft_prescription: Optional[Dict[str, Any]]
    drug_interaction_checks: Optional[List[Dict[str, Any]]]
