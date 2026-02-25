from typing import TypedDict, List, Optional, Dict, Any, Annotated
import operator
from langchain_core.messages import BaseMessage

# Reuse existing definitions where appropriate
from app.domain.states.agent_state import EventContext, UserProfile, OrderContext, TriageResult

class TriageState(TypedDict):
    """
    分诊阶段状态
    仅包含与分诊相关的核心字段。
    """
    messages: Annotated[List[BaseMessage], operator.add]
    user_input: str # 原始输入
    symptoms: Optional[str] # 提取的症状
    intent: str # 意图
    triage_result: Optional[TriageResult] # 分诊结果
    
class DiagnosisState(TypedDict):
    """
    诊断阶段状态
    """
    messages: Annotated[List[BaseMessage], operator.add]
    patient_id: str
    user_profile: Optional[UserProfile]
    # 对话历史 (可能需要从 MasterState 传入)
    dialogue_history: List[BaseMessage] 
    current_turn_input: Optional[str]
    retrieval_query: Optional[str]
    retrieval_query_variants: Optional[List[str]]
    retrieval_top_k: Optional[int]
    retrieval_plan: Optional[Dict[str, Any]]
    retrieval_index_scope: Optional[str]
    user_input: Optional[str]
    event: Optional[EventContext]
    
    # 诊断核心数据
    symptoms: str
    current_hypothesis: Optional[str] # 当前假设
    guideline_matches: List[Dict[str, Any]] # 匹配的指南
    differential_diagnosis: List[str] # 鉴别诊断列表
    confirmed_diagnosis: Optional[str] # 最终确诊
    confidence: float
    
    # [Refactor Phase 1] 状态机安全性与去耦合
    loop_count: int = 0  # 诊断循环次数
    is_diagnosis_confirmed: bool = False  # 是否已提交诊断
    last_tool_result: Optional[Dict[str, Any]] = None  # 最后一次工具调用结果
    
    # 状态控制
    step: str
    error: Optional[str]
    intent: Optional[str] # [Added] 允许诊断阶段变更意图 (如转挂号)

class PrescriptionState(TypedDict):
    """
    处方阶段状态
    """
    messages: Annotated[List[BaseMessage], operator.add]
    patient_id: str
    confirmed_diagnosis: str # 必须有确诊才能开处方
    
    # 处方数据
    draft_prescription: Optional[Dict[str, Any]] # 处方草稿
    drug_interaction_checks: List[Dict[str, Any]] # 相互作用检查结果
    audit_feedback: Optional[str] # 审计反馈
    risk_level: str # 风险等级
    
    final_prescription: Optional[Dict[str, Any]] # 最终处方
    
    # 状态控制
    audit_retry_count: int
    step: str
