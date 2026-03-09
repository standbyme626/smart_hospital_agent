from __future__ import annotations


STATUS_MAP: dict[str, str] = {
    "unified_preprocessor": "正在分析病情...",
    "fast_track": "正在生成回答...",
    "fast_reply": "正在生成回答...",
    "standard_consultant": "全科医生正在思考...",
    "expert_crew": "专家组正在会诊...",
    "expert_aggregation": "正在整合诊断结果...",
    "service": "正在处理挂号服务...",
    "service_agent": "正在处理挂号服务...",
    "tools": "正在处理挂号工具调用...",
    "intent_classifier": "正在识别意图...",
    "diagnosis": "正在进行诊断分析...",
    "Hybrid_Retriever": "正在检索医学知识...",
    "State_Sync": "正在同步患者上下文...",
    "triage_node": "正在分诊...",
    "DSPy_Reasoner": "专家系统正在推理...",
    "Diagnosis_Report": "正在生成诊断报告...",
    "Clarify_Question": "正在生成追问...",
}

STAGE_NODE_MAP: dict[str, str] = {
    "Query_Rewrite": "rewrite",
    "Quick_Triage": "rewrite",
    "Hybrid_Retriever": "retrieve",
    "retriever": "retrieve",
    "DSPy_Reasoner": "judge",
    "Decision_Judge": "judge",
    "Diagnosis_Report": "respond",
    "Clarify_Question": "respond",
    "service": "respond",
    "fast_reply": "respond",
    "booking_shortcut": "respond",
}


def resolve_stage(node: str, event_type: str) -> str:
    node_name = str(node or "").strip()
    if node_name in STAGE_NODE_MAP:
        return STAGE_NODE_MAP[node_name]
    if event_type in {"token", "final"}:
        return "respond"
    if event_type in {"thought", "status", "ping"}:
        return "route"
    return ""


def resolve_status(node: str) -> str:
    return STATUS_MAP.get(str(node or "").strip(), "")
