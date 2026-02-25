from langgraph.graph import StateGraph, END
from app.core.graph.state import AgentState
from app.core.graph.summarizer_node import summarize_history_node
from app.core.graph.nodes.expert_crew import parallel_expert_crew_node
from app.core.graph.nodes.audit import safety_audit_node
from app.core.graph.nodes.history_retrieval_node import history_retrieval_node

def build_medical_core_graph():
    """
    构建核心逻辑层子图 (Medical Core Sub-graph)
    职责：历史汇总、专家并行诊断、安全审计。
    """
    core = StateGraph(AgentState)
    
    # 添加节点
    # [Pain Point #29] Retrieve Long-term History first
    core.add_node("history_retrieval", history_retrieval_node)
    core.add_node("summarize_history", summarize_history_node)
    core.add_node("expert_crew", parallel_expert_crew_node)
    core.add_node("safety_audit", safety_audit_node)
    
    # 路由逻辑
    def audit_router(state: AgentState):
        status = state.get("status")
        audit_result = state.get("audit_result", {})
        retry_count = state.get("audit_retry_count", 0)
        
        # [Pain Point #20] Human-in-the-loop
        # 如果状态被标记为需要人工审核，则路由到 human_review (需在主图中定义，或在此处处理)
        # 在这里，如果我们希望在 Core 子图内部处理，我们需要引入 Human Review 节点。
        # 但通常 HITL 发生在图的边界或特定节点后。
        # 这里我们简单地返回 'complete'，并在主图的 core_router 中处理 'requires_human_review' 状态。
        # 或者，如果重试次数已满，也结束。
        
        if audit_result.get("passed"):
            return "complete"
            
        if retry_count >= 1: # Max 1 retry
             return "complete" # 即使失败也结束，交由 Egress 处理最终状态
             
        return "expert_crew"

    # 设置边
    # Entry -> History Retrieval
    core.set_entry_point("history_retrieval")
    core.add_edge("history_retrieval", "summarize_history")
    core.add_edge("summarize_history", "expert_crew")
    core.add_edge("expert_crew", "safety_audit")
    
    core.add_conditional_edges(
        "safety_audit",
        audit_router,
        {
            "expert_crew": "expert_crew",
            "complete": END
        }
    )
    
    return core.compile()
