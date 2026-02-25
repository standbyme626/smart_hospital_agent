from langgraph.graph import StateGraph, END
from app.core.graph.state import AgentState
from app.core.graph.nodes.aggregation import expert_aggregation_node
from app.core.graph.nodes.quality_gate import quality_gate_node

def build_egress_graph():
    """
    构建流出子图 (Egress Sub-graph)
    职责：结果聚合、质量门禁、状态标记。
    """
    egress = StateGraph(AgentState)
    
    # 添加节点
    egress.add_node("expert_aggregation", expert_aggregation_node)
    egress.add_node("quality_gate", quality_gate_node)
    
    # 设置边
    egress.set_entry_point("expert_aggregation")
    egress.add_edge("expert_aggregation", "quality_gate")
    egress.add_edge("quality_gate", END)
    
    return egress.compile()
