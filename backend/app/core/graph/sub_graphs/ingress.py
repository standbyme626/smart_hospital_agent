from langgraph.graph import END, StateGraph

from app.core.config import settings
from app.core.graph.nodes.guard import guard_node
from app.core.graph.nodes.history_injector import history_injector_node
from app.core.graph.nodes.intent_classifier import intent_classifier_node
from app.core.graph.nodes.multimodal_processor import multimodal_processor_node
from app.core.graph.nodes.pii_filter import pii_filter_node
from app.core.graph.state import AgentState


def _guard_route(state: AgentState) -> str:
    """Route blocked/crisis early to preserve guard decisions."""
    status = str(state.get("status") or "").lower()
    if status in {"blocked", "crisis"}:
        return "end"
    return "intent_classifier"


def build_ingress_graph():
    """
    Ingress 子图：
    PII -> 多模态解析 -> 历史注入 -> 意图分类。

    保持链路简洁，降低导入失败面，优先保障可用性。
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("pii_filter", pii_filter_node)
    workflow.add_node("multimodal_processor", multimodal_processor_node)
    workflow.add_node("history_injector", history_injector_node)
    workflow.add_node("guard", guard_node)
    workflow.add_node("intent_classifier", intent_classifier_node)

    workflow.set_entry_point("pii_filter")
    workflow.add_edge("pii_filter", "multimodal_processor")
    workflow.add_edge("multimodal_processor", "history_injector")

    if settings.ENABLE_INGRESS_GUARD:
        workflow.add_edge("history_injector", "guard")
        workflow.add_conditional_edges(
            "guard",
            _guard_route,
            {
                "intent_classifier": "intent_classifier",
                "end": END,
            },
        )
    else:
        workflow.add_edge("history_injector", "intent_classifier")

    workflow.add_edge("intent_classifier", END)

    return workflow.compile()
