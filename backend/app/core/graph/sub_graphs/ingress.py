from langgraph.graph import StateGraph, END
from app.core.graph.state import AgentState
from app.core.graph.nodes.intent_classifier import intent_classifier_node # [V11.29] Use Optimized Local SLM
from app.core.graph.nodes.fast_track import fast_track_node
from app.core.graph.nodes.guard import guard_node
from app.core.graph.nodes.pii_filter import pii_filter_node
from app.core.graph.nodes.multimodal_processor import multimodal_processor_node
from app.core.graph.nodes.history_injector import history_injector_node # [Pain Point #29]

async def route_ingress(state: AgentState):
    """
    Ingress Routing Logic
    """
    # [Refactor] All inputs go to IntentClassifier (Local SLM)
    # The heuristic check is removed to ensure consistent intent analysis
    return "intent_classifier"


async def route_after_guard(state: AgentState):
    status = (state.get("status") or "").lower()
    if status in {"blocked", "crisis"}:
        return "end"
    return "intent_classifier"

# [Pain Point #26] PII Filter is the first step
def build_ingress_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("pii_filter", pii_filter_node)
    workflow.add_node("multimodal_processor", multimodal_processor_node)
    workflow.add_node("history_injector", history_injector_node) # [Pain Point #29]
    workflow.add_node("guard", guard_node)
    workflow.add_node("intent_classifier", intent_classifier_node) # [Refactor] Replaced unified_preprocessor
    # workflow.add_node("fast_track", fast_track_node) # Removed legacy fast_track

    # Entry -> PII Filter
    workflow.set_entry_point("pii_filter")
    
    # PII -> Multimodal (Check for images)
    workflow.add_edge("pii_filter", "multimodal_processor")

    # Multimodal -> History Injector [New Flow]
    workflow.add_edge("multimodal_processor", "history_injector")

    # History -> Guard -> IntentClassifier / End(crisis|blocked)
    workflow.add_edge("history_injector", "guard")
    workflow.add_conditional_edges(
        "guard",
        route_after_guard,
        {
            "intent_classifier": "intent_classifier",
            "end": END,
        },
    )

    # workflow.add_edge("fast_track", END)
    workflow.add_edge("intent_classifier", END)

    return workflow.compile()
