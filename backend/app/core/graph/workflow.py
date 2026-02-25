import threading

import structlog
from langgraph.graph import StateGraph, END

from app.core.graph.state import AgentState
from app.core.graph.nodes.cache import cache_lookup_node
from app.core.graph.nodes.persistence import persistence_node
from app.core.graph.nodes.fast_reply import fast_reply_node
from app.core.graph.anamnesis_node import anamnesis_node
from app.core.graph.human_node import human_review_node

from app.core.graph.sub_graphs.ingress import build_ingress_graph
from app.core.graph.sub_graphs.medical_core import build_medical_core_graph
from app.core.graph.sub_graphs.egress import build_egress_graph
from app.core.graph.sub_graphs.diagnosis import build_diagnosis_graph
from app.core.graph.sub_graphs.service import service_graph

from langgraph.checkpoint.memory import MemorySaver

logger = structlog.get_logger(__name__)

# 默认使用 MemorySaver。
memory_checkpointer = MemorySaver()


class LazyGraphApp:
    """
    Lazily initialize the compiled graph on first use.
    This avoids import-time heavyweight side effects during tests/tooling.
    """

    def __init__(self, factory):
        self._factory = factory
        self._graph = None
        self._lock = threading.Lock()

    def _get_graph(self):
        if self._graph is None:
            with self._lock:
                if self._graph is None:
                    logger.info("workflow_graph_lazy_init_start")
                    self._graph = self._factory()
                    logger.info("workflow_graph_lazy_init_done")
        return self._graph

    def __getattr__(self, item):
        return getattr(self._get_graph(), item)

# 3. 定义主流程路由

# Cache -> Ingress
async def cache_router(state: AgentState):
    route = "persistence" if state.get("cache_hit") else "ingress"
    logger.debug(
        "workflow_router_decision",
        router="cache_router",
        cache_hit=bool(state.get("cache_hit")),
        route=route,
    )
    return route

# Triage Router (New Hard Routing)
async def triage_router(state: AgentState):
    if state.get("status") == "blocked":
        logger.debug(
            "workflow_router_decision",
            router="triage_router",
            intent=state.get("intent"),
            status=state.get("status"),
            route="persistence",
        )
        return "persistence"
        
    intent = state.get("intent")
    status = state.get("status")
    
    # 拦截直接结束
    if status == "blocked":
        logger.debug(
            "workflow_router_decision",
            router="triage_router",
            intent=intent,
            status=status,
            route="persistence",
        )
        return "persistence"
        
    # [Task: 专家组介入] 即使是 CRISIS，也要先去 expert group 拿专业依据，
    # 由 diagnosis 子图里的 emergency 逻辑快速返回，而不是在这里直接切断。
    if status == "crisis" or intent == "CRISIS":
        logger.debug(
            "workflow_router_decision",
            router="triage_router",
            intent=intent,
            status=status,
            route="diagnosis",
        )
        return "diagnosis"

    # Greeting 直接回复并结束
    if intent == "GREETING":
        logger.debug(
            "workflow_router_decision",
            router="triage_router",
            intent=intent,
            status=status,
            route="fast_reply",
        )
        return "fast_reply"

    # 追问模式 (信息不足)
    if intent == "VAGUE_SYMPTOM":
        logger.debug(
            "workflow_router_decision",
            router="triage_router",
            intent=intent,
            status=status,
            route="anamnesis",
        )
        return "anamnesis"
        
    # 信息查询 (INFO)
    if intent == "INFO":
        logger.debug(
            "workflow_router_decision",
            router="triage_router",
            intent=intent,
            status=status,
            route="medical_core",
        )
        return "medical_core" # 暂时让核心层处理 INFO，或者可以加个单独节点

    # 挂号/预约服务
    if intent in {"REGISTRATION", "SERVICE_BOOKING", "APPOINTMENT", "BOOKING"}:
        logger.debug(
            "workflow_router_decision",
            router="triage_router",
            intent=intent,
            status=status,
            route="service",
        )
        return "service"

    # 医疗咨询进入诊断-RAG链路
    if intent == "MEDICAL_CONSULT":
        logger.debug(
            "workflow_router_decision",
            router="triage_router",
            intent=intent,
            status=status,
            route="diagnosis",
        )
        return "diagnosis"
        
    # 默认专家路径
    logger.debug(
        "workflow_router_decision",
        router="triage_router",
        intent=intent,
        status=status,
        route="diagnosis",
    )
    return "diagnosis"

# Medical Core -> Egress
async def core_router(state: AgentState):
    status = state.get("status")
    
    # [Pain Point #20] Human-in-the-loop Interrupt
    # 如果核心层返回的状态是 requires_human_review，我们可以在这里通过 interrupt 机制
    # LangGraph 的 interrupt_before/after 可以在 compile 时配置。
    # 这里我们通过路由逻辑将流程引导至 human_review 节点（如果存在）或者直接中断。
    # 假设我们添加一个 explicit human_review node 在主流程中。
    
    if status == "requires_human_review":
        logger.debug(
            "workflow_router_decision",
            router="core_router",
            status=status,
            route="human_review",
        )
        return "human_review" # Route to explicit HITL node
        
    if status == "rejected":
        logger.debug(
            "workflow_router_decision",
            router="core_router",
            status=status,
            route="persistence",
        )
        return "persistence" # 审计完全失败
    logger.debug(
        "workflow_router_decision",
        router="core_router",
        status=status,
        route="egress",
    )
    return "egress"

# HITL Node -> Egress (Approved) or Persistence (Rejected)
async def human_review_router(state: AgentState):
    route = "egress" if state.get("status") == "approved" else "persistence"
    logger.debug(
        "workflow_router_decision",
        router="human_review_router",
        status=state.get("status"),
        route=route,
    )
    return route

def create_agent_graph(checkpointer=None):
    """
    [V12.0 Architecture Refactor]
    
    采用三层子图架构 (Ingress -> Core -> Egress)，实现极致解耦与性能。
    1. Ingress: 处理安全、意图、画像、危机。
    2. Medical Core: 处理历史、专家组并行诊断、审计。
    3. Egress: 处理结果聚合、质量控制。
    """
    workflow = StateGraph(AgentState)

    # 1. 注册基础节点
    workflow.add_node("cache_lookup", cache_lookup_node)
    workflow.add_node("persistence", persistence_node)
    workflow.add_node("anamnesis", anamnesis_node)
    workflow.add_node("fast_reply", fast_reply_node)
    
    # 2. 注册子图
    ingress_graph = build_ingress_graph()
    medical_core_graph = build_medical_core_graph()
    egress_graph = build_egress_graph()
    diagnosis_graph = build_diagnosis_graph()
    
    workflow.add_node("ingress", ingress_graph)
    workflow.add_node("medical_core", medical_core_graph)
    workflow.add_node("egress", egress_graph)
    workflow.add_node("diagnosis", diagnosis_graph)
    workflow.add_node("service", service_graph)

    # 4. 设置连接边
    workflow.set_entry_point("cache_lookup")
    
    workflow.add_node("human_review", human_review_node) # Register HITL node
    
    workflow.add_conditional_edges(
        "cache_lookup",
        cache_router,
        {
            "persistence": "persistence",
            "ingress": "ingress"
        }
    )
    
    # workflow.add_edge("pii_filter", "triage")
    
    workflow.add_conditional_edges(
        "ingress",
        triage_router,
        {
            "fast_reply": "fast_reply",
            "medical_core": "medical_core",
            "diagnosis": "diagnosis",
            "service": "service",
            "anamnesis": "anamnesis",
            "persistence": "persistence"
        }
    )
    
    workflow.add_edge("fast_reply", "persistence")
    
    workflow.add_edge("anamnesis", "persistence")
    workflow.add_edge("service", "persistence")
    
    workflow.add_conditional_edges(
        "medical_core",
        core_router,
        {
            "persistence": "persistence",
            "egress": "egress",
            "human_review": "human_review"
        }
    )

    # Diagnosis -> Egress (假设诊断完成后进行汇总审计)
    # 如果 Diagnosis 返回了 Clarify Question (is_diagnosis_confirmed=False), 
    # Egress 可能会处理格式化或者直接透传给 Persistence
    workflow.add_edge("diagnosis", "egress")
    
    workflow.add_conditional_edges(
        "human_review",
        human_review_router,
        {
            "egress": "egress",
            "persistence": "persistence"
        }
    )
    
    workflow.add_edge("egress", "persistence")
    workflow.add_edge("persistence", END)

    # 5. 编译
    # [Pain Point #20] Interrupt before human review
    # 这允许我们在进入 human_review 节点前暂停执行，等待人工输入
    # 如果没有传入 checkpointer，使用默认的 memory_checkpointer
    cp = checkpointer if checkpointer is not None else memory_checkpointer
    
    return workflow.compile(
        checkpointer=cp,
        interrupt_before=["human_review"] 
    )

# 为了向后兼容，提供别名
build_medical_graph = create_agent_graph

# Lazily resolve the graph for compatibility with existing imports.
app = LazyGraphApp(lambda: create_agent_graph())
