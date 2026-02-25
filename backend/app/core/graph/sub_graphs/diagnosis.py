from typing import Literal, Dict, Any, List
from app.core.llm.llm_factory import get_smart_llm
from langchain_core.messages import SystemMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode

from app.core.config import settings
from app.domain.states.sub_states import DiagnosisState
from app.core.tool_registry import registry
from app.core.prompts.diagnosis import diagnosis_prompt
from app.rag.graph_rag_service import graph_rag_service
from app.rag.dspy_modules import MedicalConsultant
from app.rag.retrieval_planner import build_retrieval_plan
import structlog
import dspy

logger = structlog.get_logger(__name__)

# Initialize DSPy module
medical_consultant = MedicalConsultant()

from app.core.services.config_manager import config_manager

# =================================================================
# Node 1: State_Sync
# 职责: 同步 UserProfile（既往史）到诊断上下文，并加载科室配置
# =================================================================
async def state_sync_node(state: DiagnosisState):
    logger.info("Diagnosis Node: State_Sync Start", state_keys=list(state.keys()))
    messages = state.get("messages", [])
    
    # 尝试从消息历史中提取画像和历史（通常由 Ingress 注入到 SystemMessage）
    history_text = "无历史记录"
    profile_text = "未知患者"
    user_profile_obj = state.get("user_profile")
    
    for msg in messages:
        if isinstance(msg, SystemMessage) and "Patient Profile" in msg.content:
            profile_text = msg.content
            break
    
    # 动态加载科室配置
    # 假设 Triage 阶段已经确定了 intent 或 department，如果没有，默认为全科
    department = state.get("department", "general") # 需要确保 Triage 或上游设置了这个字段
    
    # 如果 State 中没有 department，尝试从 user_profile 或上下文推断
    # 这里为了演示，我们假设如果找不到就用 cardiology (Mock) 或者 general
    if department == "general":
        # 如果未指定科室，尝试根据 User Profile 或消息内容动态判定
        try:
            logger.info("department_classification_start", profile_snippet=profile_text[:50])
            llm = get_smart_llm(temperature=0.0)
            
            # Simple classification prompt
            prompt = f"""Based on the patient profile and messages, classify the most likely medical department.
            
Profile: {profile_text}
History: {history_text}

Available Departments: Cardiology, Respiratory, Gastroenterology, Neurology, Orthopedics, Dermatology, Pediatrics, General.

Output ONLY the department name in English. If uncertain, output 'General'."""

            ai_msg = await llm.ainvoke(prompt)
            inferred_dept = ai_msg.content.strip().replace(".", "")
            
            # Map back to standard keys if needed (simple normalization)
            if inferred_dept.lower() in ["cardiology", "respiratory", "gastroenterology", "neurology", "orthopedics", "dermatology", "pediatrics"]:
                department = inferred_dept.lower()
                logger.info("department_inferred", department=department)
            else:
                department = "general"
                
        except Exception as e:
            logger.warning("department_inference_failed", error=str(e))
            department = "general" # Fallback 
        
    dept_config = config_manager.get_config(department)
    system_prompt = config_manager.get_system_prompt(department)
    
    logger.info("diagnosis_config_loaded", department=department, has_config=bool(dept_config))

    # Keep structured user_profile object unchanged to avoid breaking other nodes.
    # Put textual profile into a separate transient field for diagnosis reasoning.
    return {
        "profile_text": profile_text,
        "department": department,
        "system_prompt": system_prompt # 注入到 State 中供后续节点使用
    }


def _extract_retrieval_query(state: DiagnosisState) -> tuple[str, str]:
    msgs = state.get("messages", [])

    # [Fix] 增强消息提取逻辑，支持 Object, Dict, Tuple, List
    for msg in reversed(msgs):
        msg_type = None
        msg_content = None

        if hasattr(msg, "type") and hasattr(msg, "content"):
            msg_type = msg.type
            msg_content = msg.content
        elif isinstance(msg, dict):
            msg_type = msg.get("type") or msg.get("role")
            msg_content = msg.get("content")
        elif isinstance(msg, (tuple, list)) and len(msg) >= 2:
            msg_type = msg[0]
            msg_content = msg[1]

        if msg_type == "user":
            msg_type = "human"
        if msg_type == "assistant":
            msg_type = "ai"

        if msg_type in ["user", "human"] and msg_content:
            return str(msg_content), "messages"

    # 多级兜底，防止 query 丢失导致 RAG 退化
    retrieval_query = state.get("retrieval_query")
    if isinstance(retrieval_query, str) and retrieval_query.strip():
        return retrieval_query.strip(), "retrieval_query"

    event = state.get("event", {})
    if isinstance(event, dict):
        raw_input = event.get("raw_input")
        if isinstance(raw_input, str) and raw_input.strip():
            return raw_input.strip(), "event.raw_input"

    current_turn_input = state.get("current_turn_input")
    if isinstance(current_turn_input, str) and current_turn_input.strip():
        return current_turn_input.strip(), "current_turn_input"

    user_input = state.get("user_input")
    if isinstance(user_input, str) and user_input.strip():
        return user_input.strip(), "user_input"

    symptoms = state.get("symptoms")
    if isinstance(symptoms, str) and symptoms.strip():
        return symptoms.strip(), "symptoms"

    return "", "none"


# =================================================================
# Node 2: Query_Rewrite
# 职责: Ingress -> Retriever 之间的检索规划（规则优先，模型兜底）
# =================================================================
async def query_rewrite_node(state: DiagnosisState):
    query, source = _extract_retrieval_query(state)
    if not query:
        logger.warning("query_rewrite_no_query_found", available_keys=list(state.keys()))
        return {}

    intent = str(state.get("intent", "") or "")
    plan = await build_retrieval_plan(query=query, intent=intent)
    logger.info(
        "query_rewrite_plan",
        source=source,
        primary_query=plan.primary_query[:120],
        top_k=plan.top_k,
        variants=len(plan.query_variants),
        complexity=plan.complexity,
        rewrite_source=plan.rewrite_source,
    )

    return {
        "retrieval_query": plan.primary_query,
        "retrieval_query_variants": plan.query_variants,
        "retrieval_top_k": plan.top_k,
        "retrieval_plan": plan.to_state_dict(),
        "retrieval_index_scope": plan.index_scope,
    }

# =================================================================
# Node 3: Hybrid_Retriever
# 职责: 启动 Neo4j (GraphRAG) 和 Milvus (VectorRAG) 进行知识检索
# =================================================================
async def hybrid_retriever_node(state: DiagnosisState):
    logger.info("Diagnosis Node: Hybrid_Retriever Start")
    last_user_msg, query_source = _extract_retrieval_query(state)
    query_variants = state.get("retrieval_query_variants", [])
    top_k = state.get("retrieval_top_k", 3) or 3
    index_scope = str(state.get("retrieval_index_scope", "paragraph") or "paragraph")

    logger.info(
        "hybrid_retriever_query",
        query=last_user_msg,
        source=query_source or "none",
        top_k=top_k,
        index_scope=index_scope,
        variant_count=len(query_variants) if isinstance(query_variants, list) else 0,
    )

    if not last_user_msg:
        logger.warning("hybrid_retriever_no_query_found", available_keys=list(state.keys()))
        return {}

    # 简单实体提取 (Placeholder for NER)
    entities = [last_user_msg] 
    
    # 调用 GraphRAG Service (已包含 Vector + Graph 并行检索)
    try:
        context = await graph_rag_service.search(
            query=last_user_msg,
            extracted_entities=entities,
            top_k=max(1, int(top_k)),
            query_variants=query_variants if isinstance(query_variants, list) else None,
            index_scope=index_scope,
        )
        logger.info("hybrid_retriever_success", context_len=len(context))
    except Exception as e:
        logger.error("hybrid_retriever_failed", error=str(e))
        context = "Knowledge retrieval failed."
    
    # 将检索结果作为 SystemMessage 注入上下文，供 DSPy 使用
    # 使用特定前缀以便 DSPy 节点识别
    return {
        "messages": [SystemMessage(content=f"Medical Context:\n{context}")],
        "retrieval_query": last_user_msg,
        "retrieval_top_k": max(1, int(top_k)),
        "retrieval_index_scope": index_scope,
    }

# =================================================================
# Node 4: DSPy_Reasoner
# 职责: 使用 DSPy 结合知识和症状进行结构化推理
# =================================================================
async def dspy_reasoner_node(state: DiagnosisState):
    logger.info("Diagnosis Node: DSPy_Reasoner Start")
    current_loop = state.get("loop_count", 0) + 1
    
    # 1. 准备输入
    # 从 State 中获取 Profile (由 State_Sync 同步)
    profile_text = state.get("profile_text", "")
    if not profile_text:
        user_profile = state.get("user_profile")
        if isinstance(user_profile, str):
            profile_text = user_profile
        elif isinstance(user_profile, dict):
            profile_text = str(user_profile)
        elif user_profile is not None:
            profile_text = str(user_profile)
        else:
            profile_text = "未知患者"
    
    # 从 Messages 中获取 Symptoms 和 Context
    current_symptoms = ""
    conversation_history = ""
    retrieved_knowledge = "未检索到具体知识"
    
    msgs = state.get("messages", [])
    if msgs:
        # Build Conversation History
        history_lines = []
        for msg in msgs:
            role = "unknown"
            if hasattr(msg, 'type'):
                if msg.type in ["user", "human"]: role = "Patient"
                elif msg.type in ["ai", "assistant"]: role = "Doctor"
            
            if role in ["Patient", "Doctor"] and hasattr(msg, 'content'):
                history_lines.append(f"{role}: {msg.content}")
        
        conversation_history = "\n".join(history_lines)

        # Symptoms
        for msg in reversed(msgs):
            if hasattr(msg, 'type') and msg.type in ["user", "human"]:
                current_symptoms = msg.content
                break
        # Context (Look for SystemMessage from Retriever)
        for msg in reversed(msgs):
            if isinstance(msg, SystemMessage) and msg.content and msg.content.startswith("Medical Context"):
                retrieved_knowledge = msg.content
                break

    # 2. 配置 DSPy (Safety Check)
    if not dspy.settings.lm:
         lm = dspy.LM('openai/' + settings.OPENAI_MODEL_SMART, api_key=settings.OPENAI_API_KEY, api_base=settings.OPENAI_API_BASE)
         dspy.configure(lm=lm)

    # 3. 执行推理
    try:
        # [Config] 如果存在动态加载的 System Prompt，应该在这里影响 DSPy 的行为
        # 目前 DSPy 的 Signature 是静态定义的。
        # 我们可以通过 context 参数传递额外指令，或者在 medical_consultant 内部处理。
        # 这里我们将 system_prompt 作为 context 的一部分前置。
        
        system_prompt = state.get("system_prompt", "")
        if system_prompt:
             retrieved_knowledge = f"【系统指令】\n{system_prompt}\n\n【检索知识】\n{retrieved_knowledge}"

        prediction = medical_consultant(
            patient_profile=profile_text, 
            medical_history=profile_text, # 暂时复用 Profile
            conversation_history=conversation_history or "无历史记录",
            current_symptoms=current_symptoms or "无",
            retrieved_knowledge=retrieved_knowledge
        )
        
        # 4. 保存原始结果到 State
        # 我们需要将 DSPy 的结果传递给 Evaluator
        # 由于 State 类型限制，我们可以将其序列化存入 last_tool_result 或者构造一个临时的 AIMessage
        # 这里我们选择构造一个特殊的 AIMessage，包含 reasoning 元数据
        
        reasoning = prediction.reasoning
        diagnosis_list = prediction.suggested_diagnosis
        confidence = float(prediction.confidence_score) if prediction.confidence_score else 0.0
        follow_ups = prediction.follow_up_questions
        
        diag_str = ", ".join(diagnosis_list) if isinstance(diagnosis_list, list) else str(diagnosis_list)
        
        # 将结果暂存，不在 User 可见的消息流中显示（Evaluator 决定是否显示）
        # 但 LangGraph 的 messages 是追加的。
        # 我们这里返回一个 dict 更新 state 的临时字段，或者使用 last_tool_result 作为一个通用的 payload 容器
        
        result_payload = {
            "diagnosis": diag_str,
            "confidence": confidence,
            "reasoning": reasoning,
            "follow_ups": follow_ups
        }
        
        logger.info("DSPy Reasoning Result", **result_payload)
        
        return {
            "loop_count": current_loop,
            "last_tool_result": result_payload # HACK: Reuse this field to pass data to next node
        }

    except Exception as e:
        logger.error("DSPy Execution Failed", error=str(e))
        return {"loop_count": current_loop, "last_tool_result": {"error": str(e), "confidence": 0.0}}

# =================================================================
# Node 4: Confidence_Evaluator
# 职责: 判断推理结果是否可靠（>0.8）
# =================================================================
async def confidence_evaluator_node(state: DiagnosisState) -> Literal["end_diagnosis", "clarify_question"]:
    logger.info("Diagnosis Node: Confidence_Evaluator Start")
    
    payload = state.get("last_tool_result", {})
    confidence = payload.get("confidence", 0.0)
    diagnosis_str = payload.get("diagnosis", "")
    
    # [Fix Over-Safety] 即使是紧急情况，也允许完成诊断流程（提供初步判断），而不是提前跳出
    # 如果置信度极低，才进行追问。
    
    # 强制置信度阈值 0.8
    THRESHOLD = 0.8
    
    if confidence > THRESHOLD:
        logger.info("diagnosis_confirmed", confidence=confidence, diagnosis=diagnosis_str)
        return "end_diagnosis"
    elif "紧急" in diagnosis_str or "Emergency" in diagnosis_str:
         # 如果 DSPy 已经识别出紧急情况，即使置信度不高，也应尽快结束并报警，而不是反复追问
         logger.warning("emergency_detected_in_diagnosis", diagnosis=diagnosis_str)
         return "end_diagnosis"
    else:
        logger.info("diagnosis_uncertain_clarify", confidence=confidence)
        return "clarify_question"

# =================================================================
# Helper Nodes for Outputs
# =================================================================
async def generate_report_node(state: DiagnosisState):
    payload = state.get("last_tool_result", {})
    diag_str = payload.get("diagnosis", "Unknown")
    reasoning = payload.get("reasoning", "")
    confidence = payload.get("confidence", 0.0)
    
    report = f"【DSPy 诊断报告】\n诊断: {diag_str}\n置信度: {confidence}\n依据: {reasoning}"
    
    # [Smoke Test Requirement] Auto-switch to REGISTRATION if diagnosis is confirmed
    # This allows the flow to proceed to Service/Booking
    intent_update = "REGISTRATION" if diag_str and diag_str != "Unknown" else None

    return {
        "confirmed_diagnosis": diag_str,
        "clinical_report": report,
        "is_diagnosis_confirmed": True,
        "intent": intent_update, # Update intent to trigger router
        "messages": [AIMessage(content=report)]
    }

async def generate_question_node(state: DiagnosisState):
    payload = state.get("last_tool_result", {})
    follow_ups = payload.get("follow_ups", [])
    confidence = payload.get("confidence", 0.0)
    
    question = ""
    if follow_ups:
        if isinstance(follow_ups, list) and len(follow_ups) > 0:
            question = follow_ups[0]
        elif isinstance(follow_ups, str) and len(follow_ups.strip()) > 0:
            question = follow_ups
            
    if not question:
         question = "请您详细描述一下您的主要症状，包括发病时间、持续时间以及是否有其他伴随不适？"
         
    msg = f"基于目前信息，我尚不能完全确定诊断（置信度 {confidence:.2f}）。\n建议补充：{question}"
    
    return {
        "messages": [AIMessage(content=msg)]
        # 不设置 is_diagnosis_confirmed，也不增加 loop_count (reasoner 已增加)
    }

def build_diagnosis_graph():
    """
    构建诊断子图 (Diagnosis Subgraph) - [Phase 3 Refactor]
    Nodes: State_Sync -> Query_Rewrite -> Hybrid_Retriever -> DSPy_Reasoner -> Confidence_Evaluator
    """
    workflow = StateGraph(DiagnosisState)
    
    # 1. 初始化模型 (Legacy fallback)
    llm = get_smart_llm(temperature=0.0)

    # =================================================================
    # Graph Construction
    # =================================================================
    workflow.add_node("State_Sync", state_sync_node)
    workflow.add_node("Query_Rewrite", query_rewrite_node)
    workflow.add_node("Hybrid_Retriever", hybrid_retriever_node)
    workflow.add_node("DSPy_Reasoner", dspy_reasoner_node)
    
    # 动作节点
    workflow.add_node("Diagnosis_Report", generate_report_node)
    workflow.add_node("Clarify_Question", generate_question_node)

    # 边连接
    workflow.set_entry_point("State_Sync")
    workflow.add_edge("State_Sync", "Query_Rewrite")
    workflow.add_edge("Query_Rewrite", "Hybrid_Retriever")
    workflow.add_edge("Hybrid_Retriever", "DSPy_Reasoner")
    
    # 条件边
    workflow.add_conditional_edges(
        "DSPy_Reasoner",
        confidence_evaluator_node,
        {
            "end_diagnosis": "Diagnosis_Report",
            "clarify_question": "Clarify_Question"
        }
    )
    
    workflow.add_edge("Diagnosis_Report", END)
    workflow.add_edge("Clarify_Question", END) # 返回给用户等待输入

    return workflow.compile()
