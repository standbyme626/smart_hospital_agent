from typing import Annotated, Literal, TypedDict, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from app.core.config import settings
from app.core.tools.diagnosis import submit_diagnosis
from app.core.tools.ehr import query_ehr, query_lab_results
from app.core.tools.guidelines import lookup_guideline
from app.core.tools.medical_tools import check_drug_interaction
from app.core.tools.prescription import submit_prescription
from app.rag.ddinter_checker import DDInterChecker
from app.domain.states.doctor_state import DoctorState
import json

# ==================== State Definition ====================

# DoctorState has been moved to app.domain.states.doctor_state

# ==================== Tools Setup ====================

# Wrap tools to ensure they adhere to LangChain interface
@tool
async def tool_lookup_guideline(query: str):
    """
    查询医学指南 (Search Guidelines)
    
    用于查找诊断标准、治疗方案等权威医学资料。
    
    Args:
        query (str): 查询关键词或问题。
    """
    return await lookup_guideline(query)

@tool
async def tool_check_drug_interaction(drugs: list):
    """
    检查药物相互作用 (Check Drug Interactions)
    
    输入药物名称列表，检查是否存在配伍禁忌。
    
    Args:
        drugs (list): 药物名称列表 (如 ["阿司匹林", "华法林"])。
    """
    return await check_drug_interaction(drugs)

@tool
async def tool_query_ehr(patient_id: str):
    """
    查询电子病历 (Query EHR)
    
    获取患者的历史病史、过敏史等基础信息。
    
    Args:
        patient_id (str): 患者 ID。
    """
    return str(await query_ehr(patient_id))

@tool
async def tool_query_lab(patient_id: str):
    """
    查询检查检验结果 (Query Lab Results)
    
    获取患者最近的生化检查、影像学报告等。
    
    Args:
        patient_id (str): 患者 ID。
    """
    return str(await query_lab_results(patient_id))

@tool
async def tool_submit_diagnosis(diagnosis: str, confidence: str, reasoning: str):
    """
    提交诊断结果 (Submit Diagnosis)
    
    当医生确诊后，必须调用此工具以进入下一阶段。
    
    Args:
        diagnosis (str): 诊断结论。
        confidence (str): 置信度 (如 "High", "Medium")。
        reasoning (str): 诊断依据。
    """
    return await submit_diagnosis(diagnosis, confidence, reasoning)

@tool
async def tool_submit_prescription(patient_id: str, diagnosis: str, prescription_list: list):
    """
    提交处方 (Submit Prescription)
    
    开具处方并结束当前诊疗会话。
    
    Args:
        patient_id (str): 患者 ID。
        diagnosis (str): 诊断结论。
        prescription_list (list): 药品清单。
    """
    return await submit_prescription(patient_id, diagnosis, prescription_list)

# ==================== Prompts ====================

DIAGNOSIS_PROMPT = """你是由"智能医院系统"驱动的主治医生。
当前阶段: **问诊与诊断 (Diagnosis)**

你的任务:
1. 通过问诊（可使用 `query_ehr`, `query_lab`）了解病情。
2. 结合 `lookup_guideline` 辅助鉴别诊断。
3. 确认诊断后，**必须**调用 `tool_submit_diagnosis` 进入下一阶段。

注意:
- 保持专业、耐心。
- 每次只问一个核心问题。
- 若无法确诊，建议进一步检查。
"""

PRESCRIPTION_PROMPT = """你是由"智能医院系统"驱动的主治医生。
当前阶段: **开具处方 (Prescription)**

已确诊: {diagnosis}
(置信度: {confidence})

你的任务:
1. 根据诊断开具处方。
2. **必须**调用 `tool_check_drug_interaction` 检查药物安全。
3. 确认无误后，调用 `tool_submit_prescription` 提交。

注意:
- 若发现相互作用风险，需调整处方。
- 处方应包含: 药名、剂量、数量。
"""

# ==================== Nodes ====================

from app.core.llm.llm_factory import get_smart_llm

def get_model():
    """
    获取 LLM 模型实例 (已统一使用 SmartRotatingLLM)
    """
    return get_smart_llm()

async def diagnosis_node(state: DoctorState):
    """
    诊断与开方节点 (原 doctor_node)
    
    核心决策节点。根据当前阶段 (Phase) 动态加载 System Prompt 和 Tools。
    """
    phase = state.get("phase", "diagnosis")
    messages = state["messages"]
    model = get_model()
    
    # 获取审计反馈 (如果有)
    audit_result = state.get("audit_result", {})
    audit_feedback = ""
    if audit_result and not audit_result.get("passed", True):
        audit_feedback = f"\n\n⚠️ 【安全审计警报】:\n由于以下原因，上一次的建议未通过审计：{audit_result.get('reason')}\n请重新评估并修正方案。"

    if phase == "diagnosis":
        tools = [tool_lookup_guideline, tool_query_ehr, tool_query_lab, tool_submit_diagnosis]
        
        # --- 自动注入转诊上下文 (Context Injection) ---
        referral_context = ""
        # 扫描是否有转诊系统通知
        for m in messages:
            if isinstance(m, (HumanMessage, SystemMessage)) and "【系统通知：新患者转诊接入】" in str(m.content):
                referral_context = m.content
                break
        
        # 动态构建 System Prompt
        prompt_content = DIAGNOSIS_PROMPT
        if referral_context:
            prompt_content += f"\n\n{referral_context}\n(请基于上述转诊信息开始问诊)"
        
        if audit_feedback:
            prompt_content += audit_feedback
            
        sys_msg = SystemMessage(content=prompt_content)
        
    elif phase == "prescription":
        # Extract diagnosis info from state if available
        diag_info = state.get("diagnosis_data", {})
        diagnosis_str = diag_info.get("diagnosis", "Unknown")
        conf_str = diag_info.get("confidence", "Unknown")
        
        tools = [tool_lookup_guideline, tool_check_drug_interaction, tool_submit_prescription]
        
        prompt_content = PRESCRIPTION_PROMPT.format(
            diagnosis=diagnosis_str, 
            confidence=conf_str
        )
        if audit_feedback:
            prompt_content += audit_feedback
            
        sys_msg = SystemMessage(content=prompt_content)
    elif phase == "scribe":
        # Auto-generation phase, no tools needed usually, or just return text
        diag_str = state.get("diagnosis_data", {}).get("diagnosis")
        pres_str = str(state.get("prescription_data", {}).get("drugs"))
        
        return {
            "messages": [AIMessage(content=f"**SOAP 电子病历**\n\n**Diagnosis**: {diag_str}\n**Plan**: {pres_str}\n\n(Session Ended)")],
            "phase": "end"
        }
    else:
        return {"phase": "end"}

    # Bind tools
    model_with_tools = model.bind_tools(tools)
    
    # Filter messages to only include recent context or full history? 
    # LangGraph passes full history in `messages`.
    # We should ensure SystemMessage is at the front. 
    # But `messages` in state might already contain User inputs.
    # We need to prepend the *Current Phase's* System Prompt to the run, but not save it to history forever if we switch phases?
    # Actually, appending works nicely if we view it as a continuous chat.
    # But LangChain models expect specific roles.
    
    # Construction: [System, ...History...]
    # We create a temporary message list
    input_messages = [sys_msg] + messages
    
    # DEBUG: Print message types and contents
    print(f"\n[DEBUG] Doctor Node Invoke. Phase: {phase}")
    for i, m in enumerate(input_messages):
        role = m.type
        content_preview = m.content[:50]
        tool_calls = getattr(m, 'tool_calls', [])
        print(f"  {i}. {role}: {content_preview} (ToolCalls: {len(tool_calls)})")
        if role == 'tool':
            print(f"     ToolCallID: {m.tool_call_id}")
    
    response = await model_with_tools.ainvoke(input_messages)
    return {"messages": [response]}

async def safety_audit_node(state: DoctorState):
    """
    红队审计节点 (Safety Audit Node)
    
    审计维度:
    1. 药物相互作用是否存在？
    2. 剂量是否在安全范围？
    3. 诊断结论与 symptom_vector 是否存在矛盾？
    """
    print("\n[DEBUG] Node Safety Audit Start")
    
    diagnosis_data = state.get("diagnosis_data", {})
    prescription_data = state.get("prescription_data", {})
    symptom_vector = state.get("symptom_vector", {})
    
    # 1. 药物相互作用检查 (DDInterChecker + MedicalRuleService)
    checker = DDInterChecker()
    drugs = prescription_data.get("drugs", [])
    if not drugs and "prescription_list" in prescription_data:
        drugs = prescription_data["prescription_list"]
        
    # 提取药物名称
    drug_names = []
    for d in drugs:
        if isinstance(d, dict):
            drug_names.append(d.get("name", d.get("drug_name", "")))
        else:
            drug_names.append(str(d))
            
    # Use async check to leverage non-blocking DB access and MedicalRuleService
    ddi_warnings = await checker.check_async(drug_names)
    
    # 2. 剂量与矛盾检查 (使用 LLM 进行逻辑审计)
    # 构造审计 Prompt
    audit_prompt = f"""作为资深药剂师与医疗安全审计专家，请审计以下诊断与处方：
    
    【诊断结论】: {diagnosis_data.get('diagnosis', '未提供')}
    【处方清单】: {json.dumps(drugs, ensure_ascii=False)}
    【症状向量】: {json.dumps(symptom_vector, ensure_ascii=False)}
    【已知冲突】: {json.dumps(ddi_warnings, ensure_ascii=False)}
    
    审计要求：
    1. 检查药物剂量是否超出常规安全范围。
    2. 检查诊断结论是否能解释症状向量中的核心症状。
    3. 结合已知冲突，评估整体风险等级。
    
    输出协议 (JSON):
    {{"passed": bool, "risk_level": "low/med/high", "reason": "若不通过，指明具体风险"}}
    """
    
    model = get_model()
    # 强制要求 JSON 输出
    response = await model.ainvoke([
        SystemMessage(content="你是一个医疗安全审计专家，必须输出 JSON 格式。"),
        HumanMessage(content=audit_prompt)
    ])
    
    try:
        # 尝试解析 JSON
        import re
        content = response.content
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            audit_result = json.loads(json_match.group())
        else:
            # Fallback if no JSON found
            audit_result = {"passed": True, "risk_level": "low", "reason": "Audit failed to parse output, assuming safe."}
    except Exception as e:
        print(f"[ERROR] Audit parsing failed: {e}")
        audit_result = {"passed": False, "risk_level": "high", "reason": f"审计系统故障: {str(e)}"}

    # 记录日志
    if not audit_result.get("passed", False):
        print(f"[DEBUG] Safety Audit Node: Risk Detected - {audit_result.get('reason')}")
    
    return {"audit_result": audit_result}

def state_updater_node(state: DoctorState):
    """
    状态更新节点 (State Updater Node)
    
    解析工具输出并更新状态字段。
    """
    messages = state["messages"]
    last_msg = messages[-1]
    
    updates = {}
    
    # 初始化审计重试次数
    if "audit_retry_count" not in state:
        updates["audit_retry_count"] = 0
        
    # 如果刚从审计节点回来，且未通过，增加重试次数
    audit_result = state.get("audit_result", {})
    if audit_result and not audit_result.get("passed", True):
        updates["audit_retry_count"] = state.get("audit_retry_count", 0) + 1

    # 提取症状向量 (简单逻辑：从首条 HumanMessage 提取，或者根据特定标志)
    if "symptom_vector" not in state or not state["symptom_vector"]:
        for m in messages:
            if isinstance(m, HumanMessage):
                # 这里可以接入一个专门的 NLP 工具来提取症状向量
                # 暂时使用 Mock 或简单的解析
                updates["symptom_vector"] = {"raw_input": m.content}
                break

    if isinstance(last_msg, ToolMessage):
        try:
            # 查找匹配的 ToolCall 以获取函数名
            # LangGraph 会按顺序保存 ToolMessage
            # 我们需要找到对应的 ToolCall
            
            # 简单的逻辑：如果是 submit_diagnosis，更新 phase
            if last_msg.name == "tool_submit_diagnosis":
                data = json.loads(last_msg.content)
                updates["diagnosis_data"] = data
                updates["phase"] = "prescription"
            
            elif last_msg.name == "tool_submit_prescription":
                data = json.loads(last_msg.content)
                updates["prescription_data"] = data
                updates["phase"] = "end"
                
        except Exception as e:
            print(f"[ERROR] State updater failed: {e}")
            
    return updates

# ==================== Graph Construction ====================

# 1. 定义图 (Initialize Graph)
workflow = StateGraph(DoctorState)

# 2. 添加节点 (Add Nodes)
workflow.add_node("diagnosis_node", diagnosis_node)
workflow.add_node("safety_audit_node", safety_audit_node)
workflow.add_node("tools", ToolNode([
    tool_lookup_guideline, 
    tool_check_drug_interaction, 
    tool_query_ehr, 
    tool_query_lab,
    tool_submit_diagnosis,
    tool_submit_prescription
]))
workflow.add_node("state_updater", state_updater_node)

# 3. 设置入口 (Set Entry Point)
workflow.set_entry_point("diagnosis_node")

# 4. 路由逻辑 (Conditional Edges)
def router(state: DoctorState):
    """
    根据 AI 的输出决定下一步: 
    - 如果有工具调用 -> tools
    - 如果没有工具调用 -> state_updater (纯文本回复也要经过更新器吗？或者直接 END)
      这里我们选择先去 state_updater 提取可能的信息，然后在 updater 后的路由决定是否 END。
    """
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return "state_updater"

def audit_router(state: DoctorState):
    """
    红队审计路由逻辑:
    - passed 为 True -> END
    - passed 为 False 且 risk_level 为 high 且 重试次数 < 2 -> diagnosis_node
    - 否则 -> CRITICAL_ERROR (结束或人工介入)
    """
    audit_result = state.get("audit_result", {})
    retry_count = state.get("audit_retry_count", 0)
    
    if audit_result.get("passed", False):
        return "end"
    
    if audit_result.get("risk_level") == "high" and retry_count < 2:
        return "retry"
    
    return "critical_error"

def after_state_updater_router(state: DoctorState):
    """
    状态更新后的路由:
    1. 如果 phase 变成了 "end" (说明刚完成了处方提交) -> safety_audit_node
    2. 如果最后一条消息是 ToolMessage (说明刚执行完工具) -> diagnosis_node (让 LLM 继续思考)
    3. 如果最后一条消息是 AIMessage 且无工具调用 (说明是纯文本回复) -> END (等待用户输入)
    """
    messages = state["messages"]
    last_msg = messages[-1]

    if state.get("phase") == "end":
        return "safety_audit_node"
    
    if isinstance(last_msg, ToolMessage):
        return "diagnosis_node"
    
    # 如果是 AIMessage 且没有 tool_calls (已经被 router 过滤过一次，但为了保险)
    if isinstance(last_msg, AIMessage) and not last_msg.tool_calls:
        return "end"
        
    return "diagnosis_node"

workflow.add_conditional_edges(
    "diagnosis_node",
    router,
    {"tools": "tools", "state_updater": "state_updater"}
)

# 修正: 工具执行后，先去 state_updater 更新状态，而不是直接回 diagnosis_node
workflow.add_edge("tools", "state_updater")

workflow.add_conditional_edges(
    "state_updater",
    after_state_updater_router,
    {
        "safety_audit_node": "safety_audit_node", 
        "diagnosis_node": "diagnosis_node",
        "end": END
    }
)

workflow.add_conditional_edges(
    "safety_audit_node",
    audit_router,
    {
        "end": END,
        "retry": "diagnosis_node",
        "critical_error": END # 这里可以直接 END，或者跳转到一个错误节点
    }
)

# Checkpointer
checkpointer = MemorySaver()

doctor_graph = workflow.compile(checkpointer=checkpointer)
