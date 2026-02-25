
from typing import Literal
from app.core.llm.llm_factory import get_smart_llm
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage, SystemMessage

from app.core.config import settings
from app.domain.states.sub_states import PrescriptionState
from app.core.tool_registry import registry
from app.tools.prescription_tools import draft_prescription
from app.core.graph.nodes.audit_node import SafetyAuditNode

# Updated Prompt for Prescription
PRESCRIPTION_SYSTEM_PROMPT = """你是由 Smart Hospital 部署的智能药师 Agent。
你的职责是根据确诊结果为患者开具处方。

### 上下文
- 患者 ID: {patient_id}
- 确诊结果: {diagnosis}
- 当前阶段: {step}

### 流程指南
1. **草拟处方 (Drafting)**:
   - 分析病情，选择药物。
   - 使用 `check_drug_interaction` 检查相互作用。
   - 确定无误后，使用 `draft_prescription` 工具提交草稿以供审核。
   - **不要**直接调用 `submit_prescription`。

2. **审核反馈 (Auditing)**:
   - 如果收到审核驳回 (AUDIT REJECTED)，请根据反馈修改处方，并再次调用 `draft_prescription`。

3. **提交处方 (Finalizing)**:
   - 只有当审核通过 (Audit Approved) 且当前阶段为 "finalizing" 时，才使用 `submit_prescription` 正式提交。
   - 提交后，结束对话。

### 审计反馈
{audit_feedback}

### 安全规则
- 严禁为孕妇开具禁用药物。
- 处方必须清晰、准确。
"""

def build_prescription_graph():
    """
    构建处方子图 (Prescription Subgraph)
    负责: Draft -> DrugCheck -> Audit -> Finalize
    """
    workflow = StateGraph(PrescriptionState)
    
    # 1. 获取工具
    # 基础工具
    base_tools = registry.get_subset(["check_drug_interaction", "submit_prescription"])
    # 添加草拟工具
    all_tools = base_tools + [draft_prescription]
    
    # 2. 初始化模型
    llm = get_smart_llm(temperature=0.0)
    llm_with_tools = llm.bind_tools(all_tools)
    
    # 3. 定义节点
    
    async def pharmacist_node(state: PrescriptionState):
        """药师节点：起草和修改处方"""
        print("[DEBUG] Node Pharmacist Start")
        patient_id = state.get("patient_id", "UNKNOWN")
        diagnosis = state.get("confirmed_diagnosis", "UNKNOWN")
        step = state.get("step", "drafting")
        audit_feedback = state.get("audit_feedback", "无")
        
        system_msg = SystemMessage(content=PRESCRIPTION_SYSTEM_PROMPT.format(
            patient_id=patient_id,
            diagnosis=diagnosis,
            step=step,
            audit_feedback=audit_feedback
        ))
        
        # 拼接历史消息
        # 过滤掉之前的 system message，只保留最近的? 
        # 简单起见，LangGraph 传递的 messages 包含所有历史。
        # 我们prepend新的 system message。
        messages = [system_msg] + state["messages"]
        
        response = await llm_with_tools.ainvoke(messages)
        return {"messages": [response]}

    # 实例化 Audit Node
    audit_node = SafetyAuditNode()

    workflow.add_node("pharmacist", pharmacist_node)
    workflow.add_node("tools", ToolNode(all_tools))
    workflow.add_node("audit", audit_node)
    
    # 4. 定义边
    workflow.add_edge(START, "pharmacist")
    
    def route_pharmacist(state: PrescriptionState) -> Literal["tools", END]:
        last_message = state["messages"][-1]
        
        if last_message.tool_calls:
            return "tools"
        
        # 如果没有工具调用
        step = state.get("step")
        if step == "finalizing":
            # 如果在 finalizing 阶段且没有调用 submit_prescription (即没有 tool_calls)，
            # 可能是药师在说话。如果它说完了，就结束?
            # 最好是强制它调用 submit。
            return END
            
        return END

    def route_tools(state: PrescriptionState) -> Literal["audit", "pharmacist"]:
        last_message = state["messages"][-1]
        # ToolNode 执行完后，messages 最后一条是 ToolMessage。
        # 我们需要检查倒数第二条 (AIMessage) 调用的工具是什么。
        # 或者检查 ToolMessage 的 name (langchain ToolMessage has name? artifact?)
        
        # 更可靠的方法：检查最近一次 AIMessage 的 tool_calls
        messages = state["messages"]
        # Find the last AIMessage
        last_ai_msg = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
        
        if last_ai_msg and last_ai_msg.tool_calls:
            tool_name = last_ai_msg.tool_calls[0]["name"]
            if tool_name == "draft_prescription":
                return "audit"
            if tool_name == "submit_prescription":
                return "pharmacist" # allow pharmacist to see result and close? or END?
                
        return "pharmacist"

    workflow.add_conditional_edges(
        "pharmacist",
        route_pharmacist,
        {
            "tools": "tools",
            END: END
        }
    )
    
    workflow.add_conditional_edges(
        "tools",
        route_tools,
        {
            "audit": "audit",
            "pharmacist": "pharmacist"
        }
    )
    
    workflow.add_edge("audit", "pharmacist")
    
    return workflow.compile()
