from typing import Dict, Any, List
import uuid
from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.core.graph.state import AgentState, OrderContext
from app.services.mcp.his_server import his_tools, HISService
from app.core.llm.llm_factory import get_fast_llm
import json

# 暂时在本地定义 Prompt，后续可移至专门的 Prompt 文件
SERVICE_SYSTEM_PROMPT = """你是由“智慧医院”提供的服务 Agent。
你的目标是帮助用户完成预约挂号。
当前状态:
科室: {department}
号源: {slot_id}
订单状态: {status}

可用工具:
- get_department_slots(dept_id): 查询指定科室的可用号源。
- lock_slot(slot_id, patient_id): 锁定号源以便支付。
- confirm_appointment(order_id): 确认支付并完成预约。

交互规则:
1. 如果科室未知，请询问用户想挂哪个科（如心内科 [dept_001] 或 外科 [dept_002]）。
2. 如果已知科室但未选号源，调用 get_department_slots。
3. 如果工具返回了号源列表，你必须使用以下特殊的 UI 块格式输出：
   <ui_slots>
   [JSON LIST OF SLOTS]
   </ui_slots>
   请不要仅仅用文字列出。
4. 如果用户选定了号源（提供了 slot_id 或时间），调用 lock_slot。
5. 如果号源已锁定（待支付），请要求用户支付，并输出订单详情：
   <ui_payment>
   [JSON ORDER DETAILS]
   </ui_payment>
6. 如果用户确认支付，调用 confirm_appointment。
7. 如果确认成功，提供最终就诊指引。

输出格式:
使用自然语言（中文）回复。
如果需要调用工具，请直接发出工具调用。
当展示数据时，务必使用上述 XML 标签。
"""

class ServiceGraph:
    def __init__(self):
        self.llm = get_fast_llm(temperature=0) # Low temp for tools
        self.llm_with_tools = self.llm.bind_tools(his_tools)

    async def service_agent_node(self, state: AgentState):
        """
        Core node that decides what to do based on state.
        """
        # Extract context
        order_ctx = state.get("order_context") or OrderContext()
        messages = state.get("messages", [])
        user_profile = state.get("user_profile")
        
        # Robust access to order_context
        if isinstance(order_ctx, dict):
            department = order_ctx.get("department")
            order_id = order_ctx.get("order_id")
            payment_status = order_ctx.get("payment_status", "pending")
        else:
            department = order_ctx.department
            order_id = order_ctx.order_id
            payment_status = order_ctx.payment_status

        # Robust access to patient_id
        if hasattr(user_profile, "patient_id"):
            patient_id = user_profile.patient_id
        elif isinstance(user_profile, dict):
            patient_id = user_profile.get("patient_id")
        else:
            patient_id = "guest"
        
        # Prepare context for prompt
        prompt_ctx = {
            "department": department or "Unknown",
            "slot_id": order_id or "None", # using order_id as proxy for active slot transaction
            "status": payment_status
        }
        
        # Check if we need to inject system prompt
        # In a subgraph, we might want to just append the system prompt to the history or use it as context
        # Simple approach: Create a list of messages for the LLM
        
        # We need to detect if the user just clicked "Book" or "Pay"
        last_msg = messages[-1] if messages else None
        
        # Heuristic for "Slot Selection" from UI (e.g. "BOOK:s_001")
        if last_msg and isinstance(last_msg, HumanMessage) and last_msg.content.startswith("BOOK:"):
            slot_id = last_msg.content.split(":")[1].strip()
            # Direct tool call injection
            # We can't easily force the LLM to call a tool unless we prompt it or construct the ToolCall manually.
            # Let's prompt the LLM with this specific instruction.
            system_msg = f"User explicitly requested to book slot {slot_id}. Call lock_slot with patient_id='{patient_id}'."
            msgs = [HumanMessage(content=system_msg)]
        elif last_msg and isinstance(last_msg, HumanMessage) and last_msg.content.startswith("PAY:"):
            order_id = last_msg.content.split(":")[1].strip()
            system_msg = f"User confirmed payment for order {order_id}. Call confirm_appointment."
            msgs = [HumanMessage(content=system_msg)]
        else:
            msgs = [HumanMessage(content=SERVICE_SYSTEM_PROMPT.format(**prompt_ctx))] + messages[-5:] # Context window

        # Call LLM
        response = await self.llm_with_tools.ainvoke(msgs)
        
        return {"messages": [response]}

    async def tool_execution_node(self, state: AgentState):
        """
        Executes the tools requested by the LLM.
        """
        messages = state["messages"]
        last_message = messages[-1]
        
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return {}

        results = []
        order_updates = {}
        
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            args = tool_call["args"]
            tool_call_id = tool_call.get("id") or f"call_{uuid.uuid4().hex}"
            
            # Execute tool
            if tool_name == "get_department_slots":
                res = await HISService.get_department_slots.ainvoke(args)
                # If we got slots, we should update the order context department
                order_updates["department"] = args.get("dept_id")
                
            elif tool_name == "lock_slot":
                # Ensure patient_id is passed if missing (though LLM should pass it)
                if "patient_id" not in args:
                    user_profile = state.get("user_profile")
                    # Robust access
                    if hasattr(user_profile, "patient_id"):
                        pid = user_profile.patient_id
                    elif isinstance(user_profile, dict):
                        pid = user_profile.get("patient_id")
                    else:
                        pid = "guest"
                    args["patient_id"] = pid
                    
                res = await HISService.lock_slot.ainvoke(args)
                if res.get("status") == "success":
                    order_updates["order_id"] = res["order_id"]
                    order_updates["payment_status"] = "pending_payment"
                    
            elif tool_name == "confirm_appointment":
                res = await HISService.confirm_appointment.ainvoke(args)
                if res.get("status") == "success":
                    order_updates["payment_status"] = "paid"
            
            else:
                res = {"error": "Unknown tool"}

            results.append(ToolMessage(tool_call_id=tool_call_id, name=tool_name, content=json.dumps(res, ensure_ascii=False)))

        # Update State
        updates = {"messages": results}
        if order_updates:
            current_order = state.get("order_context") or OrderContext()
            # Ensure it's a Pydantic model
            if isinstance(current_order, dict):
                current_order = OrderContext(**current_order)
                
            # Merge updates
            new_order = current_order.model_copy(update=order_updates)
            updates["order_context"] = new_order
            
        return updates

    def should_continue(self, state: AgentState):
        messages = state["messages"]
        last_message = messages[-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"
        return END

    def build(self):
        workflow = StateGraph(AgentState)
        
        workflow.add_node("service_agent", self.service_agent_node)
        workflow.add_node("tools", self.tool_execution_node)
        
        workflow.set_entry_point("service_agent")
        
        workflow.add_conditional_edges(
            "service_agent",
            self.should_continue,
            {
                "tools": "tools",
                END: END
            }
        )
        
        workflow.add_edge("tools", "service_agent")
        
        return workflow.compile()

# Singleton for easy import
service_graph = ServiceGraph().build()
