import json
from app.core.llm.llm_factory import get_fast_llm
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import AIMessage

from app.core.config import settings
from app.domain.states.sub_states import TriageState
from app.core.prompts.triage import triage_prompt

def build_triage_graph():
    """
    构建分诊子图 (Triage Subgraph)
    负责: Intent Classification -> Extraction -> Initial Routing
    """
    workflow = StateGraph(TriageState)
    
    # 使用快速模型 (Flash)
    llm = get_fast_llm(temperature=0.0)
    
    async def triage_node(state: TriageState):
        chain = triage_prompt | llm
        response = await chain.ainvoke({"messages": state["messages"]})
        
        content = response.content.strip()
        # Robust JSON parsing
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].strip()

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            print(f"[WARN] Triage JSON Parse Error. Content: {content[:100]}...")
            # Fallback
            data = {
                "intent": "SYMPTOM", 
                "symptoms": state.get("user_input", ""), 
                "reply": "收到，请详细描述您的症状。",
                "reason": "Parse Error"
            }
            
        return {
            "intent": data.get("intent", "SYMPTOM"),
            "symptoms": data.get("symptoms"),
            "messages": [AIMessage(content=data.get("reply", ""))]
        }

    workflow.add_node("triage", triage_node)
    workflow.add_edge(START, "triage")
    workflow.add_edge("triage", END)
    
    return workflow.compile()
