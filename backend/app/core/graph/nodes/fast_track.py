from typing import Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from app.core.graph.state import AgentState
from app.core.llm.llm_factory import get_fast_llm

async def fast_track_node(state: AgentState, config: RunnableConfig):
    """
    Fast Track Node: Handles simple queries directly with the local SLM.
    Supports streaming via config propagation.
    """
    # [Pain Point #33] Support EventContext
    event = state.get("event", {})
    if event and "payload" in event:
        user_input = event.get("raw_input") or event["payload"].get("symptoms", "")
    else:
        user_input = state.get("user_input") or state.get("symptoms", "")
    
    # Use Cloud LLM (Preferred) - Local Disabled
    llm = get_fast_llm(temperature=0.3, prefer_local=False, allow_local=False)
    
    messages = [
        SystemMessage(content="你是一名专业的医疗分诊助手。用户咨询可能比较简单，请直接给出简短、专业的回答。"),
        HumanMessage(content=user_input)
    ]
    
    try:
        # [Critical] Pass config to enable on_chat_model_stream events
        response = await llm.ainvoke(messages, config=config)
        
        return {
            "messages": [response],
            "next_step": "END"
        }
    except Exception as e:
        return {
            "messages": [HumanMessage(content="正在为您连接专家组，请稍候...")],
            "next_step": "standard_consultant"
        }
