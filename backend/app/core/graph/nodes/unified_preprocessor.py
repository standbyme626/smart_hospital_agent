import json
from typing import Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage

from app.core.graph.state import AgentState
from app.core.llm.llm_factory import get_fast_llm
from app.core.utils.json_parser import extract_json_from_text as parse_json_markdown
from app.core.prompts.ingress import INGRESS_PROMPT

class UnifiedPreprocessor:
    """
    统一预处理节点 (Unified Preprocessor)
    合并 Intent Recognition 与 Persona Extraction，减少推理次数。
    """
    
    async def run(self, state: AgentState) -> Dict[str, Any]:
        # [Pain Point #33] 优先从 messages 获取最新输入，而非依赖可能为空的 symptoms 字段
        messages = state.get("messages", [])
        if messages and hasattr(messages[-1], 'content'):
            user_input = messages[-1].content
        else:
            user_input = state.get("user_input") or state.get("symptoms", "")

        # Get history (simplified for now)
        history = messages
        
        llm = get_fast_llm(temperature=0.0, prefer_local=True)
        
        # [Pain Point #29] Inject Medical History Summary
        summary = state.get("medical_record_summary", "")
        
        prompt = INGRESS_PROMPT.format(
            user_input=user_input,
            medical_record_summary=summary
        )
        
        input_messages = [
            SystemMessage(content="你是一个严格遵循 JSON 格式的医疗预处理系统。"),
            HumanMessage(content=prompt)
        ]
        
        try:
            response = await llm.ainvoke(input_messages)
            result = parse_json_markdown(response.content)
            
            # Extract fields
            intent = result.get("intent", "GENERAL_CONSULTATION")
            persona = result.get("persona", {})
            
            # [Pain Point #22] Temporal Consistency
            import time
            current_time = time.time()
            if persona:
                persona["_last_updated"] = current_time
                
            risk_level = result.get("risk_level", "low")
            
            # [Pain Point #33] Construct EventContext
            event_context = {
                "event_type": "SYMPTOM_DESCRIPTION" if intent != "GREETING" else "GREETING",
                "payload": {
                    "symptoms": user_input, # 保留原始描述
                    "structured_symptoms": result.get("symptoms_list", []),
                    "intent": intent,
                    "risk_level": risk_level
                },
                "raw_input": user_input,
                "timestamp": current_time
            }
            
            return {
                "intent": intent,
                "patient_persona": persona,
                "risk_level": risk_level,
                "event": event_context, # [New] Return EventContext
                "symptoms": user_input # [Legacy] Keep for compatibility until full migration
            }
            
        except Exception as e:
            # Fallback
            import time
            fallback_event = {
                "event_type": "UNKNOWN",
                "payload": {"error": str(e)},
                "raw_input": user_input,
                "timestamp": time.time()
            }
            return {
                "intent": "GENERAL_CONSULTATION", 
                "patient_persona": {},
                "risk_level": "low",
                "event": fallback_event,
                "symptoms": user_input
            }

unified_preprocessor = UnifiedPreprocessor()

async def unified_preprocessor_node(state: AgentState):
    return await unified_preprocessor.run(state)
