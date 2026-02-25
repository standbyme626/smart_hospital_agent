import json
import structlog
from typing import Dict, Any
import os
from langchain_core.messages import HumanMessage
from app.core.graph.state import AgentState
# from app.core.models.local_slm import LocalSLMService
from app.core.llm.llm_factory import get_fast_llm
from app.core.monitoring.tracing import monitor_node
from app.core.prompts.persona import get_persona_extraction_prompt

logger = structlog.get_logger(__name__)

class PersonaExtractor:
    """
    对话画像提取器 (Dialogue Persona Extractor)
    
    [Phase 6] 核心组件：
    从非结构化的用户输入中提取关键患者画像信息（年龄、性别、既往史、当前用药）。
    """
    
    def __init__(self):
        # self.slm = LocalSLMService()
        pass
        
    async def run(self, state: AgentState) -> Dict[str, Any]:
        """提取画像逻辑"""
        logger.info("node_start", node="persona_extractor")
        
        # [Pain Point #33] Use EventContext
        event = state.get("event", {})
        if event and "raw_input" in event:
            user_input = event["raw_input"]
        else:
            user_input = state.get("symptoms", "")
            
        existing_persona = state.get("patient_persona") or {}
        
        # 构造提取 Prompt
        prompt_text = get_persona_extraction_prompt(
            user_input=user_input,
            existing_persona_json=json.dumps(existing_persona, ensure_ascii=False)
        )
        
        try:
            # 使用云端 LLM 进行提取 (Force Cloud)
            allow_fallback = os.getenv("ENABLE_LOCAL_FALLBACK", "false").lower() == "true"
            llm = get_fast_llm(allow_local=allow_fallback)
            response = await llm.ainvoke([HumanMessage(content=prompt_text)])
            raw_output = response.content
            
            # 尝试解析 JSON
            import re
            import time
            json_match = re.search(r'\{.*\}', raw_output, re.DOTALL)
            if json_match:
                new_data = json.loads(json_match.group())
                
                # [Pain Point #22] Temporal Consistency - 引入时间戳
                # 不再直接覆盖，而是记录更新时间，便于后续冲突解决
                updated_persona = existing_persona.copy()
                current_time = time.time()
                
                for key, value in new_data.items():
                    if value and value != "null":
                        # 如果字段已存在，检查是否需要合并或更新
                        if key in updated_persona:
                            # 如果是列表类型（如症状、药物），进行去重合并
                            if isinstance(value, list) and isinstance(updated_persona[key], list):
                                updated_persona[key] = list(set(updated_persona[key] + value))
                            # 如果是单值（如年龄、性别），采用"最新优先"原则，但这里我们先简单覆盖
                            # 理想情况下应对比 version 或 timestamp，但 SLM 输出通常没有
                            else:
                                updated_persona[key] = value
                        else:
                            updated_persona[key] = value
                            
                    # 为每个字段（或整体画像）打上时间戳
                    # 这里我们简单更新整体的时间戳元数据
                    updated_persona["_last_updated"] = current_time
                
                logger.info("persona_extracted", persona=updated_persona)
                return {"patient_persona": updated_persona}
            
        except Exception as e:
            logger.error("persona_extraction_failed", error=str(e))
            
        return {}

persona_extractor = PersonaExtractor()

@monitor_node("persona_extractor")
async def persona_extractor_node(state: AgentState):
    return await persona_extractor.run(state)
