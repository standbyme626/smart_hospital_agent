import re
import structlog
from typing import Any, Callable
from langchain.agents.middleware import AgentMiddleware, AgentState, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage

logger = structlog.get_logger(__name__)

class HospitalSLMMiddleware(AgentMiddleware):
    """
    Middleware for Smart Hospital SLM Agent.
    Handles:
    1. Parsing [TRIAGE] tags from model output.
    2. Fallback logic for unparseable responses.
    """
    
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Any],
    ) -> ModelResponse:
        # 1. Execute Model
        response = await handler(request)
        
        # 2. Parse Output
        if response.result and isinstance(response.result[-1], AIMessage):
            ai_msg = response.result[-1]
            content = ai_msg.content
            
            # Remove <think> tags for cleaner processing
            # [V11.2] Support DPO Model Thinking Tags
            clean_content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            
            # Log raw content for debugging
            logger.info("middleware_raw_output", content=content[:200] + "..." if len(content) > 200 else content)
            
            # [V11.2] JSON Extraction Logic
            # 优先尝试提取 JSON，因为新版模型在 JSON 模式下表现极佳
            import json
            try:
                # 提取第一个 JSON 对象
                json_match = re.search(r"(\{.*?\})", clean_content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    data = json.loads(json_str)
                    if "intent" in data:
                        # 归一化 intent
                        intent = data['intent'].upper()
                        valid_intents = {"CRISIS", "GREETING", "VAGUE_SYMPTOM", "COMPLEX_SYMPTOM"}
                        
                        # 模糊匹配修复
                        if intent not in valid_intents:
                            if "CRISIS" in intent: intent = "CRISIS"
                            elif "GREETING" in intent: intent = "GREETING"
                            elif "VAGUE" in intent: intent = "VAGUE_SYMPTOM"
                            elif "COMPLEX" in intent: intent = "COMPLEX_SYMPTOM"
                        
                        response.structured_response = {
                            "intent": intent,
                            "reason": data.get("reason", "Parsed from JSON")
                        }
                        logger.info("middleware_parsed_json", intent=intent)
                        return response
            except Exception as e:
                logger.warning("middleware_json_parse_failed", error=str(e))

            # Pattern: [TRIAGE] INTENT: Reason
            # Example: [TRIAGE] CRISIS: Patient has chest pain
            # Relaxed regex: allow space after TRIAGE, optional colon
            # Search in original content (just in case) and clean content
            match = re.search(r"\[TRIAGE\]\s*([A-Z_]+)(?::\s*(.*))?", content, re.IGNORECASE)
            
            if match:
                intent = match.group(1).upper()
                reason = match.group(2).strip() if match.group(2) else "Auto-classified by SLM"
                
                # Normalize
                valid_intents = {"CRISIS", "GREETING", "VAGUE_SYMPTOM", "COMPLEX_SYMPTOM"}
                if intent not in valid_intents:
                    logger.warning("middleware_unknown_intent", intent=intent, action="defaulting_to_VAGUE_SYMPTOM")
                    intent = "VAGUE_SYMPTOM"
                
                response.structured_response = {
                    "intent": intent,
                    "reason": reason
                }
                logger.info("middleware_parsed_intent", intent=intent, reason=reason)
            else:
                # Fallback: Keyword Analysis in Content (including thoughts)
                # This is crucial for 0.6B models that fail to output tags
                lower_content = content.lower()
                
                fallback_intent = None
                fallback_reason = "Keyword heuristic fallback"
                
                if any(k in lower_content for k in ["胸痛", "呼吸困难", "heart", "chest pain", "dying", "救命", "crisis", "emergency"]):
                    fallback_intent = "CRISIS"
                elif any(k in lower_content for k in ["你好", "hello", "hi", "greeting"]):
                    fallback_intent = "GREETING"
                
                if fallback_intent:
                    response.structured_response = {"intent": fallback_intent, "reason": fallback_reason}
                    logger.info("middleware_keyword_fallback", intent=fallback_intent)
                
                # JSON Fallback (Legacy compatibility)
                elif "{" in content and "}" in content:
                    try:
                        import json
                        # Simple extraction
                        json_str = content[content.find("{"):content.rfind("}")+1]
                        data = json.loads(json_str)
                        if "intent" in data:
                            response.structured_response = data
                            logger.info("middleware_parsed_json", intent=data.get('intent'))
                    except Exception:
                        pass
            
            # If still no structured response, default to GREETING if short, else VAGUE
            if not response.structured_response:
                if len(content) < 10:
                     response.structured_response = {"intent": "GREETING", "reason": "Short input, no tag"}
                else:
                     response.structured_response = {"intent": "VAGUE_SYMPTOM", "reason": "Failed to parse intent"}
                     
        return response
