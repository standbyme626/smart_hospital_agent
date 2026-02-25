import re
import structlog
from typing import Dict, Any
from langchain_core.runnables.config import RunnableConfig
from app.core.graph.state import AgentState

logger = structlog.get_logger(__name__)

# Simple Regex Patterns for Chinese PII
PHONE_PATTERN = r'(1[3-9]\d{9})'
ID_CARD_PATTERN = r'([1-9]\d{5}[1-9]\d{3}((0\d)|(1[0-2]))(([0|1|2]\d)|3[0-1])\d{3}([0-9Xx]))'
NAME_PATTERN = r'(?<=我是)([\u4e00-\u9fa5]{2,4})(?=。|，|！)' # Very naive, catches "我是张三"

async def pii_filter_node(state: AgentState, config: RunnableConfig = None) -> Dict[str, Any]:
    """
    [Pain Point #26] PII Leakage Prevention
    Filters out sensitive information (Phone, ID, Name) from user input before it hits LLMs.
    """
    symptoms = state.get("symptoms", "")
    history = state.get("medical_history", "")
    
    original_symptoms = symptoms
    original_history = history
    
    # Redact ID Card First (Longer and more specific, to avoid Phone pattern matching inside ID)
    symptoms = re.sub(ID_CARD_PATTERN, "[ID_REDACTED]", symptoms)
    history = re.sub(ID_CARD_PATTERN, "[ID_REDACTED]", history)
    
    # Redact Phone
    symptoms = re.sub(PHONE_PATTERN, "[PHONE_REDACTED]", symptoms)
    history = re.sub(PHONE_PATTERN, "[PHONE_REDACTED]", history)
    
    # Redact Name (Naive)
    # symptoms = re.sub(NAME_PATTERN, "**", symptoms) 
    
    if symptoms != original_symptoms or history != original_history:
        logger.info("pii_redacted", original_len=len(original_symptoms), new_len=len(symptoms))
        
    return {
        "symptoms": symptoms,
        "medical_history": history
    }
