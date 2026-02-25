import re
import os
import structlog
from app.core.models.local_slm import local_slm

logger = structlog.get_logger(__name__)

class PIIService:
    """
    PII (Personally Identifiable Information) Masking Service.
    Hybrid approach: Regex (Speed) + Local SLM (Context).
    """
    
    # Common PII Patterns (China context)
    PATTERNS = {
        "PHONE": r"1[3-9]\d{9}",
        "ID_CARD": r"\d{17}[\dXx]",
        "EMAIL": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
    }

    def scrub(self, text: str) -> str:
        """
        Scrub PII from text before sending to Cloud LLM.
        """
        if not text:
            return ""
            
        # [V6.2 Fix] Pre-clean text to remove thinking tags before PII masking
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        text = text.replace("<think>", "").replace("</think>", "").strip()
        
        scrubbed_text = text
        
        # 1. Regex Fast Pass
        for label, pattern in self.PATTERNS.items():
            scrubbed_text = re.sub(pattern, f"[{label}]", scrubbed_text)
            
        # 2. Local SLM Contextual Pass (NER)
        # Only run if text length > 10 to avoid overhead on tiny inputs
        if len(scrubbed_text) > 10:
            scrubbed_text = self._scrub_with_slm(scrubbed_text)
            
        return scrubbed_text
        
    def _scrub_with_slm(self, text: str) -> str:
        """
        Use Cloud LLM (Switch from Local SLM) to identify names and locations.
        """
        prompt = f"""Identify and mask PII (Names, Addresses) in the text. Replace them with [NAME] or [ADDRESS]. Return the masked text ONLY.
        
        Text: "{text}"
        
        Masked Text:"""
        
        try:
            # [V6.2 Switch] Use Cloud LLM instead of local_slm for stability
            from app.core.llm.llm_factory import get_fast_llm
            import asyncio
            
            allow_fallback = os.getenv("ENABLE_LOCAL_FALLBACK", "false").lower() == "true"
            llm = get_fast_llm(temperature=0.0, allow_local=allow_fallback)
            # Use synchronous invoke for simplicity as scrub is currently sync
            # Note: In production, scrub should be async.
            response = llm.invoke(prompt)
            masked = response.content.strip()
            
            if not masked:
                return text
                
            # Basic validation: Only fallback if it's EXTREMELY short (likely an error)
            # Desensitized text is usually shorter, so 0.3 is a safer threshold
            if len(masked) < len(text) * 0.3:
                logger.warning("pii_cloud_failure_fallback", reason="length_too_short", original_len=len(text), masked_len=len(masked))
                return text 
                
            return masked
        except Exception as e:
            logger.error("pii_cloud_error", error=str(e))
            return text
