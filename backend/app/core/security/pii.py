
import re
from typing import List, Any

class PIIMasker:
    """
    PII (Personally Identifiable Information) Masker
    用于在发送给云端 LLM 之前脱敏敏感数据。
    """
    
    # 简单的正则规则
    PATTERNS = {
        "PHONE": r"(13[0-9]|14[01456879]|15[0-35-9]|16[2567]|17[0-8]|18[0-9]|19[0-35-9])\d{8}",
        "ID_CARD": r"\b\d{17}[\dXx]\b",
        "EMAIL": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
        # 简单的中文姓名模糊匹配 (2-4个字，且不包含常见非人名字符)
        # 注意：这只是一个非常粗糙的 heuristic，生产环境应使用 NLP 模型 (如 MS Presidio)
        # "NAME_CN": r"(?<![a-zA-Z0-9])[\u4e00-\u9fa5]{2,4}(?![a-zA-Z0-9])" 
        # 暂时不启用姓名自动脱敏，因为容易误伤医疗名词
    }

    @classmethod
    def mask(cls, text: str) -> str:
        if not text:
            return text
            
        masked_text = text
        
        # Phone
        masked_text = re.sub(cls.PATTERNS["PHONE"], "<PHONE>", masked_text)
        
        # ID Card
        masked_text = re.sub(cls.PATTERNS["ID_CARD"], "<ID_CARD>", masked_text)
        
        # Email
        masked_text = re.sub(cls.PATTERNS["EMAIL"], "<EMAIL>", masked_text)
        
        return masked_text

    @classmethod
    def mask_messages(cls, messages: List[Any]) -> List[Any]:
        """
        Mask PII in a list of LangChain messages.
        """
        from langchain_core.messages import BaseMessage
        
        masked_messages = []
        for msg in messages:
            if isinstance(msg, BaseMessage):
                # Keep original message structure/required fields (e.g., ToolMessage.tool_call_id).
                if isinstance(msg.content, str):
                    new_content = cls.mask(msg.content)
                else:
                    new_content = msg.content

                if hasattr(msg, "model_copy"):
                    new_msg = msg.model_copy(update={"content": new_content})
                else:
                    # Fallback for older message classes.
                    new_msg = msg
                masked_messages.append(new_msg)
            else:
                masked_messages.append(msg)
        return masked_messages
