import json
import structlog
from typing import Dict, Any
from datetime import datetime
try:
    from langchain_core.prompts import PromptTemplate
except ImportError:
    from langchain.prompts import PromptTemplate
from app.core.config import settings
from app.core.llm.llm_factory import get_fast_llm

logger = structlog.get_logger()

METADATA_PROMPT = """
你是一个医学数据结构化助手。请从以下文本中提取关键医学元数据。
如果文本中不包含某项信息，请留空。

文本内容：
{text}

请严格按 JSON 格式返回，包含以下字段：
- disease_name: 疾病名称 (如 "高血压")
- department: 建议挂号科室 (如 "心内科")
- icd10_code: ICD-10编码 (如果能推断出)
- keywords: 关键词列表 (最多5个)

JSON输出：
"""

class MetadataExtractor:
    """医学元数据提取器"""

    def __init__(self):
        # [V6.5.1] 统一使用自愈型 LLM，解决鉴权与 400 错误
        self.llm = get_fast_llm()
        self.prompt = PromptTemplate(template=METADATA_PROMPT, input_variables=["text"])
        logger.info("metadata_extractor.initialized")

    def extract(self, text: str) -> Dict[str, Any]:
        """提取元数据"""
        try:
            # 构造 Chain
            chain = self.prompt | self.llm
            response = chain.invoke({"text": text})
            content = response.content.strip()
            
            # 清理 Markdown 代码块格式
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            
            data = json.loads(content)
            logger.debug("metadata.extracted", data=data)
            return data
        except Exception as e:
            logger.error("metadata.extract_failed", error=str(e))
            return {
                "disease_name": None,
                "department": None, 
                "icd10_code": None,
                "keywords": []
            }
