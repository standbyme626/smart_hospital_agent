import logging
from typing import Dict, Any, List
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from app.core.graph.state import AgentState
from app.core.llm.llm_factory import get_fast_llm
from app.core.monitoring.tracing import monitor_node

logger = logging.getLogger(__name__)

@monitor_node("multimodal_processor")
async def multimodal_processor_node(state: AgentState, config: RunnableConfig = None) -> Dict[str, Any]:
    """
    [Pain Point #27] Multimodal Ingress Node
    Analyzes images (Lab results, X-rays, Photos) and converts them to text description.
    """
    images = state.get("images", [])
    user_input = state.get("user_input", "")
    
    if not images:
        return {}

    logger.info("processing_images", count=len(images))
    
    # Force use of a vision-capable model
    # We use prefer_local=False to ensure we hit the cloud API which supports vision
    llm = get_fast_llm(temperature=0.1, prefer_local=False) 
    
    content_parts = [{"type": "text", "text": "请分析这些医疗图片（如化验单、检查报告或患处照片）。提取关键的异常指标、诊断结论或可见症状。不要遗漏任何数值异常。"}]
    
    for img_url in images:
        # Simple heuristic: ensure format is correct for LangChain/OpenAI
        if isinstance(img_url, str):
            content_parts.append({"type": "image_url", "image_url": {"url": img_url}})
             
    message = HumanMessage(content=content_parts)
    
    try:
        response = await llm.ainvoke([message])
        analysis_text = response.content
        
        # Augment the user input
        augmented_input = f"{user_input}\n\n【图片分析结果】\n{analysis_text}"
        
        logger.info("image_analysis_success")
        return {
            "user_input": augmented_input,
            # Also append to symptoms so downstream nodes see it clearly
            "symptoms": f"{state.get('symptoms', '')}\n\n[Visual Findings]: {analysis_text}"
        }
    except Exception as e:
        logger.error("image_analysis_failed", error=str(e))
        return {}
