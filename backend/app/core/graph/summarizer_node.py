import time
import json
import structlog
from typing import Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
import os

from app.core.graph.state import AgentState
# from app.core.models.local_slm import LocalSLMService
from app.core.llm.llm_factory import get_fast_llm
from app.core.prompts.summarizer import get_summarization_prompt, get_summarization_system_prompt
from app.core.utils.json_parser import extract_json_from_text
from app.core.monitoring.tracing import monitor_node

logger = structlog.get_logger(__name__)

@monitor_node("summarize_history")
async def summarize_history_node(state: AgentState, config: RunnableConfig = None) -> Dict[str, Any]:
    """
    节点：历史消息摘要 (History Summarization)
    
    功能：
    当对话历史过长时，调用 SLM 对早期消息进行结构化总结，以减少 Token 消耗。
    总结内容将作为 SystemMessage 插入到历史记录头部。
    
    输入: state.messages
    输出: 更新后的 messages 列表 (Summary + Recent Messages)
    """
    print(f"[DEBUG] Node summarize_history Start")
    logger.info("node_start", node="summarize_history")
    start_time = time.time()
    
    current_messages = state.get("messages", [])
    
    # 1. 阈值检查：如果消息少于 8 条，跳过压缩
    if len(current_messages) <= 8:
        return {}
        
    # 2. Token 估算与跳过逻辑 (1 token ≈ 4 chars)
    total_chars = sum(len(m.content) for m in current_messages)
    estimated_tokens = total_chars / 4
    if estimated_tokens < 500:
        logger.info("summarize_skipped", reason="low_token_count", tokens=estimated_tokens)
        return {}
        
    # 3. 提取需要压缩的消息 (保留最近 2 条)
    to_summarize = current_messages[:-2]
    recent_messages = current_messages[-2:]
    
    # 4. 格式化对话文本
    conversation_text = ""
    for msg in to_summarize:
        role = "患者" if isinstance(msg, HumanMessage) else "医生"
        conversation_text += f"{role}: {msg.content}\n"
        
    # 5. 调用模型生成摘要 (Force Cloud)
    # local_slm = LocalSLMService()
    
    try:
        # 获取中文提示词
        prompt_text = get_summarization_prompt(conversation_text)
        system_prompt_text = get_summarization_system_prompt()
        
        messages = [
            SystemMessage(content=system_prompt_text),
            HumanMessage(content=prompt_text)
        ]
        
        # 异步调用模型
        allow_fallback = os.getenv("ENABLE_LOCAL_FALLBACK", "false").lower() == "true"
        llm = get_fast_llm(allow_local=allow_fallback) 
        response = await llm.ainvoke(messages)
        response_text = response.content
        
        # 使用鲁棒的 JSON 解析器
        data = extract_json_from_text(response_text)
        
        if data:
            # 格式化为紧凑字符串
            summary_content = (
                f"历史摘要 | 症状: {data.get('symptoms', '未知')}; "
                f"时长: {data.get('duration', '未知')}; "
                f"病史: {data.get('history', '无')}; "
                f"建议: {data.get('advice', '无')}"
            )
        else:
            # 解析失败时的回退处理
            logger.warning("summary_parse_failed", raw_output=response_text[:100])
            summary_content = f"历史摘要 (文本): {response_text[:200]}..."
            
    except Exception as e:
        logger.error("local_slm_failed", error=str(e))
        # 遇到错误时不中断流程，仅记录日志并跳过总结
        return {}
    
    # 6. 构建新的消息列表：[Summary] + [Recent]
    # 使用 SystemMessage 存储摘要，确保模型能看到但用户界面可隐藏
    summary_msg = SystemMessage(content=summary_content)
    new_messages = [summary_msg] + recent_messages
    
    duration = time.time() - start_time
    logger.info("history_pruned", 
                original_count=len(current_messages), 
                new_count=len(new_messages), 
                duration=f"{duration:.2f}s")
    
    return {"messages": new_messages}
