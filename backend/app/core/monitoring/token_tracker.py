from typing import Dict, Any, List, Optional
import time

def format_token_usage(node_name: str, usage: Dict[str, Any], model_name: str, context_len: int = 0) -> Dict[str, Any]:
    """
    格式化 Token 使用统计
    
    Args:
        node_name: 节点名称 (e.g., 'classifier', 'medical_crew')
        usage: 原始 usage 字典 (e.g., {'input_tokens': 10, 'output_tokens': 20})
        model_name: 模型名称
        context_len: RAG 上下文长度 (字符数)，用于分析
        
    Returns:
        Dict: 标准化的遥测数据结构
    """
    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0))
    total_tokens = usage.get("total_tokens", input_tokens + output_tokens)
    
    return {
        "node": node_name,
        "model": model_name,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "rag_context_len": context_len,
        "timestamp": time.time()
    }
