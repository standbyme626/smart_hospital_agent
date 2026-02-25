import json
import re
import logging
from typing import Dict, Any, Optional

# 配置日志
logger = logging.getLogger(__name__)

def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    从模型输出的文本中鲁棒地提取 JSON 对象。
    
    功能特点:
    1. 自动过滤 <think>...</think> 思考过程。
    2. 识别 Markdown 代码块 (```json ... ```)。
    3. 在全文中搜索最外层的花括号 {} 结构。
    4. 处理常见的 JSON 格式错误 (如中文引号、多余逗号等 - 待扩展)。
    
    Args:
        text (str): 模型生成的原始文本。
        
    Returns:
        Optional[Dict[str, Any]]: 解析成功的字典，失败则返回 None。
    """
    if not text:
        return None

    # 1. 移除 <think> 标签内容 (针对 DeepSeek/Qwen-Thinking 模式)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    
    # 2. 尝试提取 Markdown 代码块
    markdown_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if markdown_match:
        json_str = markdown_match.group(1)
    else:
        # 3. 尝试搜索第一个 { 和最后一个 }
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = text[start_idx : end_idx + 1]
        else:
            logger.warning("未能在文本中找到 JSON 结构")
            return None
            
    # 4. 清理与尝试解析
    try:
        # 简单清理：替换常见的错误引号 (中文引号 -> 英文引号)
        # 注意：这可能会误伤内容中的中文引号，需谨慎使用。
        # 这里仅做基础的控制字符清理
        json_str = json_str.strip()
        
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error(f"JSON 解析失败: {e} | 原始内容片段: {json_str[:50]}...")
        return None
