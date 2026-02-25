from typing import Dict, Any

def get_summarization_prompt(conversation_text: str) -> str:
    """
    生成用于总结医疗对话历史的 Prompt (中文版)。
    
    Args:
        conversation_text: 对话历史文本
        
    Returns:
        Prompt 字符串
    """
    return f"""对话历史:
{conversation_text}

任务：将上述对话中的关键医疗信息总结为一个 JSON 对象。
请重点关注：
1. 确诊的症状 (患者明确表示的不适)。
2. 症状的持续时间或频率。
3. 提及的病史 (既往疾病、正在服用的药物、过敏史)。
4. 医生的关键建议或问题。

请严格输出合法的 JSON 格式。"""

def get_summarization_system_prompt() -> str:
    """
    生成总结任务的系统提示词 (中文版)。
    """
    return """你是一个医疗数据结构化专家。请将对话总结为 JSON 格式。
输出字段 (Keys):
- symptoms: 患者的主要主诉/症状
- duration: 症状持续时间
- history: 既往病史/用药史
- advice: 医生的关键建议

确保 JSON 格式合法。不要输出 Markdown，不要有任何解释文字。"""
