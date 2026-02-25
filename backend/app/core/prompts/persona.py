from typing import Dict, Any

def get_persona_extraction_prompt(user_input: str, existing_persona_json: str) -> str:
    """
    生成画像提取 Prompt
    
    Args:
        user_input: 用户的原始输入
        existing_persona_json: 已有的画像 JSON 字符串
        
    Returns:
        完整的 Prompt 字符串
    """
    return f"""你是一个严谨的医疗数据记录员。你的任务是从用户描述中提取患者的客观画像信息。

    【核心原则】
    1. 必须完全基于用户描述，严禁猜测或编造。
    2. 如果用户未明确提及某项信息，必须将其设为 "UNKNOWN" 或保留 null，绝对不能自动填充默认值。
    3. 对于数值（如年龄），只有当用户明确说出数字时才提取，否则设为 null。

    用户描述: {user_input}
    当前已知画像: {existing_persona_json}
    
    请提取以下字段：
    1. age: 年龄 (数字，若未提及则为 null)
    2. gender: 性别 (男/女/UNKNOWN)
    3. chronic_diseases: 慢性病列表 (list，若无则为 [])
    4. medications: 当前正在服用的药物列表 (list，若无则为 [])
    5. allergies: 过敏史 (list，若无则为 [])
    
    请仅返回 JSON 格式，不要有任何解释。
    """
