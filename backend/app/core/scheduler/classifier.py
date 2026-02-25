
import re
from app.core.scheduler.priority_guard import Priority

# 紧急医学关键词 (Emergency)
EMERGENCY_KEYWORDS = [
    r"胸痛", r"呼吸困难", r"出血不止", r"昏迷", r"中毒", 
    r"心脏病发", r"脑梗", r"过敏性休克", r"骨折", r"高烧不退"
]

# 挂号咨询关键词 (Urgent)
URGENT_KEYWORDS = [
    r"挂号", r"预约", r"门诊", r"医生", r"专家", r"科室", r"检查"
]

def classify_priority(message: str) -> Priority:
    """
    轻量级意图优先级分类器。
    通过正则匹配快速确定优先级，避免 LLM 推理开销。
    """
    # 1. 检查紧急
    for kw in EMERGENCY_KEYWORDS:
        if re.search(kw, message):
            return Priority.EMERGENCY
            
    # 2. 检查挂号/咨询
    for kw in URGENT_KEYWORDS:
        if re.search(kw, message):
            return Priority.URGENT
            
    # 3. 闲聊判断 (简单长度或关键词)
    if len(message) < 5 and any(kw in message for kw in ["你好", "在吗", "嗨"]):
        return Priority.LOW
        
    # 默认普通咨询
    return Priority.NORMAL
