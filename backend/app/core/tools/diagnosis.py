from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_fixed

@retry(stop=stop_after_attempt(2), wait=wait_fixed(0.1))
async def submit_diagnosis(
    diagnosis: str,
    confidence: str,
    reasoning: str
) -> str:
    """
    提交最终诊断结果 (Submit Diagnosis)
    Doctor Agent 必须在确认诊断后、开具处方前调用此工具。
    
    Args:
        diagnosis: 确诊结果 (e.g., '急性咽炎')
        confidence: 置信度 ('High', 'Medium', 'Low')
        reasoning: 临床诊断依据 (Clinical reasoning)
    
    Returns:
        str: 提交确认消息
    """
    import json
    
    data = {
        "diagnosis": diagnosis,
        "confidence": confidence,
        "reasoning": reasoning
    }
    
    return f"DIAGNOSIS_SUBMITTED: {json.dumps(data, ensure_ascii=False)}"
