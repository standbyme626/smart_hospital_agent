from typing import List
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_fixed

class PrescriptionItem(BaseModel):
    """
    处方项模型 (Prescription Item Model)
    """
    drug_name: str = Field(..., description="药品名称 (Name of the drug)")
    dosage: str = Field(..., description="用法用量 (Dosage instruction, e.g., '10mg po qd')")
    quantity: str = Field(..., description="总量 (Total quantity, e.g., '2 boxes')")

@retry(stop=stop_after_attempt(2), wait=wait_fixed(0.1))
async def submit_prescription(
    patient_id: str,
    diagnosis: str,
    prescription_list: List[dict]
) -> str:
    """
    提交最终处方 (Submit Prescription)
    Prescription Agent 必须在方案最终确定后调用此工具。
    
    Args:
        patient_id: 患者 ID
        diagnosis: 最终确诊结果
        prescription_list: 药品列表，每项包含 {'drug_name', 'dosage', 'quantity'}
    
    Returns:
        str: 包含处方 ID 的确认消息
    """
    import json
    import uuid
    
    prescription_id = str(uuid.uuid4())
    
    # 在真实系统中，这里会将处方保存到数据库。
    # 目前返回一个格式化的标记块。
    
    data = {
        "prescription_id": prescription_id,
        "patient_id": patient_id,
        "diagnosis": diagnosis,
        "drugs": prescription_list,
        "status": "Submitted"
    }
    
    return f"PRESCRIPTION_SUBMITTED: {json.dumps(data, ensure_ascii=False)}"
