from typing import Dict, Any
import structlog

logger = structlog.get_logger()

# Mock 数据库
MOCK_EHR_DB = {
    "P001": {
        "name": "张三",
        "age": 45,
        "gender": "Male",
        "allergies": ["Penicillin", "Peanuts"],
        "history": ["Hypertension (5 years)", "Type 2 Diabetes"],
    },
    "default": {
        "name": "未知患者",
        "age": 30,
        "gender": "Female",
        "allergies": [],
        "history": [],
    }
}

MOCK_LAB_DB = {
    "P001": [
        {"date": "2023-10-01", "item": "WBC", "value": "12.5", "unit": "10^9/L", "ref": "3.5-9.5", "flag": "HIGH"},
        {"date": "2023-10-01", "item": "Hb", "value": "130", "unit": "g/L", "ref": "130-175", "flag": "NORMAL"},
    ]
}

async def query_ehr(patient_id: str = "P001") -> Dict[str, Any]:
    """
    模拟查询电子病历系统 (EHR)。
    返回患者基本信息和病史。
    """
    logger.info("tool.ehr.query", patient_id=patient_id)
    return MOCK_EHR_DB.get(patient_id, MOCK_EHR_DB["default"])

async def query_lab_results(patient_id: str = "P001") -> list:
    """
    模拟查询检验系统 (LIS)。
    返回最近的化验单结果。
    """
    logger.info("tool.lab.query", patient_id=patient_id)
    return MOCK_LAB_DB.get(patient_id, [])
