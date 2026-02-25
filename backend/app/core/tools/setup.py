from typing import List, Dict, Any, Optional
from langchain_core.tools import tool, StructuredTool
from pydantic import BaseModel, Field

from app.core.tool_registry import registry
from app.tools.guideline_lookup import GuidelineLookupTool
from app.tools.drug_checker import DrugInteractionChecker

# ==================== Wrappers for Custom BaseTool ====================

# 1. Guideline Lookup Wrapper
guideline_tool_instance = GuidelineLookupTool()

class GuidelineInput(BaseModel):
    query: str = Field(description="医学问题或症状描述 (e.g., '高血压治疗指南', '胸痛鉴别诊断')")
    department: Optional[str] = Field(None, description="可选的科室过滤 (e.g., '心内科')")

async def _guideline_wrapper(query: str, department: Optional[str] = None) -> str:
    """检索医学指南和临床知识。返回结果包含指南内容摘要。"""
    result = await guideline_tool_instance.run(query=query, department=department)
    if result.get("success"):
        guidelines = result["data"].get("guidelines", [])
        if not guidelines:
            return "未找到相关指南。"
        # Format for LLM
        return "\n\n".join([f"Source: {g.get('source')}\nContent: {g.get('content')}" for g in guidelines])
    else:
        return f"检索失败: {result.get('data', {}).get('error', '未知错误')}"

guideline_tool = StructuredTool.from_function(
    func=None,
    coroutine=_guideline_wrapper,
    name="lookup_guideline",
    description="检索医学指南和临床知识。用于获取疾病诊断标准、治疗方案等权威信息。",
    args_schema=GuidelineInput
)

# 2. Drug Interaction Wrapper
drug_tool_instance = DrugInteractionChecker()

class DrugCheckInput(BaseModel):
    drugs: List[str] = Field(description="药物名称列表 (e.g., ['阿司匹林', '华法林'])")

async def _drug_check_wrapper(drugs: List[str]) -> str:
    """检查药物相互作用。"""
    result = await drug_tool_instance.run(drugs=drugs)
    if result.get("success"):
        data = result["data"]
        if not data.get("has_interaction"):
            return "✅ 未发现已知的高危药物相互作用。"
        
        interactions = data.get("interactions", [])
        warnings = []
        for i in interactions:
            warnings.append(f"⚠️ {i['drugs']}: {i['warning']} (严重程度: {i['severity']})")
        return "\n".join(warnings)
    else:
        return f"检查失败: {result.get('data', {}).get('error', '未知错误')}"

drug_checker_tool = StructuredTool.from_function(
    func=None,
    coroutine=_drug_check_wrapper,
    name="check_drug_interaction",
    description="检查药物相互作用，识别危险的药物组合。",
    args_schema=DrugCheckInput
)

# ==================== Service Tools (Mocked/Simple) ====================

class EHRInput(BaseModel):
    patient_id: str = Field(description="患者ID")

@tool("query_ehr", args_schema=EHRInput)
async def tool_query_ehr(patient_id: str) -> str:
    """查询患者电子病历 (EHR)，包括既往史、过敏史等。"""
    # Mock data
    mock_db = {
        "p001": "患者男，45岁。既往史：高血压5年，规律服用硝苯地平。过敏史：无。吸烟史：20年，每日20支。",
        "p002": "患者女，30岁。既往史：无。过敏史：青霉素过敏。",
    }
    return mock_db.get(patient_id, "未找到该患者的电子病历记录。")

class DiagnosisInput(BaseModel):
    patient_id: str = Field(description="患者ID")
    diagnosis: str = Field(description="确诊结果")
    evidence: str = Field(description="支持诊断的依据")

@tool("submit_diagnosis", args_schema=DiagnosisInput)
async def tool_submit_diagnosis(patient_id: str, diagnosis: str, evidence: str) -> str:
    """提交最终诊断结果到系统。"""
    return f"✅ 诊断已提交 (Patient: {patient_id}): {diagnosis}\n依据: {evidence[:50]}..."

class PrescriptionItem(BaseModel):
    drug_name: str
    dosage: str
    frequency: str

class PrescriptionInput(BaseModel):
    patient_id: str = Field(description="患者ID")
    medications: List[PrescriptionItem] = Field(description="药品列表")
    notes: Optional[str] = Field(None, description="医嘱备注")

@tool("submit_prescription", args_schema=PrescriptionInput)
async def tool_submit_prescription(patient_id: str, medications: List[PrescriptionItem], notes: str = "") -> str:
    """提交电子处方。"""
    meds_str = ", ".join([f"{m.drug_name} ({m.dosage})" for m in medications])
    return f"✅ 处方已提交 (Patient: {patient_id}): {meds_str}"

# ==================== Register Tools ====================

def register_all_tools():
    """注册所有工具到 Registry"""
    registry.register("lookup_guideline", guideline_tool)
    registry.register("check_drug_interaction", drug_checker_tool)
    registry.register("query_ehr", tool_query_ehr)
    registry.register("submit_diagnosis", tool_submit_diagnosis)
    registry.register("submit_prescription", tool_submit_prescription)
    
# Auto-register on import
register_all_tools()
