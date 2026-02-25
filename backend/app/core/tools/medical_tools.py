from typing import List, Dict, Any
import json
from app.rag.retriever import get_retriever as get_shared_retriever
from app.rag.ddinter_checker import DDInterChecker

# Singleton instances (lazy loading to avoid circular imports and startup delay)
_ddinter = None

def get_retriever():
    # Reuse unified retriever singleton to prevent duplicate model instantiation.
    return get_shared_retriever()

def get_ddinter():
    global _ddinter
    if _ddinter is None:
        _ddinter = DDInterChecker()
    return _ddinter

def lookup_guideline(query: str) -> str:
    """
    检索医疗指南和临床知识
    Args:
        query: 检索问题 (e.g., "高血压的治疗方案")
    Returns:
        检索到的文档内容字符串
    """
    try:
        retriever = get_retriever()
        # 同步工具中使用同步兼容接口，避免 coroutine 未 await
        results = retriever.search(query, top_k=3)
        if not results:
            return "未找到相关指南信息。"
        if isinstance(results, list):
            normalized = []
            for item in results:
                if isinstance(item, dict):
                    normalized.append(str(item.get("content", "")))
                else:
                    normalized.append(str(item))
            normalized = [x for x in normalized if x]
            if not normalized:
                return "未找到相关指南信息。"
            return "\n\n".join(normalized)
        return str(results)
    except Exception as e:
        return f"检索失败: {str(e)}"

async def check_drug_interaction(drugs: List[str]) -> str:
    """
    检查药物相互作用
    Args:
        drugs: 药物名称列表 (e.g., ["头孢", "酒精"])
    Returns:
        警告信息，如果没有风险则返回 "未发现已知相互作用"
    """
    try:
        checker = get_ddinter()
        warnings = await checker.check_async(drugs)
        if not warnings:
            return "✅ 未发现已知的高危药物相互作用。"
        return "⚠️ 警告:\n" + "\n".join(warnings)
    except Exception as e:
        return f"检查失败: {str(e)}"

def submit_diagnosis(patient_id: str, diagnosis: str, evidence: str) -> str:
    """
    提交诊断结果到 EHR 系统
    """
    # Mock implementation
    return f"诊断已提交 (Patient: {patient_id}): {diagnosis}"

def submit_prescription(patient_id: str, prescription: List[Dict[str, Any]]) -> str:
    """
    提交处方
    """
    # Mock implementation
    return f"处方已提交 (Patient: {patient_id}), 包含 {len(prescription)} 种药物。"

def query_ehr(patient_id: str) -> str:
    """
    查询电子病历 (EHR)
    """
    # Mock implementation
    mock_ehr = {
        "p001": "患者男，45岁，既往有高血压病史5年，规律服用硝苯地平。主诉头痛2天。",
        "p002": "患者女，30岁，无既往病史。主诉咳嗽、发热1天。",
    }
    return mock_ehr.get(patient_id, "未找到该患者的电子病历。")

def query_lab_results(patient_id: str) -> str:
    """
    查询检验检查结果
    """
    # Mock implementation
    mock_lab = {
        "p001": "血压: 160/100 mmHg; 血糖: 5.8 mmol/L; 血常规: 正常。",
        "p002": "体温: 38.5°C; 血常规: WBC 12.0*10^9/L (升高), NEUT% 80% (升高)。",
    }
    return mock_lab.get(patient_id, "未找到该患者的近期检验结果。")
