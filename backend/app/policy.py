from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import DATA_DIR

CRISIS_TERMS = [
    "胸痛并大汗",
    "胸痛放射",
    "放射左臂",
    "濒死感",
    "大汗",
    "割腕",
    "流血不止",
    "出血不止",
    "大量出血",
    "止不住",
    "吃药过量",
    "服药过量",
    "药物过量",
    "农药",
    "脑出血",
    "意识下降",
    "意识模糊",
    "呼吸骤停",
    "呼吸停止",
    "叫120",
    "急救",
    "剧烈胸痛",
    "压榨性胸痛",
    "胸痛伴出汗",
    "呼吸困难",
    "喘不上气",
    "窒息",
    "昏迷",
    "意识不清",
    "大出血",
    "呕血",
    "咯血",
    "抽搐",
    "惊厥",
    "休克",
    "血压测不到",
    "自杀",
    "自残",
    "中毒",
    "误服",
    "严重过敏",
    "喉头水肿",
    "中风",
    "偏瘫",
    "言语不清",
    "心梗",
    "心跳骤停",
    "无呼吸",
    "高热惊厥",
    "车祸",
    "重伤",
]

DEPARTMENT_KEYWORDS: list[tuple[list[str], str]] = [
    (["儿童", "孩子", "宝宝", "婴儿", "小儿", "新生儿"], "儿科"),
    (["孕妇", "妊娠", "怀孕", "备孕", "产检", "月经", "痛经", "白带", "妇科", "产科"], "妇产科"),
    (["药物", "用药", "相互作用", "相斥", "剂量", "副作用", "不良反应"], "药剂科_精准医学"),
    (["胸痛", "心悸", "心慌", "高血压", "血压", "冠心病", "心律", "心衰", "心脏"], "心血管内科"),
    (["头痛", "头晕", "眩晕", "麻木", "癫痫", "抽搐", "偏瘫", "言语不清", "失眠"], "神经内科"),
    (["咳嗽", "咳痰", "气喘", "哮喘", "呼吸困难", "肺炎", "支气管", "肺"], "呼吸内科"),
    (["腹痛", "胃痛", "腹泻", "恶心", "呕吐", "便秘", "反酸", "黄疸", "消化"], "消化内科"),
    (["水肿", "尿频", "尿急", "尿痛", "血尿", "蛋白尿", "肾功能", "肾"], "肾内科"),
    (["血糖", "糖尿病", "甲状腺", "多饮", "多尿", "体重下降", "内分泌"], "内分泌科"),
    (["牙痛", "牙龈", "口腔溃疡", "口腔"], "口腔科"),
    (["耳鸣", "耳痛", "鼻塞", "咽痛", "扁桃体", "声音嘶哑", "耳鼻喉"], "耳鼻喉科"),
    (["眼痛", "视力", "红眼", "流泪", "眼科"], "眼科"),
    (["皮疹", "瘙痒", "湿疹", "荨麻疹", "痘痘", "皮肤"], "皮肤科"),
    (["腰痛", "关节", "骨折", "颈椎", "腰椎", "扭伤", "骨科"], "骨科"),
    (["尿频", "尿急", "尿痛", "前列腺", "结石", "泌尿"], "泌尿外科"),
    (["外伤", "伤口", "肿块", "疝", "阑尾", "手术", "外科"], "普通外科"),
    (["康复", "理疗", "功能训练"], "康复医学科"),
    (["中医", "针灸", "调理", "中药"], "中医科"),
    (["急诊", "急救", "危机"], "急诊科"),
]


def load_specialists() -> dict[str, Any]:
    path = DATA_DIR / "specialists.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("specialists", {})


SPECIALISTS = load_specialists()
DEPARTMENT_NAMES = [item.get("name_cn", key) for key, item in SPECIALISTS.items()]
_DEPARTMENT_ALIASES: dict[str, str] = {}
for _key, _item in SPECIALISTS.items():
    _canonical = _item.get("name_cn", _key)
    for _alias in {_key, _canonical, *_item.get("aliases", [])}:
        if _alias:
            _DEPARTMENT_ALIASES[str(_alias).lower()] = _canonical


def normalize_department(value: str | None) -> str:
    text = str(value or "").strip()
    if not text:
        return "General Practice"
    if text in DEPARTMENT_NAMES:
        return text
    direct = _DEPARTMENT_ALIASES.get(text.lower())
    if direct:
        return direct
    for alias, canonical in _DEPARTMENT_ALIASES.items():
        if alias and alias in text.lower():
            return canonical
    for keywords, department in DEPARTMENT_KEYWORDS:
        for keyword in keywords:
            if keyword in text:
                return department
    return "General Practice"


def detect_crisis(text: str) -> tuple[bool, list[str]]:
    content = text or ""
    matched = [term for term in CRISIS_TERMS if term in content]
    return bool(matched), matched


def rule_triage(text: str) -> dict[str, Any]:
    content = text or ""
    crisis, crisis_terms = detect_crisis(content)
    if crisis:
        return {
            "intent": "CRISIS",
            "department": "急诊科",
            "urgency": "critical",
            "symptoms": content[:200],
            "reason": "命中危机关键词：" + "、".join(crisis_terms),
            "confidence": 0.95,
        }
    for keywords, department in DEPARTMENT_KEYWORDS:
        for keyword in keywords:
            if keyword in content:
                return {
                    "intent": "MEDICAL_CONSULT",
                    "department": department,
                    "urgency": "urgent" if any(term in content for term in ["剧烈", "持续", "加重", "高烧"]) else "normal",
                    "symptoms": content[:200],
                    "reason": f"命中科室关键词：{keyword}",
                    "confidence": 0.6,
                }
    return {
        "intent": "MEDICAL_CONSULT",
        "department": "General Practice",
        "urgency": "normal",
        "symptoms": content[:200],
        "reason": "未命中特定科室关键词，进入全科",
        "confidence": 0.4,
    }


class DrugChecker:
    def __init__(self, path: Path | None = None):
        path = path or (DATA_DIR / "drug_interactions.json")
        data = json.loads(path.read_text(encoding="utf-8"))
        self.drugs = data.get("drugs", [])
        self.interactions = data.get("interactions", [])
        self.pregnancy = data.get("pregnancy_contraindications", [])
        self.alias_map: list[tuple[str, str]] = []
        for drug in self.drugs:
            for alias in {drug["name"], *drug.get("aliases", [])}:
                if alias:
                    self.alias_map.append((alias.lower(), drug["name"]))
        self.alias_map.sort(key=lambda item: len(item[0]), reverse=True)

    def extract_drugs(self, text: str) -> list[str]:
        content = (text or "").lower()
        found: list[str] = []
        for alias, name in self.alias_map:
            if alias in content and name not in found:
                found.append(name)
        return found

    def check(self, drugs: list[str]) -> list[dict[str, Any]]:
        present = set(drugs)
        results: list[dict[str, Any]] = []
        for item in self.interactions:
            if item["a"] in present and item["b"] in present:
                results.append(
                    {
                        "drug_a": item["a"],
                        "drug_b": item["b"],
                        "severity": item["severity"],
                        "description": item["description"],
                    }
                )
        return results

    def check_text(self, text: str) -> dict[str, Any]:
        drugs = self.extract_drugs(text)
        interactions = self.check(drugs)
        pregnancy_flags: list[dict[str, Any]] = []
        if any(term in (text or "") for term in ["妊娠", "怀孕", "孕妇", "备孕"]):
            for item in self.pregnancy:
                if item["drug"] in drugs:
                    pregnancy_flags.append(item)
        return {
            "drugs": drugs,
            "interactions": interactions,
            "pregnancy_contraindications": pregnancy_flags,
            "highest_severity": self._highest(interactions),
        }

    @staticmethod
    def _highest(interactions: list[dict[str, Any]]) -> str:
        order = {"contraindicated": 4, "major": 3, "moderate": 2, "minor": 1}
        if not interactions:
            return "none"
        return max(interactions, key=lambda item: order.get(item["severity"], 0))["severity"]


DRUG_CHECKER = DrugChecker()


def safety_gate(report: dict[str, Any], drug_result: dict[str, Any] | None = None) -> dict[str, Any]:
    issues: list[str] = []
    severity = (drug_result or {}).get("highest_severity", "none")
    if severity in {"major", "contraindicated"}:
        issues.append(f"药物相互作用风险为 {severity}")
    if (drug_result or {}).get("pregnancy_contraindications"):
        issues.append("存在妊娠禁忌药物")
    if report.get("urgency") == "critical":
        issues.append("危机级别，需要立即人工/急诊介入")
    department = report.get("department")
    if department and department not in DEPARTMENT_NAMES and department != "General Practice":
        issues.append(f"科室不在注册表：{department}")
    confidence = float(report.get("confidence") or 0)
    if confidence < 0.5:
        issues.append("置信度低于 0.5")
    return {
        "requires_human_review": bool(issues),
        "issues": issues,
        "severity": severity,
    }
