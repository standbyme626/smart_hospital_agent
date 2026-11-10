from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .config import DATA_DIR

URGENCY_RANK = {"normal": 0, "urgent": 1, "critical": 2}
HISTORY_CRISIS_TURNS = 4

# 危机概念词典：subtype 决定急救话术，urgency 为该概念的最低紧急度。
CRISIS_CONCEPTS: list[dict[str, Any]] = [
    {
        "subtype": "suicide",
        "urgency": "critical",
        "terms": ["自杀", "自残", "自伤", "割腕", "不想活", "活不下去", "轻生", "结束生命", "结束自己的生命", "跳楼"],
    },
    {
        "subtype": "anaphylaxis",
        "urgency": "critical",
        "terms": ["严重过敏", "过敏性休克", "过敏反应", "喉头水肿", "喉咙肿", "嗓子肿", "舌头肿", "嘴唇肿"],
    },
    {
        "subtype": "overdose",
        "urgency": "critical",
        "terms": [
            "吃药过量",
            "服药过量",
            "药物过量",
            "过量服用",
            "服用过量",
            "吃了过量",
            "中毒",
            "误服",
            "农药",
            "毒物",
        ],
    },
    {
        "subtype": "hypoglycemia",
        "urgency": "critical",
        "terms": ["低血糖", "血糖过低"],
    },
    {
        "subtype": "stroke_acs",
        "urgency": "critical",
        "terms": [
            "晕倒",
            "晕厥",
            "突然倒地",
            "雷劈",
            "炸裂样",
            "炸裂性头痛",
            "最剧烈的头痛",
            "一生中最剧烈",
            "胸口压着疼",
            "胸口压着痛",
            "压榨性胸痛",
            "剧烈胸痛",
            "胸痛伴出汗",
            "胸痛伴大汗",
            "胸痛并大汗",
            "半边身子没劲",
            "半边身子无力",
            "半边身体没劲",
            "半边没劲",
            "半边无力",
            "半身无力",
            "半身不遂",
            "一侧肢体无力",
            "偏瘫",
            "说话大舌头",
            "大舌头",
            "言语不清",
            "口齿不清",
            "说话不清",
        ],
    },
    {
        "subtype": "stroke_acs",
        "urgency": "urgent",
        "terms": ["胸痛", "心前区疼痛", "胸口痛", "心跳很快", "心跳加速", "心跳得很快"],
    },
    {
        "subtype": "seizure",
        "urgency": "critical",
        "terms": ["抽搐", "惊厥", "癫痫发作", "癫痫持续状态", "高热惊厥"],
    },
    {
        "subtype": "bleeding",
        "urgency": "critical",
        "terms": ["大出血", "出血不止", "流血不止", "止不住血", "呕血", "咯血", "咳血", "大量吐血"],
    },
    {
        "subtype": "respiratory",
        "urgency": "critical",
        "terms": [
            "呼吸困难",
            "喘不过气",
            "喘不上气",
            "无法呼吸",
            "窒息",
            "不能平卧",
            "端坐呼吸",
            "口唇发紫",
            "呼吸停止",
            "呼吸骤停",
            "无呼吸",
        ],
    },
    {
        "subtype": "altered_mental",
        "urgency": "critical",
        "terms": ["意识模糊", "意识不清", "意识下降", "昏迷", "叫不醒", "神志不清"],
    },
    {
        "subtype": "other",
        "urgency": "critical",
        "terms": ["休克", "血压测不到", "心跳骤停", "车祸", "重伤"],
    },
]

CRISIS_TERMS = [term for concept in CRISIS_CONCEPTS for term in concept["terms"]]

SUBTYPE_PRIORITY = [
    "suicide",
    "anaphylaxis",
    "overdose",
    "hypoglycemia",
    "stroke_acs",
    "seizure",
    "bleeding",
    "respiratory",
    "altered_mental",
    "other",
]

CHEST_PAIN_TERMS = ("胸痛", "胸口压着疼", "胸口压着痛", "心前区疼痛", "胸口痛", "压榨性胸痛", "剧烈胸痛")
SWEAT_TERMS = ("出冷汗", "冒冷汗", "大汗", "冷汗", "出汗")
HEART_RACING_TERMS = ("心跳很快", "心跳加速", "心跳得很快")
PREGNANCY_CUES = ("孕妇", "怀孕", "妊娠", "孕期", "备孕")
DIABETES_CUES = ("糖尿病", "低血糖", "胰岛素", "降糖药", "格列美脲", "二甲双胍")

_CLAUSE_BOUNDARY = "，。；！？,\n;!?"
_NEGATION_CUES = (
    "没有",
    "没在",
    "没吃",
    "没喝",
    "未见",
    "未吃",
    "未服",
    "未用",
    "未出现",
    "未发生",
    "未曾",
    "从未",
    "否认",
    "不是",
    "并非",
    "已停",
    "停用",
    "停药",
    "停掉",
    "不吃",
    "不喝",
    "戒酒",
    "戒了",
    "无",
    "未",
    "没",
)
_PAST_CUES = ("以前", "既往", "曾", "之前", "过去", "小时候", "得过", "做过", "用过", "曾患")
_RESET_CUES = (
    "但是现在",
    "但现在",
    "但是",
    "不过",
    "现在",
    "目前",
    "正在",
    "这次",
    "今天",
    "刚刚",
    "刚才",
    "突然",
    "只是",
    "想问",
    "问问",
    "能不能",
    "可以",
    "能吃",
    "还能",
    "未缓解",
    "未见好转",
    "未好转",
    "加重",
    "越来越",
)

CRISIS_PROTOCOLS: dict[str, dict[str, Any]] = {
    "suicide": {
        "summary": "你现在的表达让我们非常担心你的安全。请立即联系身边可信任的人陪伴你，不要独处，并拨打心理援助热线或 120/110 求助。",
        "advice": [
            "不要独处，立即请家人或朋友陪伴",
            "拨打全国心理援助热线 12356（或当地 24 小时危机干预热线）",
            "移除身边可能伤害自己的物品",
            "如有立即自伤风险，请拨打 120 或 110，或直接前往急诊",
        ],
    },
    "anaphylaxis": {
        "summary": "这可能是严重过敏反应（过敏性休克），属于急危重症，请立即拨打 120。",
        "advice": [
            "立即拨打 120",
            "如医生已为你处方肾上腺素自动注射器，请立即按医嘱使用",
            "平卧并抬高下肢，保持呼吸道通畅",
            "不要自行服用不明药物；如出现呼吸停止，立即心肺复苏",
        ],
    },
    "hypoglycemia": {
        "summary": "这可能是严重低血糖，请立即处理并呼叫急救。",
        "advice": [
            "意识清醒且能吞咽时，立即进食含糖食物（糖水、果汁或糖块）",
            "15 分钟后复测血糖，未缓解重复补糖",
            "不要给意识不清者喂食，立即拨打 120",
            "携带降糖药/胰岛素清单供急救人员参考",
        ],
    },
    "overdose": {
        "summary": "这可能是药物过量或中毒，属于急危重症，请立即拨打 120。",
        "advice": [
            "立即拨打 120，并说明摄入的药物/毒物和大致剂量",
            "不要自行催吐",
            "保留药瓶、包装或呕吐物样本供医生判断",
            "意识不清时侧卧位，保持呼吸道通畅",
        ],
    },
    "stroke_acs": {
        "summary": "这可能是卒中或急性冠脉综合征，请立即拨打 120，不要自行驾车。",
        "advice": [
            "立即拨打 120",
            "记录最后正常的时间（卒中溶栓时间窗的关键信息）",
            "不要自行驾车或独自前往医院",
            "不要自行服用阿司匹林等药物，是否用药需由急救/医生判断",
            "保持安静，松开衣领，等待救援",
        ],
    },
    "seizure": {
        "summary": "这可能是抽搐发作，请保护患者并立即呼叫急救。",
        "advice": [
            "移开周围硬物，保护头部，不要按压肢体",
            "不要往口中塞任何东西",
            "记录发作开始时间和持续时间",
            "发作超过 5 分钟、反复发作或意识不恢复时立即拨打 120",
        ],
    },
    "bleeding": {
        "summary": "这可能存在严重出血，请立即拨打 120 并压迫止血。",
        "advice": [
            "立即拨打 120",
            "对可见伤口直接持续压迫止血",
            "呕血/咯血时侧卧位，防止误吸",
            "不要进食饮水，保留出血情况供医生判断",
        ],
    },
    "respiratory": {
        "summary": "这可能是严重呼吸困难，请立即拨打 120。",
        "advice": [
            "立即拨打 120",
            "取坐位或半卧位，保持呼吸道通畅",
            "如备有医生处方的急救药，按医嘱使用",
            "出现意识不清或呼吸停止时立即心肺复苏",
        ],
    },
    "altered_mental": {
        "summary": "这可能是意识障碍，属于急危重症，请立即拨打 120。",
        "advice": [
            "立即拨打 120",
            "保持呼吸道通畅，侧卧位防止误吸",
            "不要喂食、喂水或自行喂药",
            "记录意识变化时间和伴随症状",
        ],
    },
    "other": {
        "summary": "你描述的情况可能属于急危重症，请立即拨打 120 或前往最近的急诊科。",
        "advice": [
            "立即拨打 120",
            "不要自行驾车",
            "保持安静，等待救援",
            "携带既往病历和用药清单",
        ],
    },
}


def crisis_protocol(subtype: str | None) -> dict[str, Any]:
    protocol = CRISIS_PROTOCOLS.get(str(subtype or "other"), CRISIS_PROTOCOLS["other"])
    return {"subtype": subtype or "other", **protocol}


DEPARTMENT_KEYWORDS: list[tuple[list[str], str]] = [
    (["儿童", "孩子", "宝宝", "婴儿", "小儿", "新生儿"], "儿科"),
    (["孕妇", "妊娠", "怀孕", "备孕", "产检", "月经", "痛经", "白带", "妇科", "产科"], "妇产科"),
    (["药物", "用药", "相互作用", "相斥", "剂量", "副作用", "不良反应"], "精准药剂科"),
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

PRIORITY_DEPARTMENT_KEYWORDS: list[tuple[list[str], str]] = [
    (["药物流产", "药流"], "妇产科"),
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
        return "全科"
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
    return "全科"


def split_member_prefix(text: str) -> tuple[str, str]:
    """Split the legacy ``{member_context}\\n\\n用户输入：{message}`` envelope."""
    content = text or ""
    marker = "用户输入："
    if marker not in content:
        return "", content
    prefix, current = content.rsplit(marker, 1)
    return prefix.strip(), current.strip()


def strip_member_prefix(text: str) -> str:
    return split_member_prefix(text)[1]


def _clause_prefix(content: str, position: int) -> str:
    start = 0
    for char in _CLAUSE_BOUNDARY:
        index = content.rfind(char, 0, position)
        if index >= 0:
            start = max(start, index + 1)
    return content[start:position]


def _is_negated(content: str, position: int) -> bool:
    prefix = _clause_prefix(content, position)
    if not prefix:
        return False
    cues = _NEGATION_CUES + _PAST_CUES
    cue = ""
    cue_at = -1
    for candidate in cues:
        index = prefix.rfind(candidate)
        if index > cue_at:
            cue = candidate
            cue_at = index
    if cue_at < 0:
        return False
    tail = prefix[cue_at:]
    if cue == "无" and "诱因" in tail[:6]:
        return False
    if any(reset in tail for reset in _RESET_CUES):
        return False
    return True


def _match_terms(content: str, terms: tuple[str, ...] | list[str]) -> list[str]:
    hits: list[str] = []
    for term in terms:
        start = 0
        while True:
            index = content.find(term, start)
            if index < 0:
                break
            if not _is_negated(content, index):
                if term not in hits:
                    hits.append(term)
                break
            start = index + len(term)
    return hits


def _pregnancy_cue(content: str) -> bool:
    return bool(_match_terms(content, PREGNANCY_CUES))


def _scan_crisis(content: str) -> tuple[list[str], set[str], str]:
    text = content or ""
    terms: list[str] = []
    subtypes: set[str] = set()
    urgency = "normal"
    for concept in CRISIS_CONCEPTS:
        hits = _match_terms(text, concept["terms"])
        if not hits:
            continue
        for hit in hits:
            if hit not in terms:
                terms.append(hit)
        subtypes.add(concept["subtype"])
        if concept["urgency"] == "critical":
            urgency = "critical"
        elif urgency == "normal":
            urgency = "urgent"

    chest = bool(_match_terms(text, CHEST_PAIN_TERMS))
    sweat = bool(_match_terms(text, SWEAT_TERMS))
    heart = bool(_match_terms(text, HEART_RACING_TERMS))
    pregnant = _pregnancy_cue(text)
    diabetes = bool(_match_terms(text, DIABETES_CUES))
    if (chest and sweat) or (heart and sweat) or (chest and pregnant):
        subtypes.add("stroke_acs")
        urgency = "critical"
    if diabetes and (sweat or "意识模糊" in terms):
        subtypes.add("hypoglycemia")
        urgency = "critical"
    return terms, subtypes, urgency


def _recent_user_texts(history: list[dict[str, Any]] | None, limit: int = HISTORY_CRISIS_TURNS) -> list[str]:
    if not history:
        return []
    texts: list[str] = []
    for item in history:
        if not isinstance(item, dict) or item.get("role") != "user":
            continue
        content = item.get("content")
        if isinstance(content, str) and content.strip():
            texts.append(strip_member_prefix(content))
    return texts[-limit:]


def detect_crisis_context(
    text: str,
    *,
    history: list[dict[str, Any]] | None = None,
    member_conditions: list[str] | str | None = None,
) -> dict[str, Any]:
    """Crisis detection over the current utterance plus recent user turns.

    ``member_conditions`` is accepted for call-site compatibility but is never
    scanned: past medical history must not trigger a current crisis.
    """
    current = strip_member_prefix(text or "")
    terms, subtypes, urgency = _scan_crisis(current)
    source = "current" if terms else None
    for item in _recent_user_texts(history):
        history_terms, history_subtypes, history_urgency = _scan_crisis(item)
        if not history_terms:
            continue
        if source is None:
            source = "history"
        for term in history_terms:
            if term not in terms:
                terms.append(term)
        subtypes = subtypes | history_subtypes
        if URGENCY_RANK.get(history_urgency, 0) > URGENCY_RANK.get(urgency, 0):
            urgency = history_urgency
    ordered = [name for name in SUBTYPE_PRIORITY if name in subtypes]
    return {
        "matched": bool(terms),
        "terms": terms,
        "subtypes": ordered,
        "urgency": urgency if terms else None,
        "source": source,
    }


def detect_crisis(
    text: str,
    *,
    history: list[dict[str, Any]] | None = None,
    member_conditions: list[str] | str | None = None,
) -> tuple[bool, list[str]]:
    context = detect_crisis_context(text, history=history, member_conditions=member_conditions)
    return context["matched"], context["terms"]


def apply_urgency_floor(
    triage: dict[str, Any],
    text: str,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Never let a model output silently downgrade a rule-positive crisis."""
    context = detect_crisis_context(text, history=history)
    if not context["matched"]:
        return triage
    result = dict(triage)
    current = str(result.get("urgency") or "normal").strip().lower()
    if URGENCY_RANK.get(current, 0) < URGENCY_RANK[context["urgency"]]:
        result["urgency"] = context["urgency"]
    result["crisis_terms"] = context["terms"]
    result["crisis_subtype"] = context["subtypes"][0] if context["subtypes"] else "other"
    if context["urgency"] == "critical":
        result["intent"] = "CRISIS"
        result["department"] = "急诊科"
    return result


def _first_keyword(content: str, keywords: list[str]) -> str | None:
    for keyword in keywords:
        start = 0
        while True:
            index = content.find(keyword, start)
            if index < 0:
                break
            if not _is_negated(content, index):
                return keyword
            start = index + len(keyword)
    return None


def rule_triage(text: str, *, history: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    content = strip_member_prefix(text or "")
    context = detect_crisis_context(content, history=history)
    if context["matched"]:
        critical = context["urgency"] == "critical"
        return {
            "intent": "CRISIS" if critical else "MEDICAL_CONSULT",
            "department": "急诊科" if critical else "心血管内科",
            "urgency": context["urgency"],
            "symptoms": content[:200],
            "reason": "命中危机概念：" + "、".join(context["terms"]),
            "confidence": None,
            "rule_triggered": True,
            "crisis_terms": context["terms"],
            "crisis_subtype": context["subtypes"][0] if context["subtypes"] else "other",
        }
    for keywords, department in PRIORITY_DEPARTMENT_KEYWORDS:
        keyword = _first_keyword(content, keywords)
        if keyword:
            return {
                "intent": "MEDICAL_CONSULT",
                "department": department,
                "urgency": "urgent" if _first_keyword(content, ["剧烈", "持续", "加重", "高烧"]) else "normal",
                "symptoms": content[:200],
                "reason": f"命中科室关键词：{keyword}",
                "confidence": 0.6,
            }
    for keywords, department in DEPARTMENT_KEYWORDS:
        keyword = _first_keyword(content, keywords)
        if keyword:
            urgency_keyword = _first_keyword(content, ["剧烈", "持续", "加重", "高烧"])
            return {
                "intent": "MEDICAL_CONSULT",
                "department": department,
                "urgency": "urgent" if urgency_keyword else "normal",
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


_UNKNOWN_ENTITY_RE = re.compile(
    r"[\u4e00-\u9fff]{2,10}(?:片|胶囊|颗粒|冲剂|口服液|糖浆|软膏|乳膏|滴眼液|滴耳液|喷雾剂|气雾剂|贴剂|栓剂|注射液|注射剂)"
    r"|[\u4e00-\u9fff]{2,8}(?:西林|霉素|沙星|地平|洛尔|普利|沙坦|他汀|拉唑|唑仑|列汀|列净|替丁|帕明)"
    r"|头孢[\u4e00-\u9fff]{1,4}"
)
_UNKNOWN_BLACKLIST = (
    "照片",
    "图片",
    "卡片",
    "名片",
    "芯片",
    "胶片",
    "片刻",
    "肉片",
    "薯片",
    "影片",
    "唱片",
    "刀片",
    "叶片",
    "雪片",
)
_UNKNOWN_STOPWORDS = (
    "我在吃",
    "正在吃",
    "在吃",
    "吃了",
    "服用",
    "服用了",
    "用了",
    "使用",
    "我正在",
    "一种",
    "这个",
    "那个",
    "我",
    "在",
    "吃",
    "了",
    "用",
    "服",
    "的",
    "有",
    "是",
    "和",
    "与",
    "及",
    "、",
    "，",
    ",",
    " ",
    "想",
    "问",
    "能",
    "可以",
    "吗",
    "呢",
)
_DOSAGE_WORDS = (
    "缓释",
    "肠溶",
    "控释",
    "分散",
    "咀嚼",
    "泡腾",
    "胶囊",
    "颗粒",
    "冲剂",
    "口服液",
    "糖浆",
    "软膏",
    "乳膏",
    "滴眼液",
    "滴耳液",
    "喷雾剂",
    "气雾剂",
    "贴剂",
    "栓剂",
    "注射液",
    "注射剂",
    "片",
    "粒",
    "袋",
    "支",
)

_DRUG_NEGATION_CUES = _NEGATION_CUES + _PAST_CUES
_DRUG_RESET_CUES = _RESET_CUES + ("只是问", "只是问问")


class DrugChecker:
    def __init__(self, path: Path | None = None):
        path = path or (DATA_DIR / "drug_interactions.json")
        data = json.loads(path.read_text(encoding="utf-8"))
        self.drugs = data.get("drugs", [])
        self.interactions = data.get("interactions", [])
        self.pregnancy = data.get("pregnancy_contraindications", [])
        self.alias_map: list[tuple[str, str]] = []
        self._kinds: dict[str, str] = {}
        for drug in self.drugs:
            self._kinds[drug["name"]] = str(drug.get("kind") or "drug")
            for alias in {drug["name"], *drug.get("aliases", [])}:
                if alias:
                    self.alias_map.append((alias.lower(), drug["name"]))
        self.alias_map.sort(key=lambda item: len(item[0]), reverse=True)
        self._known_aliases = [alias for alias, _ in self.alias_map]

    def is_exposure(self, name: str) -> bool:
        return self._kinds.get(name) == "exposure"

    def _negated_at(self, content: str, position: int) -> bool:
        prefix = _clause_prefix(content, position)
        if not prefix:
            return False
        cue_at = -1
        for candidate in _DRUG_NEGATION_CUES:
            index = prefix.rfind(candidate)
            if index > cue_at:
                cue_at = index
        if cue_at < 0:
            return False
        tail = prefix[cue_at:]
        if any(reset in tail for reset in _DRUG_RESET_CUES):
            return False
        return True

    def extract_drugs(self, text: str) -> list[str]:
        content = (text or "").lower()
        found: list[str] = []
        for alias, name in self.alias_map:
            start = 0
            while True:
                index = content.find(alias, start)
                if index < 0:
                    break
                if not self._negated_at(content, index):
                    if name not in found:
                        found.append(name)
                    break
                start = index + len(alias)
        return found

    def _clean_candidate(self, candidate: str) -> str:
        value = candidate
        for alias in sorted(self._known_aliases, key=len, reverse=True):
            value = value.replace(alias, "")
        for word in _UNKNOWN_STOPWORDS:
            value = value.replace(word, "")
        for word in _DOSAGE_WORDS:
            value = value.replace(word, "")
        return value.strip()

    def find_unknown(self, text: str) -> list[str]:
        content = text or ""
        unknowns: list[str] = []
        for match in _UNKNOWN_ENTITY_RE.finditer(content):
            candidate = match.group(0)
            cleaned = self._clean_candidate(candidate)
            if len(cleaned) < 2:
                continue
            if cleaned in _UNKNOWN_BLACKLIST or candidate in _UNKNOWN_BLACKLIST:
                continue
            known = any(alias in candidate.lower() for alias in self._known_aliases)
            if known and not cleaned:
                continue
            if cleaned not in unknowns:
                unknowns.append(cleaned)
        return unknowns

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

    def _pregnant(self, text: str) -> bool:
        return _pregnancy_cue(text)

    def check_text(self, text: str, *, pregnant: bool | None = None) -> dict[str, Any]:
        recognized = self.extract_drugs(text)
        exposures = [name for name in recognized if self.is_exposure(name)]
        drugs = [name for name in recognized if not self.is_exposure(name)]
        interactions = self.check(recognized)
        pregnancy_flags: list[dict[str, Any]] = []
        is_pregnant = self._pregnant(text) if pregnant is None else bool(pregnant)
        if is_pregnant:
            for item in self.pregnancy:
                if item["drug"] in recognized:
                    pregnancy_flags.append(item)
        unrecognized = self.find_unknown(text)
        if recognized and not unrecognized:
            coverage = "full"
        elif recognized:
            coverage = "partial"
        else:
            coverage = "none"
        return {
            "drugs": drugs,
            "exposures": exposures,
            "interactions": interactions,
            "pregnancy_contraindications": pregnancy_flags,
            "unrecognized": unrecognized,
            "coverage": coverage,
            "highest_severity": self._highest(interactions),
        }

    @staticmethod
    def _highest(interactions: list[dict[str, Any]]) -> str:
        order = {"contraindicated": 4, "major": 3, "moderate": 2, "minor": 1}
        if not interactions:
            return "none"
        return max(interactions, key=lambda item: order.get(item["severity"], 0))["severity"]


DRUG_CHECKER = DrugChecker()

SEVERITY_RANK = {"none": 0, "minor": 1, "moderate": 2, "major": 3, "high": 4, "block": 5}


def merge_drug_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate every drug-check result without letting a later 'none' win."""
    merged: dict[str, Any] = {
        "drugs": [],
        "exposures": [],
        "interactions": [],
        "pregnancy_contraindications": [],
        "unrecognized": [],
        "coverage": "none",
        "highest_severity": "none",
    }
    if not results:
        return merged
    seen_pairs: set[tuple[str, str, str]] = set()
    seen_pregnancy: set[str] = set()
    for result in results:
        for name in result.get("drugs") or []:
            if name not in merged["drugs"]:
                merged["drugs"].append(name)
        for name in result.get("exposures") or []:
            if name not in merged["exposures"]:
                merged["exposures"].append(name)
        for name in result.get("unrecognized") or []:
            if name not in merged["unrecognized"]:
                merged["unrecognized"].append(name)
        for item in result.get("interactions") or []:
            key = (str(item.get("drug_a")), str(item.get("drug_b")), str(item.get("severity")))
            if key not in seen_pairs:
                seen_pairs.add(key)
                merged["interactions"].append(item)
        for item in result.get("pregnancy_contraindications") or []:
            drug = str(item.get("drug"))
            if drug not in seen_pregnancy:
                seen_pregnancy.add(drug)
                merged["pregnancy_contraindications"].append(item)
        coverage = str(result.get("coverage") or "none")
        if coverage == "partial":
            merged["coverage"] = "partial"
        elif coverage == "full" and merged["coverage"] == "none":
            merged["coverage"] = "full"
    merged["highest_severity"] = DrugChecker._highest(merged["interactions"])
    return merged


_CONTENT_POLICY: list[tuple[str, list[re.Pattern[str]]]] = [
    (
        "high",
        [
            re.compile(r"已确诊"),
            re.compile(r"确诊为"),
            re.compile(r"确诊了"),
            re.compile(r"诊断为"),
            re.compile(r"明确诊断"),
        ],
    ),
    (
        "block",
        [
            re.compile(r"(?:每天|每日)[^。；\n]{0,10}\d+\s*次"),
            re.compile(r"\d+\s*次\s*/?\s*(?:日|天)"),
            re.compile(r"每次[^。；\n]{0,6}[0-9一二两三四五六七八九十半]+\s*(?:片|粒|袋|支)"),
            re.compile(r"每次[^。；\n]{0,6}\d+\s*(?:ml|毫升|mg|毫克|g|克)"),
            re.compile(r"口服[^。；\n]{0,30}(?:\d|每次|每日|一天)"),
            re.compile(r"\d+\s*(?:mg|毫克|g|克|ml|毫升)\s*(?:/次|每次|每日|一天|bid|tid|qd)", re.IGNORECASE),
            re.compile(r"停药"),
            re.compile(r"停用"),
            re.compile(r"继续用药"),
            re.compile(r"继续服用"),
            re.compile(r"开具处方"),
            re.compile(r"开处方"),
            re.compile(r"给你开"),
            re.compile(r"处方[：:]"),
        ],
    ),
]


def _content_policy_issues(report: dict[str, Any]) -> list[tuple[str, str]]:
    content = "\n".join(
        [
            str(report.get("summary") or ""),
            " ".join(str(item) for item in (report.get("advice") or [])),
            " ".join(str(item) for item in (report.get("medications") or [])),
        ]
    )
    issues: list[tuple[str, str]] = []
    for level, patterns in _CONTENT_POLICY:
        if any(pattern.search(content) for pattern in patterns):
            if level == "high":
                issues.append((level, "报告包含确诊/诊断结论，需人工复核"))
            else:
                issues.append((level, "报告包含处方/剂量/停药等治疗指令，需人工复核"))
    return issues


def safety_gate(report: dict[str, Any], drug_result: dict[str, Any] | None = None) -> dict[str, Any]:
    issues: list[str] = []
    drug = drug_result or {}
    severity = str(drug.get("highest_severity") or "none").lower()
    if severity in {"major", "contraindicated"}:
        issues.append(f"药物相互作用风险为 {severity}")
    if drug.get("pregnancy_contraindications"):
        issues.append("存在妊娠禁忌药物")
    if drug.get("unrecognized"):
        issues.append("存在未识别药物/暴露，未完成相互作用核查：" + "、".join(drug["unrecognized"]))
    urgency = str(report.get("urgency") or "").strip().lower()
    if urgency == "critical":
        issues.append("危机级别，需要立即人工/急诊介入")
    elif urgency not in {"normal", "urgent", "critical"}:
        issues.append(f"紧急度无效：{report.get('urgency')}")
    raw_department = str(report.get("department") or "").strip()
    if raw_department and raw_department != "General Practice":
        canonical = normalize_department(raw_department)
        if canonical == "全科" and raw_department != "全科":
            issues.append(f"科室不在注册表：{raw_department}")
    confidence = report.get("confidence")
    if confidence is not None and not report.get("rule_triggered") and float(confidence or 0) < 0.5:
        issues.append("置信度低于 0.5")
    content_issues = _content_policy_issues(report)
    issues.extend(issue for _, issue in content_issues)
    gate_severity = severity
    for level, _ in content_issues:
        if SEVERITY_RANK.get(level, 0) > SEVERITY_RANK.get(gate_severity, 0):
            gate_severity = level
    return {
        "requires_human_review": bool(issues),
        "issues": issues,
        "severity": gate_severity,
    }
