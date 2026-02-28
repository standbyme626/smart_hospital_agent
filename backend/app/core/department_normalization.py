from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEPARTMENT_LIST: List[str] = [
    "内科",
    "外科",
    "儿科",
    "妇科",
    "耳鼻喉科",
    "眼科",
    "皮肤科",
    "口腔科",
    "精神内科",
    "神经内科",
    "心内科",
    "呼吸内科",
    "消化内科",
    "内分泌科",
    "肾内科",
    "血液科",
    "肿瘤科",
    "骨科",
    "泌尿科",
    "康复医学科",
    "中医",
]


MANUAL_ALIASES: Dict[str, str] = {
    "心血管内科": "心内科",
    "cardiology": "心内科",
    "精神科": "精神内科",
    "心理科": "精神内科",
    "心理内科": "精神内科",
    "心理治疗": "精神内科",
    "心理治疗/内科": "精神内科",
    "psychiatry": "精神内科",
    "mental health": "精神内科",
    "普通外科": "外科",
    "general surgery": "外科",
    "泌尿外科": "泌尿科",
    "urology": "泌尿科",
    "中医科": "中医",
    "tcm": "中医",
    "耳鼻咽喉科": "耳鼻喉科",
    "ent": "耳鼻喉科",
    "全科": "内科",
    "通用": "内科",
    "general": "内科",
    "妇产科": "妇科",
    "皮肤性病科": "皮肤科",
    "肝病科": "消化内科",
    "感染科": "内科",
    "呼吸科": "呼吸内科",
    "药剂科": "内科",
    "药学科": "内科",
}


def normalize_text(text: str) -> str:
    return str(text or "").strip().lower().replace("（", "(").replace("）", ")")


def _build_alias_patterns() -> List[Tuple[int, str, str]]:
    patterns: List[Tuple[int, str, str]] = []

    def add(alias: str, canonical: str, priority: int) -> None:
        norm_alias = normalize_text(alias)
        norm_canonical = str(canonical or "").strip()
        if norm_alias and norm_canonical:
            patterns.append((priority, norm_alias, norm_canonical))

    for alias, canonical in MANUAL_ALIASES.items():
        add(alias, canonical, priority=0)
    for dept in DEPARTMENT_LIST:
        priority = 2 if dept in {"内科", "外科"} else 1
        add(dept, dept, priority=priority)
    patterns.sort(key=lambda x: (x[0], -len(x[1])))
    return patterns


ALIAS_PATTERNS: List[Tuple[int, str, str]] = _build_alias_patterns()


def normalize_department_name(raw_department: str) -> Tuple[str, str]:
    raw = str(raw_department or "").strip()
    if not raw:
        return "Unknown", "empty"

    normalized = normalize_text(raw)
    for _, alias, canonical in ALIAS_PATTERNS:
        if alias in normalized:
            return canonical, f"alias:{alias}"

    for dept in DEPARTMENT_LIST:
        if dept in raw:
            return dept, f"token:{dept}"

    if len(raw) > 24 or "归类" in raw:
        return "Unknown", "long_unmapped_text"
    return raw, "passthrough"


def extract_department_mentions(text: str, top_k: int = 3) -> Tuple[List[str], List[str]]:
    body = normalize_text(text)
    if not body:
        return [], []

    hits: List[Tuple[int, int, str, str]] = []
    for priority, alias, canonical in ALIAS_PATTERNS:
        pos = body.find(alias)
        if pos >= 0:
            hits.append((pos, priority, alias, canonical))
    if not hits:
        return [], []

    hits.sort(key=lambda x: (x[0], x[1], -len(x[2])))
    raw_top: List[str] = []
    canonical_top: List[str] = []
    seen_raw = set()
    seen_canonical = set()

    for _, _, alias, canonical in hits:
        if alias not in seen_raw:
            raw_top.append(alias)
            seen_raw.add(alias)
        if canonical not in seen_canonical:
            canonical_top.append(canonical)
            seen_canonical.add(canonical)
        if len(raw_top) >= top_k and len(canonical_top) >= top_k:
            break
    return raw_top[:top_k], canonical_top[:top_k]


def normalize_department_candidates(candidates: Sequence[str], top_k: int = 3) -> List[str]:
    canonical_top: List[str] = []
    seen = set()
    for item in candidates:
        canonical, _ = normalize_department_name(str(item or ""))
        if canonical == "Unknown":
            continue
        if canonical in seen:
            continue
        seen.add(canonical)
        canonical_top.append(canonical)
        if len(canonical_top) >= top_k:
            break
    return canonical_top


def _to_confidence(value: Any) -> float:
    try:
        conf = float(value)
    except Exception:
        return 0.0
    if conf < 0:
        return 0.0
    if conf > 1:
        if conf <= 100:
            return round(conf / 100.0, 4)
        return 1.0
    return round(conf, 4)


def build_department_result(
    *,
    top3: Optional[Iterable[str]] = None,
    top1: str = "",
    confidence: Any = None,
    source: str = "",
) -> Optional[Dict[str, Any]]:
    candidates: List[str] = []
    if top1:
        candidates.append(str(top1))
    if top3:
        candidates.extend(str(x) for x in top3 if str(x or "").strip())

    canonical_top3 = normalize_department_candidates(candidates, top_k=3)
    if not canonical_top3:
        return None

    conf = _to_confidence(confidence)
    if conf <= 0:
        conf = 0.85 if len(canonical_top3) == 1 else 0.7

    result = {
        "department_top1": canonical_top3[0],
        "department_top3": canonical_top3,
        "confidence": conf,
    }
    if source:
        result["source"] = source
    return result
