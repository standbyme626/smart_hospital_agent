from __future__ import annotations

from dataclasses import dataclass, asdict
import re
from typing import Dict, List, Optional

import structlog

from app.core.config import settings

logger = structlog.get_logger(__name__)


@dataclass
class RetrievalPlan:
    original_query: str
    primary_query: str
    query_variants: List[str]
    top_k: int
    complexity: str
    rewrite_source: str
    index_scope: str

    def to_state_dict(self) -> Dict[str, object]:
        return asdict(self)


_COMPLEXITY_KEYWORDS = {
    "high": ["并且", "同时", "多年", "反复", "既往", "家族史", "药物", "检查", "复查", "诊断"],
    "medium": ["多久", "怎么", "是否", "症状", "原因", "治疗"],
}

_RULE_REWRITE_PAIRS = [
    ("胃疼", "胃痛"),
    ("拉肚子", "腹泻"),
    ("心慌", "心悸"),
    ("头晕", "眩晕"),
]


def _normalize_text(text: str) -> str:
    cleaned = (text or "").strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def _estimate_complexity(query: str) -> str:
    q = query or ""
    if len(q) >= 40:
        return "high"
    if any(k in q for k in _COMPLEXITY_KEYWORDS["high"]):
        return "high"
    if len(q) >= 20 or any(k in q for k in _COMPLEXITY_KEYWORDS["medium"]):
        return "medium"
    return "low"


def _adaptive_k(intent: str, complexity: str) -> int:
    default_k = max(1, int(settings.ADAPTIVE_RETRIEVAL_K_DEFAULT))
    if not settings.ENABLE_ADAPTIVE_RETRIEVAL_K:
        return default_k

    low = max(1, int(settings.ADAPTIVE_RETRIEVAL_K_MIN))
    high = max(low, int(settings.ADAPTIVE_RETRIEVAL_K_MAX))

    intent_base = {
        "GREETING": low,
        "REGISTRATION": low,
        "SERVICE_BOOKING": low,
        "INFO": max(low, min(high, default_k)),
        "MEDICAL_CONSULT": max(low, min(high, default_k + 1)),
        "CRISIS": max(low, min(high, default_k + 2)),
    }.get((intent or "").upper(), default_k)

    bump = {"low": 0, "medium": 1, "high": 2}.get(complexity, 0)
    return max(low, min(high, intent_base + bump))


def _rule_rewrite(query: str, intent: str) -> List[str]:
    normalized = _normalize_text(query)
    if not normalized:
        return []

    variants: List[str] = [normalized]

    expanded = normalized
    for src, dst in _RULE_REWRITE_PAIRS:
        if src in expanded and dst not in expanded:
            expanded = expanded.replace(src, dst)
    if expanded != normalized:
        variants.append(expanded)

    if len(normalized) <= 12 and (intent or "").upper() in {"MEDICAL_CONSULT", "CRISIS", ""}:
        variants.append(f"{normalized} 症状 原因 治疗 建议")

    deduped: List[str] = []
    seen = set()
    for item in variants:
        key = _normalize_text(item)
        if key and key not in seen:
            seen.add(key)
            deduped.append(key)
    return deduped


async def _llm_rewrite(query: str) -> Optional[str]:
    try:
        from app.core.models.local_slm import local_slm

        prompt = (
            "你是医疗检索改写器。把用户问题改写成更适合检索的一句话，"
            "保留原意，不要编造新症状。只输出改写句子。\n"
            f"用户问题：{query}\n改写："
        )
        results = await local_slm.generate_batch_async(
            [prompt],
            system_prompt="You are a medical retrieval query rewriter.",
            thinking_mode=False,
        )
        candidate = _normalize_text(results[0] if results else "")
        if not candidate or candidate == _normalize_text(query):
            return None
        return candidate
    except Exception as exc:
        logger.warning("query_rewrite_llm_failed", error=str(exc))
        return None


async def build_retrieval_plan(query: str, intent: str = "") -> RetrievalPlan:
    normalized = _normalize_text(query)
    complexity = _estimate_complexity(normalized)
    top_k = _adaptive_k(intent, complexity)

    variants = [normalized] if normalized else []
    rewrite_source = "none"

    if settings.ENABLE_QUERY_REWRITE and normalized:
        variants = _rule_rewrite(normalized, intent)
        rewrite_source = "rule"

        if settings.QUERY_REWRITE_USE_LLM:
            llm_variant = await _llm_rewrite(normalized)
            if llm_variant:
                variants.append(llm_variant)
                rewrite_source = "rule+llm"

    max_variants = max(1, int(settings.QUERY_REWRITE_MAX_VARIANTS))
    variants = variants[:max_variants]

    primary = variants[0] if variants else normalized
    index_scope = (settings.DEFAULT_RETRIEVAL_INDEX_SCOPE or "paragraph").lower().strip()
    if index_scope not in {"document", "section", "paragraph"}:
        index_scope = "paragraph"

    plan = RetrievalPlan(
        original_query=normalized,
        primary_query=primary,
        query_variants=variants,
        top_k=top_k,
        complexity=complexity,
        rewrite_source=rewrite_source,
        index_scope=index_scope,
    )

    logger.info(
        "retrieval_plan_built",
        query=normalized[:80],
        intent=(intent or "").upper(),
        top_k=top_k,
        complexity=complexity,
        rewrite_source=rewrite_source,
        variants=len(variants),
        index_scope=index_scope,
    )
    return plan
