from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence

import structlog

logger = structlog.get_logger(__name__)


DEFAULT_REWRITE_TYPE_BUDGET: Dict[str, int] = {
    "synonym": 2,
    "typo_fix": 1,
    "llm_expand": 1,
}

SYNONYM_REWRITE_RULES = [
    ("胃疼", "胃痛"),
    ("拉肚子", "腹泻"),
    ("心慌", "心悸"),
    ("头晕", "眩晕"),
]

TYPO_FIX_RULES = [
    ("拉杜子", "拉肚子"),
    ("頭晕", "头晕"),
    ("胃同", "胃痛"),
]


def _normalize_text(text: Any) -> str:
    value = str(text or "").strip()
    return re.sub(r"\s+", " ", value)


def _truncate_text(text: str, max_len: int) -> str:
    if max_len <= 0:
        return text
    if len(text) <= max_len:
        return text
    return text[:max_len].strip()


def _detect_lang(text: str) -> str:
    has_zh = bool(re.search(r"[\u4e00-\u9fff]", text))
    has_en = bool(re.search(r"[A-Za-z]", text))
    if has_zh and has_en:
        return "mixed"
    if has_zh:
        return "zh"
    if has_en:
        return "en"
    return "mixed"


def _safe_weight(value: Any, default: float = 0.8) -> float:
    try:
        parsed = float(value)
    except Exception:
        return default
    return round(min(max(parsed, 0.0), 1.5), 4)


def _parse_budget(raw_budget: Any) -> Dict[str, int]:
    budget = dict(DEFAULT_REWRITE_TYPE_BUDGET)
    if isinstance(raw_budget, dict):
        source = raw_budget
    elif isinstance(raw_budget, str) and raw_budget.strip():
        source = None
        try:
            loaded = json.loads(raw_budget)
            if isinstance(loaded, dict):
                source = loaded
        except Exception:
            source = None
        if source is None:
            pairs: Dict[str, int] = {}
            for item in raw_budget.split(","):
                if ":" not in item:
                    continue
                key, value = item.split(":", 1)
                pairs[key.strip()] = value.strip()
            source = pairs
    else:
        return budget

    if not isinstance(source, dict):
        return budget

    for key in DEFAULT_REWRITE_TYPE_BUDGET:
        value = source.get(key)
        try:
            parsed = int(value)
        except Exception:
            continue
        budget[key] = max(0, parsed)
    return budget


def _infer_variant_type(original_query: str, variant_text: str) -> str:
    if variant_text == original_query:
        return "original"
    if len(variant_text) >= max(len(original_query) + 8, 24):
        return "llm_expand"
    return "synonym"


def extract_variant_texts(
    variants: Optional[Sequence[Any]],
    *,
    original_query: Optional[str] = None,
) -> List[str]:
    deduped: List[str] = []
    seen = set()

    def add_text(value: Any) -> None:
        text = _normalize_text(value)
        if text and text not in seen:
            seen.add(text)
            deduped.append(text)

    if original_query:
        add_text(original_query)
    for item in variants or []:
        if isinstance(item, dict):
            add_text(item.get("text"))
        else:
            add_text(item)
    return deduped


class QueryExpanderAdapter:
    def __init__(
        self,
        *,
        max_variants: int = 4,
        max_query_len_per_variant: int = 120,
        rewrite_type_budget: Any = None,
    ) -> None:
        self.max_variants = max(1, int(max_variants))
        self.max_query_len_per_variant = max(1, int(max_query_len_per_variant))
        self.rewrite_type_budget = _parse_budget(rewrite_type_budget)

    def _build_variant(
        self,
        *,
        text: str,
        variant_type: str,
        source: str,
        weight: float,
    ) -> Dict[str, Any]:
        normalized = _truncate_text(_normalize_text(text), self.max_query_len_per_variant)
        return {
            "text": normalized,
            "type": variant_type,
            "source": source,
            "weight": _safe_weight(weight),
            "lang": _detect_lang(normalized),
        }

    def expand(
        self,
        *,
        query: str,
        planned_variants: Optional[Sequence[Any]] = None,
        enable_query_expansion: bool = False,
        original_only: bool = False,
    ) -> List[Dict[str, Any]]:
        original = _truncate_text(_normalize_text(query), self.max_query_len_per_variant)
        if not original:
            return []

        variants: List[Dict[str, Any]] = []
        seen = set()
        used_budget = {key: 0 for key in self.rewrite_type_budget}

        def add_variant(item: Dict[str, Any]) -> None:
            text = _normalize_text(item.get("text"))
            if not text or text in seen:
                return
            v_type = str(item.get("type") or "synonym")
            if v_type != "original":
                quota = self.rewrite_type_budget.get(v_type, 0)
                if used_budget.get(v_type, 0) >= quota:
                    return
                used_budget[v_type] = used_budget.get(v_type, 0) + 1
            seen.add(text)
            variants.append(item)

        add_variant(
            self._build_variant(
                text=original,
                variant_type="original",
                source="rule",
                weight=1.0,
            )
        )

        if original_only or not enable_query_expansion:
            return variants[:1]

        for item in planned_variants or []:
            if len(variants) >= self.max_variants:
                break
            if isinstance(item, dict):
                candidate = _normalize_text(item.get("text"))
                variant_type = str(item.get("type") or _infer_variant_type(original, candidate))
                source = str(item.get("source") or "rule")
                weight = _safe_weight(item.get("weight"), default=0.8)
            else:
                candidate = _normalize_text(item)
                variant_type = _infer_variant_type(original, candidate)
                source = "rule"
                weight = 0.8 if variant_type != "llm_expand" else 0.7
            if not candidate:
                continue
            add_variant(
                self._build_variant(
                    text=candidate,
                    variant_type=variant_type,
                    source=source,
                    weight=weight,
                )
            )

        if len(variants) < self.max_variants:
            for src, dst in TYPO_FIX_RULES:
                if len(variants) >= self.max_variants:
                    break
                if src in original:
                    add_variant(
                        self._build_variant(
                            text=original.replace(src, dst),
                            variant_type="typo_fix",
                            source="rule",
                            weight=0.88,
                        )
                    )

        if len(variants) < self.max_variants:
            for src, dst in SYNONYM_REWRITE_RULES:
                if len(variants) >= self.max_variants:
                    break
                if src in original and dst not in original:
                    add_variant(
                        self._build_variant(
                            text=original.replace(src, dst),
                            variant_type="synonym",
                            source="rule",
                            weight=0.84,
                        )
                    )

        if len(variants) < self.max_variants and self.rewrite_type_budget.get("llm_expand", 0) > 0:
            add_variant(
                self._build_variant(
                    text=f"{original} 症状 原因 治疗",
                    variant_type="llm_expand",
                    source="rule",
                    weight=0.68,
                )
            )

        expanded = variants[: self.max_variants]
        logger.info(
            "query_expander_result",
            original=original[:80],
            variant_count=len(expanded),
            max_variants=self.max_variants,
            max_query_len_per_variant=self.max_query_len_per_variant,
            rewrite_type_budget=self.rewrite_type_budget,
            enable_query_expansion=enable_query_expansion,
            original_only=original_only,
        )
        return expanded
