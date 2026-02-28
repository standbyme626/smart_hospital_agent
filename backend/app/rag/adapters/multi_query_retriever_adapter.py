from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Sequence, Tuple

import structlog

from app.rag.adapters.query_expander_adapter import extract_variant_texts

logger = structlog.get_logger(__name__)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except Exception:
        return default
    return parsed


def _doc_key(doc: Dict[str, Any]) -> str:
    return str(doc.get("id") or doc.get("chunk_id") or doc.get("content") or "")


def _snapshot_hit(doc: Dict[str, Any]) -> Dict[str, Any]:
    payload = {
        "id": doc.get("id"),
        "score": _safe_float(doc.get("score"), 0.0),
        "source": doc.get("source"),
    }
    if doc.get("department") is not None:
        payload["department"] = doc.get("department")
    return payload


class MultiQueryRetrieverAdapter:
    def __init__(
        self,
        *,
        retriever: Any,
        fusion_method: str = "weighted_rrf",
        rrf_k: int = 60,
    ) -> None:
        self.retriever = retriever
        self.fusion_method = (fusion_method or "weighted_rrf").strip().lower()
        self.rrf_k = max(1, int(rrf_k))

    def _coerce_variants(
        self,
        *,
        query: str,
        retrieval_query_variants: Optional[Sequence[Any]],
    ) -> List[Dict[str, Any]]:
        original = str(query or "").strip()
        texts = extract_variant_texts(retrieval_query_variants, original_query=original)
        variants: List[Dict[str, Any]] = []
        for idx, text in enumerate(texts):
            variant = {"text": text, "weight": 1.0, "type": "original" if idx == 0 else "synonym"}
            for item in retrieval_query_variants or []:
                if isinstance(item, dict) and str(item.get("text") or "").strip() == text:
                    variant["weight"] = _safe_float(item.get("weight"), 1.0 if idx == 0 else 0.8)
                    variant["type"] = str(item.get("type") or variant["type"])
                    break
            variants.append(variant)
        return variants

    async def _search_one_variant(
        self,
        *,
        query: str,
        top_k: int,
        use_rerank: Optional[bool],
        rerank_threshold: Optional[float],
        pure_mode: Optional[bool],
    ) -> List[Dict[str, Any]]:
        result = await self.retriever.search_rag30(
            query,
            top_k=top_k,
            use_rerank=use_rerank,
            rerank_threshold=rerank_threshold,
            pure_mode=pure_mode,
        )
        return result if isinstance(result, list) else []

    def _concat_merge(
        self,
        *,
        ranked_lists: Dict[str, List[Dict[str, Any]]],
        top_k: int,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, float]]]:
        merged: List[Dict[str, Any]] = []
        seen = set()
        contributors: Dict[str, Dict[str, float]] = {}
        for variant_text, docs in ranked_lists.items():
            for rank, doc in enumerate(docs):
                key = _doc_key(doc)
                if not key or key in seen:
                    continue
                seen.add(key)
                merged_doc = dict(doc)
                merged_doc["score"] = _safe_float(doc.get("score"), 0.0)
                merged.append(merged_doc)
                contributors[key] = {variant_text: 1.0 / (rank + 1)}
                if len(merged) >= top_k:
                    return merged, contributors
        return merged, contributors

    def _rrf_merge(
        self,
        *,
        ranked_lists: Dict[str, List[Dict[str, Any]]],
        variant_weights: Dict[str, float],
        top_k: int,
        weighted: bool,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, float]]]:
        fused: Dict[str, Dict[str, Any]] = {}
        contributors: Dict[str, Dict[str, float]] = {}
        for variant_text, docs in ranked_lists.items():
            weight = variant_weights.get(variant_text, 1.0) if weighted else 1.0
            for rank, doc in enumerate(docs):
                key = _doc_key(doc)
                if not key:
                    continue
                contribution = weight / (self.rrf_k + rank + 1)
                if key not in fused:
                    merged_doc = dict(doc)
                    merged_doc["score"] = 0.0
                    merged_doc["source"] = "multi_query_fused"
                    fused[key] = merged_doc
                    contributors[key] = {}
                fused[key]["score"] = _safe_float(fused[key].get("score"), 0.0) + contribution
                contributors[key][variant_text] = contributors[key].get(variant_text, 0.0) + contribution

                # 保留更好的原始分数用于可读性展示
                if _safe_float(doc.get("score"), 0.0) > _safe_float(fused[key].get("raw_score"), 0.0):
                    fused[key]["raw_score"] = _safe_float(doc.get("score"), 0.0)

        merged = sorted(fused.values(), key=lambda item: _safe_float(item.get("score"), 0.0), reverse=True)
        return merged[:top_k], contributors

    def _build_source_ratio(
        self,
        *,
        docs: List[Dict[str, Any]],
        contributors: Dict[str, Dict[str, float]],
        original_variant_text: str,
    ) -> Dict[str, Any]:
        original_count = 0
        expanded_count = 0
        for doc in docs:
            key = _doc_key(doc)
            source_scores = contributors.get(key, {})
            if not source_scores:
                original_count += 1
                continue
            dominant_variant = max(source_scores.items(), key=lambda item: item[1])[0]
            if dominant_variant == original_variant_text:
                original_count += 1
            else:
                expanded_count += 1

        total = max(1, len(docs))
        return {
            "original": round(original_count / total, 4),
            "expanded": round(expanded_count / total, 4),
            "original_count": original_count,
            "expanded_count": expanded_count,
        }

    async def retrieve(
        self,
        *,
        query: str,
        retrieval_query_variants: Optional[Sequence[Any]],
        top_k: int,
        enable_multi_query: bool,
        original_only: bool,
        use_rerank: Optional[bool],
        rerank_threshold: Optional[float],
        pure_mode: Optional[bool],
        fusion_method: Optional[str] = None,
    ) -> Dict[str, Any]:
        variants = self._coerce_variants(query=query, retrieval_query_variants=retrieval_query_variants)
        if not variants:
            return {
                "fused_docs": [],
                "variant_hits_map": {},
                "topk_source_ratio": {"original": 1.0, "expanded": 0.0, "original_count": 0, "expanded_count": 0},
                "fusion_method": "original_only",
            }

        effective_method = (fusion_method or self.fusion_method or "weighted_rrf").strip().lower()
        if effective_method not in {"weighted_rrf", "rrf", "concat_merge"}:
            effective_method = "weighted_rrf"

        selected_variants = variants if enable_multi_query and not original_only else variants[:1]
        if len(selected_variants) == 1:
            effective_method = "original_only"

        tasks = [
            asyncio.create_task(
                self._search_one_variant(
                    query=item["text"],
                    top_k=max(1, int(top_k)),
                    use_rerank=use_rerank,
                    rerank_threshold=rerank_threshold,
                    pure_mode=pure_mode,
                )
            )
            for item in selected_variants
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        ranked_lists: Dict[str, List[Dict[str, Any]]] = {}
        variant_hits_map: Dict[str, List[Dict[str, Any]]] = {}
        variant_weights: Dict[str, float] = {}
        for idx, item in enumerate(results):
            variant = selected_variants[idx]
            variant_text = variant["text"]
            variant_weights[variant_text] = _safe_float(variant.get("weight"), 1.0)
            if isinstance(item, Exception):
                logger.warning("multi_query_variant_failed", query=variant_text, error=str(item))
                ranked_lists[variant_text] = []
                variant_hits_map[variant_text] = []
                continue
            docs = item if isinstance(item, list) else []
            ranked_lists[variant_text] = docs
            variant_hits_map[variant_text] = [_snapshot_hit(doc) for doc in docs[: max(1, int(top_k))]]

        if effective_method == "concat_merge":
            fused_docs, contributors = self._concat_merge(ranked_lists=ranked_lists, top_k=max(1, int(top_k)))
        elif effective_method == "rrf":
            fused_docs, contributors = self._rrf_merge(
                ranked_lists=ranked_lists,
                variant_weights=variant_weights,
                top_k=max(1, int(top_k)),
                weighted=False,
            )
        else:
            fused_docs, contributors = self._rrf_merge(
                ranked_lists=ranked_lists,
                variant_weights=variant_weights,
                top_k=max(1, int(top_k)),
                weighted=True,
            )

        original_variant_text = selected_variants[0]["text"]
        topk_source_ratio = self._build_source_ratio(
            docs=fused_docs,
            contributors=contributors,
            original_variant_text=original_variant_text,
        )

        logger.info(
            "multi_query_retriever_result",
            query=query[:80],
            variant_count=len(selected_variants),
            fusion_method=effective_method,
            top_k=top_k,
            original_ratio=topk_source_ratio.get("original"),
            expanded_ratio=topk_source_ratio.get("expanded"),
        )
        return {
            "fused_docs": fused_docs,
            "variant_hits_map": variant_hits_map,
            "topk_source_ratio": topk_source_ratio,
            "fusion_method": effective_method,
        }
