from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _as_text(value: Any) -> str:
    return str(value or "").strip()


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except Exception:
        return default
    return parsed


def _as_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


@dataclass
class _EvidenceDoc:
    doc_id: str
    chunk_id: str
    source_id: str
    split_id: str
    split_idx: Optional[int]
    score: float
    content: str
    source_type: str
    parent_id: str
    child_ids: List[str]

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "source_id": self.source_id,
            "score": round(self.score, 6),
            "content": self.content,
            "source_type": self.source_type,
        }
        if self.split_id:
            payload["split_id"] = self.split_id
        if self.split_idx is not None:
            payload["split_idx"] = self.split_idx
        if self.parent_id:
            payload["parent_id"] = self.parent_id
        if self.child_ids:
            payload["child_ids"] = self.child_ids
        return payload


class ContextWindowAdapter:
    def __init__(
        self,
        *,
        window_size: int = 1,
        max_evidence: int = 6,
        max_per_source: int = 2,
        ordering_strategy: str = "score_desc",
        enable_window: bool = True,
        enable_merge: bool = True,
        enable_diversity: bool = True,
        max_context_chars: int = 3200,
    ) -> None:
        self.window_size = max(0, int(window_size))
        self.max_evidence = max(1, int(max_evidence))
        self.max_per_source = max(1, int(max_per_source))
        normalized_ordering = _as_text(ordering_strategy).lower() or "score_desc"
        if normalized_ordering not in {"score_desc", "lost_in_middle_mitigate"}:
            normalized_ordering = "score_desc"
        self.ordering_strategy = normalized_ordering
        self.enable_window = bool(enable_window)
        self.enable_merge = bool(enable_merge)
        self.enable_diversity = bool(enable_diversity)
        self.max_context_chars = max(400, int(max_context_chars))

    def build_context_pack(
        self,
        *,
        docs: Optional[Sequence[Dict[str, Any]]],
        fusion_method: str,
        fallback_context: str = "",
        ordering_strategy: Optional[str] = None,
    ) -> Dict[str, Any]:
        ordering = _as_text(ordering_strategy).lower() or self.ordering_strategy
        if ordering not in {"score_desc", "lost_in_middle_mitigate"}:
            ordering = self.ordering_strategy

        normalized = self._normalize_docs(docs or [])
        expanded = self._apply_window(normalized)
        merged = self._apply_merge(expanded)
        diversified = self._apply_diversity(merged)
        ordered = self._apply_ordering(diversified, ordering)
        evidence_payload, truncation = self._truncate(ordered)

        if not evidence_payload:
            fallback_text = _as_text(fallback_context)
            if fallback_text:
                evidence_payload = [
                    {
                        "doc_id": "fallback_context",
                        "chunk_id": "fallback_context",
                        "source_id": "fallback",
                        "score": 0.0,
                        "content": fallback_text,
                        "source_type": "fallback",
                    }
                ]

        return {
            "evidence": evidence_payload,
            "ordering": ordering,
            "fusion_method": _as_text(fusion_method) or "weighted_rrf",
            "truncation": truncation,
        }

    def render_context(self, *, context_pack: Dict[str, Any], fallback_context: str = "") -> str:
        evidence = context_pack.get("evidence") if isinstance(context_pack, dict) else None
        if not isinstance(evidence, list) or not evidence:
            return _as_text(fallback_context)

        lines = ["[Reorganized Evidence]"]
        for idx, item in enumerate(evidence, start=1):
            if not isinstance(item, dict):
                continue
            doc_id = _as_text(item.get("doc_id") or item.get("chunk_id") or "unknown")
            chunk_id = _as_text(item.get("chunk_id") or doc_id)
            score = _as_float(item.get("score"), 0.0)
            content = _as_text(item.get("content"))
            if len(content) > 260:
                content = f"{content[:260]}..."
            lines.append(f"{idx}. [doc={doc_id} chunk={chunk_id} score={score:.3f}] {content}")

        fallback = _as_text(fallback_context)
        if fallback:
            lines.append("\n[Original Hybrid Context]")
            lines.append(fallback)
        return "\n".join(lines)

    def _normalize_docs(self, docs: Sequence[Dict[str, Any]]) -> List[_EvidenceDoc]:
        normalized: List[_EvidenceDoc] = []
        seen = set()
        for raw in docs:
            if not isinstance(raw, dict):
                continue
            content = _as_text(raw.get("content"))
            if not content:
                continue

            metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
            doc_id = _as_text(raw.get("doc_id") or raw.get("id") or raw.get("chunk_id") or metadata.get("doc_id"))
            chunk_id = _as_text(raw.get("chunk_id") or raw.get("id") or metadata.get("chunk_id") or doc_id)
            source_id = _as_text(
                raw.get("source_id")
                or metadata.get("source_id")
                or raw.get("doc_id")
                or metadata.get("doc_id")
                or raw.get("source")
                or doc_id
            )
            split_id = _as_text(raw.get("split_id") or metadata.get("split_id") or chunk_id)
            split_idx = _as_int(raw.get("split_idx") if raw.get("split_idx") is not None else metadata.get("split_idx"))
            parent_id = _as_text(raw.get("parent_id") or metadata.get("parent_id"))
            child_ids_raw = raw.get("child_ids") if isinstance(raw.get("child_ids"), list) else metadata.get("child_ids")
            child_ids = [_as_text(item) for item in child_ids_raw or [] if _as_text(item)]
            source_type = _as_text(raw.get("source_type") or metadata.get("source_type") or raw.get("source") or "unknown")
            score = _as_float(raw.get("score"), 0.0)

            key = (doc_id or chunk_id, content)
            if key in seen:
                continue
            seen.add(key)

            normalized.append(
                _EvidenceDoc(
                    doc_id=doc_id or chunk_id or f"doc_{len(normalized)+1}",
                    chunk_id=chunk_id or doc_id or f"chunk_{len(normalized)+1}",
                    source_id=source_id or doc_id or f"source_{len(normalized)+1}",
                    split_id=split_id,
                    split_idx=split_idx,
                    score=score,
                    content=content,
                    source_type=source_type,
                    parent_id=parent_id,
                    child_ids=child_ids,
                )
            )

        return sorted(normalized, key=lambda item: item.score, reverse=True)

    def _apply_window(self, docs: Sequence[_EvidenceDoc]) -> List[_EvidenceDoc]:
        if not self.enable_window or self.window_size <= 0 or not docs:
            return list(docs)

        grouped: Dict[Tuple[str, str], List[_EvidenceDoc]] = {}
        for item in docs:
            if item.split_idx is None:
                continue
            grouped.setdefault((item.source_id, item.split_id), []).append(item)

        for values in grouped.values():
            values.sort(key=lambda item: item.split_idx if item.split_idx is not None else 0)

        selected: Dict[Tuple[str, str], _EvidenceDoc] = {(item.doc_id, item.chunk_id): item for item in docs}

        for item in docs:
            if item.split_idx is None:
                continue
            key = (item.source_id, item.split_id)
            siblings = grouped.get(key, [])
            if not siblings:
                continue
            for sibling in siblings:
                if sibling.split_idx is None:
                    continue
                if abs(sibling.split_idx - item.split_idx) <= self.window_size:
                    selected[(sibling.doc_id, sibling.chunk_id)] = sibling

        return list(selected.values())

    def _apply_merge(self, docs: Sequence[_EvidenceDoc]) -> List[_EvidenceDoc]:
        if not self.enable_merge or not docs:
            return list(docs)

        id_map: Dict[str, _EvidenceDoc] = {}
        for item in docs:
            id_map[item.chunk_id] = item
            id_map[item.doc_id] = item

        merged_output: List[_EvidenceDoc] = []
        consumed = set()

        for item in docs:
            if item.chunk_id in consumed:
                continue

            merge_members = [item]
            parent_key = item.parent_id
            if parent_key and parent_key in id_map:
                merge_members.append(id_map[parent_key])
            for child_key in item.child_ids:
                if child_key in id_map:
                    merge_members.append(id_map[child_key])

            unique_members: List[_EvidenceDoc] = []
            member_seen = set()
            for member in merge_members:
                marker = (member.doc_id, member.chunk_id)
                if marker in member_seen:
                    continue
                member_seen.add(marker)
                unique_members.append(member)

            if len(unique_members) <= 1:
                merged_output.append(item)
                continue

            merged_content_parts = [_as_text(member.content) for member in unique_members if _as_text(member.content)]
            merged_content = "\n".join(merged_content_parts)
            merged_doc = _EvidenceDoc(
                doc_id=item.doc_id,
                chunk_id=item.chunk_id,
                source_id=item.source_id,
                split_id=item.split_id,
                split_idx=item.split_idx,
                score=max(member.score for member in unique_members),
                content=merged_content,
                source_type=item.source_type,
                parent_id=item.parent_id,
                child_ids=item.child_ids,
            )
            merged_output.append(merged_doc)
            consumed.add(item.chunk_id)

        return merged_output

    def _apply_diversity(self, docs: Sequence[_EvidenceDoc]) -> List[_EvidenceDoc]:
        if not self.enable_diversity or not docs:
            return list(docs)

        source_count: Dict[str, int] = {}
        diversified: List[_EvidenceDoc] = []
        for item in sorted(docs, key=lambda row: row.score, reverse=True):
            source_key = item.source_id or item.doc_id
            current = source_count.get(source_key, 0)
            if current >= self.max_per_source:
                continue
            source_count[source_key] = current + 1
            diversified.append(item)
        return diversified

    def _apply_ordering(self, docs: Sequence[_EvidenceDoc], ordering: str) -> List[_EvidenceDoc]:
        ranked = sorted(docs, key=lambda row: row.score, reverse=True)
        if ordering != "lost_in_middle_mitigate" or len(ranked) <= 2:
            return ranked

        head = 0
        tail = len(ranked) - 1
        output: List[_EvidenceDoc] = []
        while head <= tail:
            output.append(ranked[head])
            if head != tail:
                output.append(ranked[tail])
            head += 1
            tail -= 1
        return output

    def _truncate(self, docs: Sequence[_EvidenceDoc]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        payload: List[Dict[str, Any]] = []
        total_chars = 0
        applied = False

        for item in docs:
            if len(payload) >= self.max_evidence:
                applied = True
                break
            content = _as_text(item.content)
            if not content:
                continue
            next_chars = total_chars + len(content)
            if next_chars > self.max_context_chars:
                remain = max(0, self.max_context_chars - total_chars)
                if remain > 120:
                    clipped = f"{content[:remain]}..."
                    clipped_item = _EvidenceDoc(
                        doc_id=item.doc_id,
                        chunk_id=item.chunk_id,
                        source_id=item.source_id,
                        split_id=item.split_id,
                        split_idx=item.split_idx,
                        score=item.score,
                        content=clipped,
                        source_type=item.source_type,
                        parent_id=item.parent_id,
                        child_ids=item.child_ids,
                    )
                    payload.append(clipped_item.to_dict())
                applied = True
                break

            payload.append(item.to_dict())
            total_chars = next_chars

        reason = "token_limit" if applied else "none"
        return payload, {"applied": applied, "reason": reason, "max_chars": self.max_context_chars}
