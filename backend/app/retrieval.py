from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import jieba

from .config import Settings
from .db import Database
from .model_gateway import ModelGateway
from .policy import normalize_department

CJK = re.compile(r"[\u4e00-\u9fff]")
ASCII = re.compile(r"[a-zA-Z0-9_]+")
PUNCT = re.compile(r"^[\W_]+$", re.UNICODE)


def tokenize(text: str) -> str:
    tokens: list[str] = []
    for token in jieba.cut(text or ""):
        token = token.strip().lower()
        if not token or PUNCT.match(token):
            continue
        tokens.append(token)
    tokens.extend(token.lower() for token in ASCII.findall(text or ""))
    seen: list[str] = []
    for token in tokens:
        if token not in seen:
            seen.append(token)
    return " ".join(seen)


def vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(f"{value:.6f}" for value in vector) + "]"


def rrf_fuse(
    ranked_lists: Iterable[list[str]],
    *,
    k: int = 60,
    weights: list[float] | None = None,
) -> list[tuple[str, float]]:
    lists = [list(items) for items in ranked_lists]
    if weights is None:
        weights = [1.0] * len(lists)
    scores: dict[str, float] = defaultdict(float)
    for weight, items in zip(weights, lists):
        for rank, item in enumerate(items):
            scores[item] += weight / (k + rank + 1)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


class RetrievalService:
    def __init__(self, settings: Settings, db: Database, gateway: ModelGateway):
        self.settings = settings
        self.db = db
        self.gateway = gateway
        self.last_filtered = False
        self.last_dept_unavailable = False

    async def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        department: str | None = None,
    ) -> list[dict[str, Any]]:
        top_k = top_k or self.settings.RETRIEVAL_TOP_K
        self.last_filtered = False
        self.last_dept_unavailable = False
        if not query.strip():
            return []
        normalized = normalize_department(department) if department else None
        query_vector = (await self.gateway.embed([query]))[0]
        dense = await self._dense(query_vector, normalized)
        sparse = await self._sparse(query, normalized)
        if normalized and not dense and not sparse:
            self.last_dept_unavailable = True
            dense = await self._dense(query_vector, None)
            sparse = await self._sparse(query, None)
        elif normalized:
            self.last_filtered = True
        fused = rrf_fuse(
            [[row["id"] for row in dense], [row["id"] for row in sparse]],
            k=60,
            weights=[1.0, 0.7],
        )
        by_id: dict[str, dict[str, Any]] = {row["id"]: row for row in dense}
        by_id.update({row["id"]: row for row in sparse if row["id"] not in by_id})
        candidates = [by_id[doc_id] for doc_id, _ in fused[: self.settings.RERANK_TOP_N] if doc_id in by_id]
        if not candidates:
            return []
        reranked = await self.gateway.rerank(
            query,
            [f"{row['title']}\n{row['content'][:1200]}" for row in candidates],
            top_n=min(self.settings.RERANK_TOP_N, len(candidates)),
        )
        results: list[dict[str, Any]] = []
        for index, score in reranked[:top_k]:
            if index >= len(candidates):
                continue
            row = candidates[index]
            results.append(
                {
                    "id": row["id"],
                    "doc_key": row["doc_key"],
                    "title": row["title"],
                    "snippet": row["content"][:400],
                    "department": row["department"],
                    "source": row["source"],
                    "score": round(float(score), 4),
                }
            )
        return results

    async def _dense(self, vector: list[float], department: str | None) -> list[dict[str, Any]]:
        literal = vector_literal(vector)
        if department:
            rows = await self.db.fetch(
                """
                SELECT id, doc_key, title, content, department, source,
                       1 - (embedding <=> $1::vector) AS score
                FROM documents
                WHERE department = $2
                ORDER BY embedding <=> $1::vector
                LIMIT $3
                """,
                literal,
                department,
                self.settings.RETRIEVAL_DENSE_K,
            )
        else:
            rows = await self.db.fetch(
                """
                SELECT id, doc_key, title, content, department, source,
                       1 - (embedding <=> $1::vector) AS score
                FROM documents
                ORDER BY embedding <=> $1::vector
                LIMIT $2
                """,
                literal,
                self.settings.RETRIEVAL_DENSE_K,
            )
        return [dict(row) for row in rows]

    async def _sparse(self, query: str, department: str | None) -> list[dict[str, Any]]:
        tokens = tokenize(query)
        if not tokens:
            return []
        if department:
            rows = await self.db.fetch(
                """
                SELECT id, doc_key, title, content, department, source,
                       ts_rank_cd(tsv, plainto_tsquery('simple', $1)) AS score
                FROM documents
                WHERE department = $2
                  AND tsv @@ plainto_tsquery('simple', $1)
                ORDER BY score DESC
                LIMIT $3
                """,
                tokens,
                department,
                self.settings.RETRIEVAL_SPARSE_K,
            )
        else:
            rows = await self.db.fetch(
                """
                SELECT id, doc_key, title, content, department, source,
                       ts_rank_cd(tsv, plainto_tsquery('simple', $1)) AS score
                FROM documents
                WHERE tsv @@ plainto_tsquery('simple', $1)
                ORDER BY score DESC
                LIMIT $2
                """,
                tokens,
                self.settings.RETRIEVAL_SPARSE_K,
            )
        return [dict(row) for row in rows]

    async def ingest(self, rows: list[dict[str, Any]], batch_size: int = 8) -> int:
        inserted = 0
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            texts = [f"{row['title']}\n{row['content']}" for row in batch]
            vectors = await self.gateway.embed(texts)
            for row, vector in zip(batch, vectors):
                tokens = tokenize(f"{row['title']} {row['content']}")
                await self.db.execute(
                    """
                    INSERT INTO documents (doc_key, title, content, department, source, embedding, tsv, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6::vector, to_tsvector('simple', $7), $8::jsonb)
                    ON CONFLICT (doc_key) DO UPDATE SET
                        title = EXCLUDED.title,
                        content = EXCLUDED.content,
                        department = EXCLUDED.department,
                        source = EXCLUDED.source,
                        embedding = EXCLUDED.embedding,
                        tsv = EXCLUDED.tsv,
                        metadata = EXCLUDED.metadata
                    """,
                    row["doc_key"],
                    row["title"],
                    row["content"],
                    row.get("department"),
                    row.get("source", "unknown"),
                    vector_literal(vector),
                    tokens,
                    json.dumps(row.get("metadata", {}), ensure_ascii=False),
                )
                inserted += 1
        return inserted

    async def seed_from_jsonl(self, path: Path, limit: int | None = None) -> int:
        limit = limit or self.settings.SEED_LIMIT
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                item = json.loads(line)
                title = str(item.get("question") or item.get("title") or "").strip()
                content = str(item.get("vector_text") or item.get("content") or item.get("answer") or "").strip()
                if not title or not content:
                    continue
                doc_key = hashlib.sha256((title + "\n" + content[:500]).encode("utf-8")).hexdigest()[:32]
                rows.append(
                    {
                        "doc_key": doc_key,
                        "title": title,
                        "content": content,
                        "department": normalize_department(item.get("department") or item.get("cate1")),
                        "source": item.get("source") or "medical_gold",
                        "metadata": {"raw_keys": sorted(item.keys())},
                    }
                )
                if len(rows) >= limit:
                    break
        return await self.ingest(rows)

    async def stats(self) -> dict[str, Any]:
        total = await self.db.fetchval("SELECT count(*) FROM documents")
        departments = await self.db.fetch(
            "SELECT department, count(*) AS count FROM documents GROUP BY department ORDER BY count DESC LIMIT 30"
        )
        return {
            "documents": int(total or 0),
            "departments": [dict(row) for row in departments],
        }
