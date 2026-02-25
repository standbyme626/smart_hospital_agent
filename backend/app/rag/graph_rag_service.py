import asyncio
from typing import Any, Dict, List, Optional

import structlog

from app.db.neo4j_client import neo4j_client
from app.rag.hierarchical_index import hierarchical_index_gateway
from app.rag.retriever import get_retriever

logger = structlog.get_logger(__name__)


class GraphRAGService:
    def __init__(self):
        # Reuse global retriever singleton to avoid duplicate heavy model loads at startup.
        self.vector_retriever = get_retriever()

    async def initialize(self):
        """
        Initialize connections.
        """
        await neo4j_client.connect()
        await neo4j_client.init_schema()

    async def search(
        self,
        query: str,
        extracted_entities: Optional[List[str]] = None,
        top_k: int = 3,
        query_variants: Optional[List[str]] = None,
        index_scope: str = "paragraph",
    ) -> str:
        """
        Hybrid search combining Vector Search (Milvus), Graph Search (Neo4j),
        and optional Hierarchical Index placeholders.
        """

        safe_top_k = max(1, int(top_k))

        # 1. Vector Search (Parallel)
        vector_task = asyncio.create_task(
            self._run_vector_search(query=query, top_k=safe_top_k, query_variants=query_variants)
        )

        # 2. Graph Search (Parallel)
        graph_task = asyncio.create_task(self._run_graph_search(extracted_entities or []))

        # 3. Hierarchical Index (Optional)
        hier_task = asyncio.create_task(
            self._run_hierarchical_search(query=query, index_scope=index_scope, top_k=safe_top_k)
        )

        # [Phase C] Wait for all with robust error handling
        try:
            results = await asyncio.gather(vector_task, graph_task, hier_task, return_exceptions=True)

            vector_res = results[0] if not isinstance(results[0], Exception) else f"Vector search error: {str(results[0])}"
            graph_res = results[1] if not isinstance(results[1], Exception) else f"Graph search error: {str(results[1])}"
            hier_res = results[2] if not isinstance(results[2], Exception) else f"Hierarchical index error: {str(results[2])}"

        except Exception as e:
            logger.error("hybrid_search_crash", error=str(e))
            return "System Error during retrieval."

        # 4. Fusion
        context = f"""
=== Retrieved Medical Literature (Vector DB) ===
{vector_res}

=== Knowledge Graph Context (Neo4j) ===
{graph_res}

=== Hierarchical Index ({index_scope}) ===
{hier_res}
"""
        return context

    async def _run_vector_search(self, query: str, top_k: int = 3, query_variants: Optional[List[str]] = None) -> str:
        try:
            normalized = (query or "").strip()
            candidates: List[str] = []
            seen = set()

            for q in [normalized] + list(query_variants or []):
                key = (q or "").strip()
                if key and key not in seen:
                    seen.add(key)
                    candidates.append(key)

            if not candidates:
                return "No vector query provided."

            per_query_k = max(1, int(top_k))
            tasks = [
                asyncio.create_task(self.vector_retriever.search_rag30(q, top_k=per_query_k))
                for q in candidates
            ]

            search_results = await asyncio.gather(*tasks, return_exceptions=True)

            merged_docs: List[Dict[str, Any]] = []
            doc_seen = set()
            for idx, item in enumerate(search_results):
                if isinstance(item, Exception):
                    logger.warning("vector_variant_failed", query=candidates[idx], error=str(item))
                    continue
                if not isinstance(item, list):
                    continue
                for doc in item:
                    if not isinstance(doc, dict):
                        continue
                    dedup_key = str(doc.get("id") or doc.get("content") or "")
                    if not dedup_key or dedup_key in doc_seen:
                        continue
                    doc_seen.add(dedup_key)
                    merged_docs.append(doc)

            if not merged_docs:
                return "No vector context available."

            merged_docs = sorted(merged_docs, key=lambda x: float(x.get("score", 0.0)), reverse=True)[:top_k]
            return self._format_vector_results(merged_docs)

        except Exception as e:
            logger.error("vector_search_failed", error=str(e))
            return "No vector context available due to error."

    def _format_vector_results(self, docs: List[Dict[str, Any]]) -> str:
        lines = []
        for idx, doc in enumerate(docs, start=1):
            content = str(doc.get("content", "")).strip().replace("\n", " ")
            source = str(doc.get("source", "unknown"))
            score = float(doc.get("score", 0.0))
            if len(content) > 220:
                content = f"{content[:220]}..."
            lines.append(f"{idx}. ({source}, score={score:.3f}) {content}")
        return "\n".join(lines)

    async def _run_hierarchical_search(self, query: str, index_scope: str, top_k: int) -> str:
        hits = await hierarchical_index_gateway.search(
            query=query,
            level=index_scope if index_scope in {"document", "section", "paragraph"} else "paragraph",
            top_k=top_k,
        )
        if not hits:
            return "No hierarchical hits (disabled or empty backend)."

        lines = []
        for i, hit in enumerate(hits[:top_k], start=1):
            text = str(hit.get("text", "")).replace("\n", " ")
            if len(text) > 180:
                text = f"{text[:180]}..."
            lines.append(f"{i}. [{hit.get('level', 'paragraph')}] {text}")
        return "\n".join(lines)

    async def _run_graph_search(self, entities: List[str]) -> str:
        if not entities:
            return "No entities provided for graph search."

        context_parts = []

        # For each entity, find what diseases cause it (if it's a symptom)
        # or what symptoms it has (if it's a disease)
        for entity in entities:
            # Try finding diseases that cause this symptom
            related = await neo4j_client.get_related_entities(
                entity_name=entity,
                entity_type="Symptom",
                relation_type="HAS_SYMPTOM",
                direction="INCOMING",
            )

            if related:
                names = [r["name"] for r in related]
                context_parts.append(f"Symptom '{entity}' is associated with: {', '.join(names)}")

        if not context_parts:
            return "No specific graph connections found for the provided entities."

        return "\n".join(context_parts)


# Singleton
graph_rag_service = GraphRAGService()
