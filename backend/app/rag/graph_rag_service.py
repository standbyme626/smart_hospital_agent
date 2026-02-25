import asyncio
import structlog
from typing import List, Dict, Any

from app.rag.retriever import get_retriever
from app.db.neo4j_client import neo4j_client

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

    async def search(self, query: str, extracted_entities: List[str] = None) -> str:
        """
        Hybrid search combining Vector Search (Milvus) and Graph Search (Neo4j).
        """
        
        # 1. Vector Search (Parallel)
        vector_task = asyncio.create_task(self._run_vector_search(query))
        
        # 2. Graph Search (Parallel)
        graph_task = asyncio.create_task(self._run_graph_search(extracted_entities))
        
        # [Phase 4] Wait for both with robust error handling
        try:
            results = await asyncio.gather(vector_task, graph_task, return_exceptions=True)
            
            vector_res = results[0] if not isinstance(results[0], Exception) else f"Vector search error: {str(results[0])}"
            graph_res = results[1] if not isinstance(results[1], Exception) else f"Graph search error: {str(results[1])}"
            
        except Exception as e:
            logger.error("hybrid_search_crash", error=str(e))
            return "System Error during retrieval."
        
        # 3. Fusion
        context = f"""
        === Retrieved Medical Literature (Vector DB) ===
        {vector_res}
        
        === Knowledge Graph Context (Neo4j) ===
        {graph_res}
        """
        return context

    async def _run_vector_search(self, query: str) -> str:
        try:
            # Reusing existing retriever logic
            # Note: MedicalRetriever.search_rag30 is ASYNC in the error log ("coroutine ... was never awaited")
            # This contradicts the usual naming convention but let's fix it.
            # If it is a coroutine, we should await it directly, not wrap in to_thread.
            
            # Check if it is async by calling it and checking if it is awaitable
            # But here we are inside an async function, so we can just await it.
            
            # However, in previous turn I saw `def search_rag30(self, query, top_k=3):` in `medical_tools.py`'s `lookup_guideline` 
            # calling `retriever.search_rag30(query, top_k=3)`. 
            # Wait, `medical_tools.py` calls it synchronously.
            # But the error log says: `RuntimeWarning: coroutine 'MedicalRetriever.search_rag30' was never awaited`
            # This implies `search_rag30` IS async defined in `MedicalRetriever`.
            
            # Let's assume it IS async.
            results = await self.vector_retriever.search_rag30(query, top_k=3)
                 
            # Format results
            if isinstance(results, list):
                return "\n".join([f"- {r}" for r in results])
            return str(results)
            
        except Exception as e:
            logger.error("Vector search failed", error=str(e))
            return "No vector context available due to error."

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
                direction="INCOMING"
            )
            
            if related:
                names = [r['name'] for r in related]
                context_parts.append(f"Symptom '{entity}' is associated with: {', '.join(names)}")
            
            # Also try finding drugs that treat this (if it's a disease) - simplified logic
            # Or if it's a disease, find symptoms
            # We can try a generic search if we don't know the type.
            # For Phase 2, let's stick to Symptom -> Disease mapping as primary value add.
            
        if not context_parts:
            return "No specific graph connections found for the provided entities."
            
        return "\n".join(context_parts)

# Singleton
graph_rag_service = GraphRAGService()
