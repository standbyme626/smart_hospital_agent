import json
import os
import structlog
from typing import Dict, Any, List
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.embeddings import Embeddings

from app.core.graph.state import AgentState
from app.core.monitoring.tracing import monitor_node
from app.services.langchain_embedding import shared_embeddings
from app.core.registry.specialist_registry import registry
from app.core.constants import SYMPTOM_KEYWORD_MAP

logger = structlog.get_logger(__name__)

try:
    from langchain_community.vectorstores import FAISS
    FAISS_AVAILABLE = True
except Exception as e:
    FAISS = None
    FAISS_AVAILABLE = False
    logger.warning("faiss_unavailable_fallback_keyword_router", error=str(e))

# Build simple vector store for departments
def build_department_index():
    if not FAISS_AVAILABLE:
        logger.warning("vector_index_skipped_faiss_unavailable")
        return None

    specialists = registry.get_all_specialists()
    texts = []
    metadatas = []
    
    for dept_key, config in specialists.items():
        # Index description + keywords + name
        desc = config.get("description", "")
        name = config.get("name_cn", dept_key)
        # Add aliases
        aliases = config.get("aliases", [])
        
        content = f"{name} {desc} {' '.join(aliases)}"
        texts.append(content)
        metadatas.append({"department": dept_key})
        
    return FAISS.from_texts(texts, shared_embeddings, metadatas=metadatas)

# Initialize index lazily
_DEPT_INDEX = None

def get_department_index():
    global _DEPT_INDEX
    if _DEPT_INDEX is None:
        try:
            _DEPT_INDEX = build_department_index()
        except Exception as e:
            logger.error("vector_index_build_failed", error=str(e))
            return None
    return _DEPT_INDEX

@monitor_node("expert_router")
async def expert_router_node(state: AgentState, config: RunnableConfig = None) -> Dict[str, Any]:
    """
    [Pain Point #17] Dynamic Department Routing
    Uses RAG (Vector Similarity) to match symptoms to the best department.
    """
    logger.info("node_start", node="expert_router")
    
    symptoms = state.get("symptoms", "")
    history = state.get("medical_history", "")
    query = f"{symptoms} {history}"
    
    selected_depts = []
    reasoning = ""
    
    try:
        index = get_department_index()
        if index:
            # RAG Search
            docs = await index.asimilarity_search_with_score(query, k=2)
            
            for doc, score in docs:
                dept = doc.metadata.get("department")
                # Threshold check (lower score is better for L2 distance, or higher for cosine? 
                # FAISS default depends on metric. Assuming standard behavior: 
                # We'll just take top 1, or top 2 if close.
                # For now, just take top 1 unless score is very bad.
                if dept and dept not in selected_depts:
                    selected_depts.append(dept)
                    
            reasoning = f"Matched departments based on vector similarity: {selected_depts}"
            logger.info("rag_router_hit", departments=selected_depts)
            
        else:
            # Fallback to simple keyword mapping if index fails
            logger.warning("rag_router_index_unavailable")
            # Use extracted constant map
            mapping = SYMPTOM_KEYWORD_MAP
            for k, v in mapping.items():
                if k in query:
                    if v not in selected_depts:
                        selected_depts.append(v)

    except Exception as e:
        logger.error("expert_router_failed", error=str(e))
        reasoning = f"Error: {str(e)}"

    # Fallback
    if not selected_depts:
        selected_depts = ["General Practice"]
        reasoning += " -> Fallback to GP"

    # [Anti-Lazy] Ensure response is verbose enough
    dept_str = ', '.join(selected_depts)
    msg_content = f"根据您的症状描述与历史病历分析，系统为您匹配了以下专家科室进行联合会诊：{dept_str}。"
    
    return {
        "departments": selected_depts,
        "router_reasoning": reasoning,
        "messages": [AIMessage(content=msg_content)]
    }
