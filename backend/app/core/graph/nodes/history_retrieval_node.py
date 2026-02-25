from app.db.session import AsyncSessionLocal
from app.rag.history_retriever import HistoryRetriever
from app.core.graph.state import AgentState
from app.core.monitoring.tracing import monitor_node
import structlog

logger = structlog.get_logger(__name__)

@monitor_node("history_retrieval")
async def history_retrieval_node(state: AgentState):
    """
    Retrieves patient history from SQL (recent visits) and Milvus (semantic memory).
    """
    patient_id = state.get("patient_id", "guest")
    if patient_id == "guest":
        # Try to find patient_id in persona if available
        persona = state.get("persona", {})
        if persona and "id" in persona:
             patient_id = persona["id"]
    
    # Retrieve user query for semantic search
    query = state.get("symptoms", "")
    
    context = ""
    try:
        async with AsyncSessionLocal() as session:
            retriever = HistoryRetriever(session)
            # This combines PG and Milvus retrieval
            context = await retriever.get_patient_context(patient_id, query)
            
        if context:
            logger.info("history_retrieved", patient_id=patient_id, length=len(context))
        else:
            logger.info("no_history_found", patient_id=patient_id)
            context = "No medical history found."
            
    except Exception as e:
        logger.error("history_retrieval_failed", error=str(e))
        # Fail gracefully, don't block the flow
        context = ""

    return {"medical_history": context}
