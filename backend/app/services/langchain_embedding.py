from typing import List
from langchain_core.embeddings import Embeddings
from app.services.embedding import EmbeddingService

class SharedEmbeddingService(Embeddings):
    """
    [Optimization] LangChain-compatible wrapper for the singleton EmbeddingService.
    Ensures that CrewAI agents share the same model instance in VRAM.
    """
    def __init__(self):
        self.service = EmbeddingService()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using the singleton service."""
        # Use batch processing if available in the service
        return self.service.batch_get_embeddings(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query using the singleton service."""
        return self.service.get_embedding(text)

# Singleton instance for global use
shared_embeddings = SharedEmbeddingService()
