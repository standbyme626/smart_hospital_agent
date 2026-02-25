from app.core.config import settings

logger = structlog.get_logger()

class MedicalSemanticChunker:
    """医学文本语义分块器"""

    def __init__(self, model_name: str = None, device: str = "cpu"):
        if model_name is None:
            model_name = settings.EMBEDDING_MODEL_PATH
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device, "trust_remote_code": True}, # Qwen 需要 trust_remote_code
            encode_kwargs={"normalize_embeddings": True}
        )
        # 使用百分位阈值策略，适合医学长文
        self.splitter = SemanticChunker(
            embeddings=self.embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=90
        )
        logger.info("chunker.initialized", model=model_name)

    def split_text(self, text: str) -> List[str]:
        """将文本分割为语义块"""
        try:
            chunks = self.splitter.split_text(text)
            logger.info("chunker.split", input_len=len(text), chunks_count=len(chunks))
            return chunks
        except Exception as e:
            logger.error("chunker.split_failed", error=str(e))
            # 降级策略：如果语义分块失败，返回原始文本（或考虑按长度切分）
            return [text]
