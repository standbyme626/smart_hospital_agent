import structlog
from transformers import AutoTokenizer
from app.core.config import settings
import os

logger = structlog.get_logger(__name__)

class TokenizerService:
    """
    Global Tokenizer Service (Singleton)
    
    Provides accurate token counting using HuggingFace AutoTokenizer.
    Prioritizes local models defined in settings, falls back to standard BERT.
    """
    _instance = None
    _tokenizer = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TokenizerService, cls).__new__(cls)
            cls._instance._load_tokenizer()
        return cls._instance

    def _load_tokenizer(self):
        """
        Attempt to load tokenizer from multiple sources:
        1. settings.EMBEDDING_MODEL_PATH (Local)
        2. bert-base-chinese (Remote/Cache)
        """
        candidates = []
        
        # 1. Local path from settings
        if settings.EMBEDDING_MODEL_PATH and os.path.exists(settings.EMBEDDING_MODEL_PATH):
            candidates.append(settings.EMBEDDING_MODEL_PATH)
            
        # 2. Fallback standard
        candidates.append("bert-base-chinese")

        for model_path in candidates:
            try:
                logger.info("loading_tokenizer", path=model_path)
                # trust_remote_code=True is needed for some custom models like Qwen
                self._tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                logger.info("tokenizer_loaded", path=model_path)
                return
            except Exception as e:
                logger.warning("tokenizer_load_failed", path=model_path, error=str(e))
        
        logger.error("all_tokenizers_failed", detail="Falling back to heuristic counting")
        self._tokenizer = None

    def count_tokens(self, text: str) -> int:
        """
        Count tokens for a given text string.
        Returns 0 for empty input.
        Falls back to heuristic (len * 0.7) if tokenizer is unavailable.
        """
        if not text:
            return 0
            
        if self._tokenizer:
            try:
                # add_special_tokens=False ensures we don't count [CLS]/[SEP] for partial fragments
                return len(self._tokenizer.encode(str(text), add_special_tokens=False))
            except Exception as e:
                # Log only once to avoid spamming
                pass
        
        # Fallback Heuristic
        # Chinese chars are often 1-2 tokens? No, usually 1 char = 1-1.5 tokens in Qwen/BERT
        # English words are 1.3 tokens.
        # Conservative estimate: 1 char ~= 1 token for Chinese, 0.5 for English.
        # Let's use 0.7 as a general mix factor.
        return int(len(str(text)) * 0.7)

# Singleton instance
global_tokenizer = TokenizerService()
