from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Any
import os
import itertools
import logging
from dotenv import load_dotenv

# 项目根目录绝对路径（统一路径管理）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 强制加载根目录 .env（单一真源）
# [Fix] 不要删除系统环境变量，允许外部注入
# for env_key in ["OPENAI_API_KEY", "DASHSCOPE_API_KEY", "API_KEY_ROTATION_LIST"]:
#    if env_key in os.environ:
#        del os.environ[env_key]

# [Fix] override=False，优先使用系统环境变量，根目录 .env 仅作为默认值
ROOT_ENV_FILE = os.path.join(PROJECT_ROOT, ".env")
load_dotenv(ROOT_ENV_FILE, override=False)

logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    """
    系统全局配置类 (Global Settings)
    读取 .env 文件或环境变量，管理项目的所有配置项。
    包括数据库、Redis、Milvus、API 前缀等信息。
    """
    PROJECT_NAME: str = "Smart Hospital Agent"
    VERSION: str = "6.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Expose PROJECT_ROOT
    PROJECT_ROOT: str = PROJECT_ROOT

    # DATABASE
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_USER: str = "admin"
    POSTGRES_PASSWORD: str = "admin123"
    POSTGRES_DB: str = "smart_triage"
    POSTGRES_PORT: int = 5432
    DATABASE_URL: str = ""

    # REDIS
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_MAX_CONNECTIONS: int = 100 # [Phase 5.5] Increased default pool size

    MAX_TOKENS: int = 1024
    TEMPERATURE: float = 0.0  # [PROJECT_DNA] 必须死锁为 0.0，严禁修改

    # SECRETS (Required from environment)
    OPENAI_API_KEY: str = ""  # Must be set in .env
    DASHSCOPE_API_KEY: str = "" # Aliyun DashScope Key
    DASHSCOPE_API_KEY_POOL: str = "" # Pool of DashScope Keys
    
    # Key Rotation Strategy
    # If OPENAI_API_KEY is provided in .env, it will be the first candidate.
    # We will append the other keys here.
    API_KEY_ROTATION_LIST: str = ""
    API_KEY_CANDIDATES: List[str] = [] # Initialize empty
    LLM_KEY_BLACKLIST_ENABLED: bool = os.getenv("LLM_KEY_BLACKLIST_ENABLED", "false").lower() == "true"
    LLM_KEY_BLACKLIST_ON_AUTH_ONLY: bool = os.getenv("LLM_KEY_BLACKLIST_ON_AUTH_ONLY", "true").lower() == "true"
    
    # Internal iterator for Round-Robin rotation
    _key_iterator: Any = None

    OPENAI_API_BASE: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    
    # Model Configuration (Unified Management)
    # [V6.4.2] 从环境变量读取，支持随时更换
    # [Refactor] 移除硬编码的默认模型名称，强制从环境变量获取
    MODEL_SMART: str = (os.getenv("MODEL_SMART") or os.getenv("OPENAI_MODEL_NAME") or "qwen-max").split(",")[0].strip()
    MODEL_FAST: str = os.getenv("MODEL_FAST", "qwen-turbo").split(",")[0].strip()
    MODEL_CODER: str = os.getenv("MODEL_CODER", "qwen-max").split(",")[0].strip()
    MODEL_LONG: str = os.getenv("MODEL_LONG", "qwen-long").split(",")[0].strip()    
    @property
    def MODEL_CANDIDATES_LIST(self) -> List[str]:
        # Priority 1: OPENAI_MODEL_NAME (single or comma list)
        # 这样当用户显式指定本地/私有模型时，不会被历史 MODEL_CANDIDATES 覆盖。
        val = os.getenv("OPENAI_MODEL_NAME", "")
        if val:
            models = [m.strip() for m in val.split(",") if m.strip()]
            if models:
                return models

        # Priority 2: Explicit MODEL_CANDIDATES env var
        env_candidates = os.getenv("MODEL_CANDIDATES", "")
        if env_candidates:
            models = [m.strip() for m in env_candidates.split(",") if m.strip()]
            if models:
                return models

        # Fallback
        return ["qwen-max"]
    
    OPENAI_MODEL_NAME: str = os.getenv("OPENAI_MODEL_NAME")
    
    @property
    def OPENAI_MODEL_NAME_DYN(self) -> str:
        """动态获取模型名称，确保实时响应环境变量变化。如果包含逗号，取第一个作为默认模型。"""
        val = os.getenv("OPENAI_MODEL_NAME")
        if val: 
            return val.split(",")[0].strip()
        return (self.OPENAI_MODEL_NAME or "qwen-max").split(",")[0].strip()

    OPENAI_MODEL_FAST: str = os.getenv("MODEL_FAST", "qwen-turbo").split(",")[0].strip()
    OPENAI_MODEL_SMART: str = (os.getenv("OPENAI_MODEL_NAME") or "qwen-max").split(",")[0].strip()
    
    # [V6.3.5] 本地优先模式。当云端 API Key 失效或网络不稳定时，设为 True。
    PREFER_LOCAL: bool = False
    LOCAL_SLM_URL: str = os.getenv("LOCAL_SLM_URL", "http://localhost:8002/v1")
    LOCAL_SLM_MODEL: str = os.getenv("LOCAL_SLM_MODEL", "qwen3-0.6b-dpo-v11_2")
    LOCAL_SLM_QUANTIZATION: str = os.getenv("LOCAL_SLM_QUANTIZATION", "int8")  # [Optim] Support int8/awq/gptq

    # [Evolution Lab] 实验室模式开关
    EVOLUTION_MODE: bool = os.getenv("EVOLUTION_MODE", "False").lower() == "true"
    
    # [Config] Global switch for Local Fallback
    # default to False to strictly disable local models as per user request
    ENABLE_LOCAL_FALLBACK: bool = os.getenv("ENABLE_LOCAL_FALLBACK", "false").lower() == "true"
    
    @property
    def LLM_MODEL(self) -> str:
        return self.OPENAI_MODEL_NAME
    
    @property
    def LLM_API_KEY(self) -> str:
        return self.OPENAI_API_KEY
    
    @property
    def LLM_API_BASE(self) -> str:
        return self.OPENAI_API_BASE

    # [V6.5.8] Enforce OPENAI_BASE_URL for newer SDKs
    @property
    def OPENAI_BASE_URL(self) -> str:
        return self.OPENAI_API_BASE

    # MILVUS
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: str = "19530"

    # PATHS (Centralized)
    @property
    def LOG_DIR(self) -> str:
        dir_path = os.path.join(self.PROJECT_ROOT, "logs")
        os.makedirs(dir_path, exist_ok=True)
        return dir_path

    @property
    def TRACE_LOG_PATH(self) -> str:
        return os.path.join(self.LOG_DIR, "rag_trace.log")

    # RAG
    EMBEDDING_MODEL_PATH: str = os.path.join(PROJECT_ROOT, "models", "Qwen3-Embedding-0.6B")
    EMBEDDING_MODEL_NAME: str = "Qwen3-Embedding-0.6B"
    RERANKER_MODEL_PATH: str = os.path.join(PROJECT_ROOT, "models", "Qwen3-Reranker-0.6B")
    # Device policy: auto | cuda | cpu
    EMBEDDING_DEVICE: str = os.getenv("EMBEDDING_DEVICE", "auto")
    RERANKER_DEVICE: str = os.getenv("RERANKER_DEVICE", "auto")
    # Retrieval performance knobs
    RAG_RECALL_WINDOW: int = int(os.getenv("RAG_RECALL_WINDOW", "100"))
    RAG_RERANK_CANDIDATE_K: int = int(os.getenv("RAG_RERANK_CANDIDATE_K", "12"))
    # Pure retrieval mode: bypass all LLM-based rewrite/enhancement in RAG path.
    RAG_PURE_RETRIEVAL_MODE: bool = os.getenv("RAG_PURE_RETRIEVAL_MODE", "false").lower() == "true"
    RAG_DISABLE_INTENT_ROUTER_WHEN_PURE: bool = os.getenv("RAG_DISABLE_INTENT_ROUTER_WHEN_PURE", "true").lower() == "true"
    RAG_DISABLE_HYDE_WHEN_PURE: bool = os.getenv("RAG_DISABLE_HYDE_WHEN_PURE", "true").lower() == "true"
    RAG_DISABLE_QUERY_REWRITE_WHEN_PURE: bool = os.getenv("RAG_DISABLE_QUERY_REWRITE_WHEN_PURE", "true").lower() == "true"
    RAG_DISABLE_LOW_SCORE_FALLBACK_REWRITE_WHEN_PURE: bool = os.getenv(
        "RAG_DISABLE_LOW_SCORE_FALLBACK_REWRITE_WHEN_PURE",
        "true",
    ).lower() == "true"
    RAG_DISABLE_SUMMARIZE_WHEN_PURE: bool = os.getenv("RAG_DISABLE_SUMMARIZE_WHEN_PURE", "true").lower() == "true"
    # Semantic cache hit verification gates
    RAG_CACHE_VERIFY_ON_HIT: bool = os.getenv("RAG_CACHE_VERIFY_ON_HIT", "true").lower() == "true"
    RAG_CACHE_VERIFY_MIN_TERM_OVERLAP: int = int(os.getenv("RAG_CACHE_VERIFY_MIN_TERM_OVERLAP", "1"))
    RAG_CACHE_VERIFY_MIN_RERANK_SCORE: float = float(os.getenv("RAG_CACHE_VERIFY_MIN_RERANK_SCORE", "0.25"))
    RAG_CACHE_VERIFY_MAX_DOC_CHARS: int = int(os.getenv("RAG_CACHE_VERIFY_MAX_DOC_CHARS", "600"))
    RAG_SEMANTIC_CACHE_ENABLED: bool = os.getenv("RAG_SEMANTIC_CACHE_ENABLED", "true").lower() == "true"
    # Semantic cache governance (minimal hardening set)
    RAG_CACHE_NAMESPACE_VERSION: str = os.getenv("RAG_CACHE_NAMESPACE_VERSION", "v2")
    RAG_CACHE_TTL_SECONDS: int = int(os.getenv("RAG_CACHE_TTL_SECONDS", "86400"))
    RAG_CACHE_REJECT_TTL_SECONDS: int = int(os.getenv("RAG_CACHE_REJECT_TTL_SECONDS", "21600"))
    RAG_CACHE_WRITE_GATE_ON: bool = os.getenv("RAG_CACHE_WRITE_GATE_ON", "true").lower() == "true"
    RAG_CACHE_WRITE_MIN_TOP_SCORE: float = float(os.getenv("RAG_CACHE_WRITE_MIN_TOP_SCORE", "0.35"))
    RAG_CACHE_WRITE_MIN_SCORE_GAP: float = float(os.getenv("RAG_CACHE_WRITE_MIN_SCORE_GAP", "0.03"))
    RAG_CACHE_WRITE_REQUIRE_NON_UNKNOWN_DEPT: bool = os.getenv("RAG_CACHE_WRITE_REQUIRE_NON_UNKNOWN_DEPT", "false").lower() == "true"
    # Department post-processing: normalization + lightweight consistency gate
    RAG_DEPT_NORMALIZE_ON_RESULT: bool = os.getenv("RAG_DEPT_NORMALIZE_ON_RESULT", "true").lower() == "true"
    RAG_DEPT_CONSISTENCY_GATE_ON: bool = os.getenv("RAG_DEPT_CONSISTENCY_GATE_ON", "true").lower() == "true"
    RAG_DEPT_CONSISTENCY_BONUS: float = float(os.getenv("RAG_DEPT_CONSISTENCY_BONUS", "0.05"))
    RAG_DEPT_CONSISTENCY_MAX_SCORE_GAP: float = float(os.getenv("RAG_DEPT_CONSISTENCY_MAX_SCORE_GAP", "0.12"))
    RAG_DEPT_CONSISTENCY_MIN_KEYWORD_HITS: int = int(os.getenv("RAG_DEPT_CONSISTENCY_MIN_KEYWORD_HITS", "1"))
    RAG_DEPT_CONSISTENCY_REQUIRE_TOP1_UNSUPPORTED: bool = os.getenv(
        "RAG_DEPT_CONSISTENCY_REQUIRE_TOP1_UNSUPPORTED",
        "true",
    ).lower() == "true"
    # Pure 模式质量优先：在不启用重写回退时，适度放宽低分阻断阈值。
    # 例如 rerank_threshold=0.2 且 factor=0.75 -> pure 实际阈值 0.15。
    RAG_PURE_RERANK_THRESHOLD_FACTOR: float = float(os.getenv("RAG_PURE_RERANK_THRESHOLD_FACTOR", "0.75"))

    # Retrieval planner switches (Phase C skeleton; default off to avoid behavior drift)
    ENABLE_QUERY_REWRITE: bool = os.getenv("ENABLE_QUERY_REWRITE", "false").lower() == "true"
    QUERY_REWRITE_USE_LLM: bool = os.getenv("QUERY_REWRITE_USE_LLM", "false").lower() == "true"
    QUERY_REWRITE_MAX_VARIANTS: int = int(os.getenv("QUERY_REWRITE_MAX_VARIANTS", "3"))
    ENABLE_QUERY_EXPANSION: bool = os.getenv("ENABLE_QUERY_EXPANSION", "false").lower() == "true"
    ENABLE_MULTI_QUERY: bool = os.getenv("ENABLE_MULTI_QUERY", "false").lower() == "true"
    ENABLE_RETRIEVAL_ROUTER_ADAPTER: bool = os.getenv("ENABLE_RETRIEVAL_ROUTER_ADAPTER", "false").lower() == "true"
    QUERY_EXPANSION_MAX_VARIANTS: int = int(os.getenv("QUERY_EXPANSION_MAX_VARIANTS", "4"))
    QUERY_EXPANSION_MAX_QUERY_LEN_PER_VARIANT: int = int(
        os.getenv("QUERY_EXPANSION_MAX_QUERY_LEN_PER_VARIANT", "120")
    )
    QUERY_EXPANSION_REWRITE_TYPE_BUDGET: str = os.getenv(
        "QUERY_EXPANSION_REWRITE_TYPE_BUDGET",
        '{"synonym":2,"typo_fix":1,"llm_expand":1}',
    )
    MULTI_QUERY_FUSION_METHOD: str = os.getenv("MULTI_QUERY_FUSION_METHOD", "weighted_rrf")
    MULTI_QUERY_RRF_K: int = int(os.getenv("MULTI_QUERY_RRF_K", "60"))
    ENABLE_CONTEXT_WINDOW_AUTOMERGE: bool = os.getenv("ENABLE_CONTEXT_WINDOW_AUTOMERGE", "false").lower() == "true"
    CONTEXT_WINDOW_SIZE: int = int(os.getenv("CONTEXT_WINDOW_SIZE", "1"))
    CONTEXT_AUTOMERGE_ENABLED: bool = os.getenv("CONTEXT_AUTOMERGE_ENABLED", "true").lower() == "true"
    CONTEXT_DIVERSITY_FILTER_ENABLED: bool = os.getenv("CONTEXT_DIVERSITY_FILTER_ENABLED", "true").lower() == "true"
    CONTEXT_DIVERSITY_MAX_PER_SOURCE: int = int(os.getenv("CONTEXT_DIVERSITY_MAX_PER_SOURCE", "2"))
    CONTEXT_ORDERING_STRATEGY: str = os.getenv("CONTEXT_ORDERING_STRATEGY", "score_desc")
    CONTEXT_MAX_EVIDENCE: int = int(os.getenv("CONTEXT_MAX_EVIDENCE", "6"))
    CONTEXT_MAX_CHARS: int = int(os.getenv("CONTEXT_MAX_CHARS", "3200"))
    ENABLE_JSON_SCHEMA_GUARDRAIL: bool = os.getenv("ENABLE_JSON_SCHEMA_GUARDRAIL", "false").lower() == "true"
    DIAGNOSIS_SCHEMA_VERSION: str = os.getenv("DIAGNOSIS_SCHEMA_VERSION", "v1")
    ENABLE_DEBUG_SNAPSHOT: bool = os.getenv("ENABLE_DEBUG_SNAPSHOT", "false").lower() == "true"
    DEBUG_INCLUDE_NODES: str = os.getenv("DEBUG_INCLUDE_NODES", "")
    DIAGNOSIS_GRAPH_VERSION: str = os.getenv("DIAGNOSIS_GRAPH_VERSION", VERSION)
    DIAGNOSIS_DATA_CONTRACT_VERSION: str = os.getenv("DIAGNOSIS_DATA_CONTRACT_VERSION", "v1")
    DIAGNOSIS_DECISION_CONFIDENCE_THRESHOLD: float = float(
        os.getenv("DIAGNOSIS_DECISION_CONFIDENCE_THRESHOLD", "0.8")
    )
    DIAGNOSIS_DECISION_MIN_EVIDENCE: int = int(os.getenv("DIAGNOSIS_DECISION_MIN_EVIDENCE", "1"))
    DIAGNOSIS_DECISION_HIGH_RISK_KEYWORDS: str = os.getenv(
        "DIAGNOSIS_DECISION_HIGH_RISK_KEYWORDS",
        "紧急,急性,胸痛,呼吸困难,抽搐,昏迷,休克,high risk,emergency",
    )
    ENABLE_DECISION_GOVERNANCE: bool = os.getenv("ENABLE_DECISION_GOVERNANCE", "false").lower() == "true"
    ENABLE_EVAL_GATE: bool = os.getenv("ENABLE_EVAL_GATE", "false").lower() == "true"
    EVAL_GATE_ENFORCE: bool = os.getenv("EVAL_GATE_ENFORCE", "false").lower() == "true"
    ENABLE_DYNAMIC_REGRESSION: bool = os.getenv("ENABLE_DYNAMIC_REGRESSION", "false").lower() == "true"
    DYNAMIC_REGRESSION_ENFORCE: bool = os.getenv("DYNAMIC_REGRESSION_ENFORCE", "false").lower() == "true"

    ENABLE_ADAPTIVE_RETRIEVAL_K: bool = os.getenv("ENABLE_ADAPTIVE_RETRIEVAL_K", "false").lower() == "true"
    ADAPTIVE_RETRIEVAL_K_MIN: int = int(os.getenv("ADAPTIVE_RETRIEVAL_K_MIN", "2"))
    ADAPTIVE_RETRIEVAL_K_DEFAULT: int = int(os.getenv("ADAPTIVE_RETRIEVAL_K_DEFAULT", "3"))
    ADAPTIVE_RETRIEVAL_K_MAX: int = int(os.getenv("ADAPTIVE_RETRIEVAL_K_MAX", "6"))

    ENABLE_HIERARCHICAL_INDEX: bool = os.getenv("ENABLE_HIERARCHICAL_INDEX", "false").lower() == "true"
    DEFAULT_RETRIEVAL_INDEX_SCOPE: str = os.getenv("DEFAULT_RETRIEVAL_INDEX_SCOPE", "paragraph")
    ENABLE_INGRESS_GUARD: bool = os.getenv("ENABLE_INGRESS_GUARD", "true").lower() == "true"
    # Triage fast-path guardrails
    TRIAGE_FAST_PATH_ENABLED: bool = os.getenv("TRIAGE_FAST_PATH_ENABLED", "true").lower() == "true"
    TRIAGE_TOOL_TIMEOUT_SECONDS: float = float(os.getenv("TRIAGE_TOOL_TIMEOUT_SECONDS", "2.8"))
    TRIAGE_FAST_CONFIDENCE_THRESHOLD: float = float(os.getenv("TRIAGE_FAST_CONFIDENCE_THRESHOLD", "0.62"))
    QUERY_REWRITE_TIMEOUT_SECONDS: float = float(os.getenv("QUERY_REWRITE_TIMEOUT_SECONDS", "4.0"))
    CRISIS_FASTLANE_ENABLED: bool = os.getenv("CRISIS_FASTLANE_ENABLED", "true").lower() == "true"
    SSE_PING_INTERVAL_SECONDS: float = float(os.getenv("SSE_PING_INTERVAL_SECONDS", "8.0"))

    # SSE protocol compatibility
    ENABLE_UNIFIED_STREAM_SCHEMA: bool = os.getenv("ENABLE_UNIFIED_STREAM_SCHEMA", "false").lower() == "true"
    UNIFIED_STREAM_SCHEMA_VERSION: str = os.getenv("UNIFIED_STREAM_SCHEMA_VERSION", "v1")
    # HIS integration mode:
    # - legacy_sim: simulate MCP bridge to legacy HIS protocol (default for local non-HIS env)
    # - mock_direct: use direct in-process mock data without protocol framing
    HIS_MCP_MODE: str = os.getenv("HIS_MCP_MODE", "legacy_sim")


    # LangSmith
    LANGCHAIN_TRACING_V2: str = "false" # 强制关闭追踪，避免 429 报错干扰
    LANGCHAIN_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGCHAIN_API_KEY: str = ""
    LANGCHAIN_PROJECT: str = "Smart-Hospital-Agent-V6"
    LANGFUSE_ENABLED: bool = os.getenv("LANGFUSE_ENABLED", "false").lower() == "true"
    LANGFUSE_HOST: str = os.getenv("LANGFUSE_HOST", "http://127.0.0.1:3000")
    LANGFUSE_PUBLIC_KEY: str = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    LANGFUSE_SECRET_KEY: str = os.getenv("LANGFUSE_SECRET_KEY", "")
    LANGFUSE_ENVIRONMENT: str = os.getenv("LANGFUSE_ENVIRONMENT", "dev")

    # 单一配置源：固定读取项目根目录 .env，避免受启动目录影响
    model_config = SettingsConfigDict(case_sensitive=True, env_file=ROOT_ENV_FILE, extra="ignore")

    @staticmethod
    def _is_valid_key(key: str) -> bool:
        if not key or len(key) < 20:
            return False
        placeholders = ["sk-placeholder", "sk-example"]
        return not any(p in key for p in placeholders)

    @staticmethod
    def _mask_key(key: str) -> str:
        if not key:
            return "empty"
        if len(key) <= 12:
            return "****"
        return f"{key[:8]}...{key[-4:]}"

    def _build_key_candidates(self) -> List[str]:
        candidates = set()

        if self.OPENAI_API_KEY and self._is_valid_key(self.OPENAI_API_KEY):
            candidates.add(self.OPENAI_API_KEY)

        if self.DASHSCOPE_API_KEY and self._is_valid_key(self.DASHSCOPE_API_KEY):
            candidates.add(self.DASHSCOPE_API_KEY)

        if self.DASHSCOPE_API_KEY_POOL:
            pool_keys = [
                k.strip()
                for k in self.DASHSCOPE_API_KEY_POOL.split(",")
                if self._is_valid_key(k.strip())
            ]
            candidates.update(pool_keys)

        if self.API_KEY_ROTATION_LIST:
            rotation_keys = [
                k.strip()
                for k in self.API_KEY_ROTATION_LIST.split(",")
                if self._is_valid_key(k.strip())
            ]
            candidates.update(rotation_keys)

        if self.OPENAI_API_KEY and len(self.OPENAI_API_KEY) > 10:
            candidates.add(self.OPENAI_API_KEY)

        if not candidates:
            env_key = os.getenv("OPENAI_API_KEY")
            if env_key and len(env_key) > 10:
                candidates.add(env_key)

        return list(candidates)

    def _export_runtime_env(self) -> None:
        os.environ["LANGCHAIN_TRACING_V2"] = self.LANGCHAIN_TRACING_V2
        os.environ["LANGCHAIN_ENDPOINT"] = self.LANGCHAIN_ENDPOINT
        os.environ["LANGCHAIN_API_KEY"] = self.LANGCHAIN_API_KEY
        os.environ["LANGCHAIN_PROJECT"] = self.LANGCHAIN_PROJECT
        os.environ["OPENAI_API_BASE"] = self.OPENAI_API_BASE
        os.environ["OPENAI_BASE_URL"] = self.OPENAI_API_BASE
        os.environ["OPENAI_API_KEY"] = self.OPENAI_API_KEY
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    def __init__(self, **kwargs):
        """
        初始化配置 (Initialize Settings)
        如果未设置 DATABASE_URL，则根据独立的 POSTGRES_* 变量自动拼接生成。
        """
        super().__init__(**kwargs)
        if not self.DATABASE_URL:
            self.DATABASE_URL = f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        
        # 验证敏感配置 (Production Hardening)
        critical_keys = {
            "OPENAI_API_KEY": self.OPENAI_API_KEY,
            "DASHSCOPE_API_KEY": self.DASHSCOPE_API_KEY,
            "POSTGRES_PASSWORD": self.POSTGRES_PASSWORD
        }
        
        for key, value in critical_keys.items():
            if key in ["OPENAI_API_KEY", "DASHSCOPE_API_KEY"] and value:
                 logger.debug("%s configured: %s (len=%d)", key, self._mask_key(value), len(value))
            if not value or str(value).startswith("sk-mock") or value == "admin123":
                 logger.warning("⚠️ [Security] %s is using a weak or default value! Not recommended for production.", key)

        # [V6.4.12] 彻底清除无效的主 Key
        if not self._is_valid_key(self.OPENAI_API_KEY):
            self.OPENAI_API_KEY = ""
            if "OPENAI_API_KEY" in os.environ:
                del os.environ["OPENAI_API_KEY"]
                
        if not self._is_valid_key(self.DASHSCOPE_API_KEY):
            self.DASHSCOPE_API_KEY = ""
            if "DASHSCOPE_API_KEY" in os.environ:
                del os.environ["DASHSCOPE_API_KEY"]

        self.API_KEY_CANDIDATES = self._build_key_candidates()
        logger.debug("Final API_KEY_CANDIDATES count: %d", len(self.API_KEY_CANDIDATES))
        
        if not self.API_KEY_CANDIDATES:
             # 强制添加一个标识，防止下游循环直接跳过
             # 这会导致下游报错更清晰（鉴权错误而非连接错误）
             self.API_KEY_CANDIDATES = ["sk-no-key-configured"]

        # 2. Auto-Adaptation: If OPENAI_API_KEY is empty, promote the first candidate
        if not self.OPENAI_API_KEY and self.API_KEY_CANDIDATES:
            self.OPENAI_API_KEY = self.API_KEY_CANDIDATES[0]
            logger.info("OPENAI_API_KEY was empty, promoted %s as primary.", self._mask_key(self.OPENAI_API_KEY))

        # 3. Initialize Round-Robin Iterator
        if self.API_KEY_CANDIDATES:
            self._key_iterator = itertools.cycle(self.API_KEY_CANDIDATES)
        
        self._export_runtime_env()
        logger.info("HF_ENDPOINT set to https://hf-mirror.com for domestic model access.")

    def get_next_api_key(self) -> str:
        """Get next API key from rotation pool (Thread-safe Round Robin)"""
        if self._key_iterator:
            key = next(self._key_iterator)
            # Add logging for key rotation
            import logging
            mask = f"{key[:8]}...{key[-4:]}" if len(key) > 12 else "****"
            logging.info(f"🔑 [Rotation] Using API Key: {mask} (Pool Size: {len(self.API_KEY_CANDIDATES)})")
            return key
        return self.OPENAI_API_KEY

settings = Settings()
