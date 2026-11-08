from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent
DATA_DIR = BACKEND_DIR / "app" / "data"


def _read_opencode_key() -> str:
    if os.getenv("OPENCODE_API_KEY"):
        return os.environ["OPENCODE_API_KEY"]
    path = Path.home() / ".local" / "share" / "opencode" / "auth.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return str(data.get("opencode-go", {}).get("key", ""))
    except Exception:
        return ""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    APP_NAME: str = "smart-hospital-agent"
    APP_VERSION: str = "3.1.0"
    ENV: str = "dev"
    DEBUG: bool = False

    # Security
    AUTH_ENABLED: bool = True
    JWT_SECRET: str = "change-me-in-production"
    JWT_TTL_MINUTES: int = 720
    ADMIN_USER: str = "admin"
    ADMIN_PASSWORD: str = "admin123456"
    PATIENT_USER: str = "patient"
    PATIENT_PASSWORD: str = "patient123456"
    CORS_ORIGINS: str = "http://localhost:3000,http://127.0.0.1:3000"
    RATE_LIMIT_PER_MINUTE: int = 60
    CHAT_DAILY_LIMIT: int = 60

    # Data
    DATABASE_URL: str = "postgresql://sha:sha_pwd@127.0.0.1:5434/sha"
    REDIS_URL: str = "redis://127.0.0.1:6381/0"

    # Models
    OPENCODE_API_BASE: str = "https://opencode.ai/inference/go/openai/v1"
    OPENCODE_API_KEY: str = ""
    LLM_MODEL: str = "deepseek-v4.1-flash"
    LLM_REASONING_EFFORT: str = ""

    SILICONFLOW_API_BASE: str = "https://api.siliconflow.cn/v1"
    SILICONFLOW_API_KEY: str = ""
    EMBEDDING_MODEL: str = "Qwen/Qwen3-Embedding-8B"
    EMBEDDING_DIM: int = 4096
    RERANK_MODEL: str = "Qwen/Qwen3-Reranker-8B"

    OLLAMA_BASE_URL: str = "http://127.0.0.1:11434/v1"
    OLLAMA_MODEL: str = "qwen3-0.6b-dpo-v11_2"
    OLLAMA_ENABLED: bool = True
    OLLAMA_TIMEOUT_SECONDS: float = 20.0

    # Retrieval
    SEED_LIMIT: int = 2000
    RETRIEVAL_TOP_K: int = 5
    RETRIEVAL_DENSE_K: int = 30
    RETRIEVAL_SPARSE_K: int = 30
    RERANK_TOP_N: int = 20

    # Hospital MCP
    HOSPITAL_MCP_URL: str = "http://127.0.0.1:9100/mcp"
    HOSPITAL_MCP_TOKEN: str = ""
    HITL_CONFIRM_TOKEN: str = "change-me"
    UPLOADS_DIR: str = str(BACKEND_DIR / "uploads")
    MAX_UPLOAD_MB: int = 8

    # Costs (USD per 1M tokens, for accounting only)
    LLM_INPUT_COST_PER_M: float = 0.15
    LLM_OUTPUT_COST_PER_M: float = 0.6

    @property
    def opencode_key(self) -> str:
        return self.OPENCODE_API_KEY or _read_opencode_key()

    @property
    def cors_origins(self) -> list[str]:
        return [item.strip() for item in self.CORS_ORIGINS.split(",") if item.strip()]

    @property
    def auth_enabled(self) -> bool:
        return bool(self.AUTH_ENABLED)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    if not settings.OPENCODE_API_KEY:
        settings.OPENCODE_API_KEY = _read_opencode_key()
    if not settings.SILICONFLOW_API_KEY:
        settings.SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
    return settings
