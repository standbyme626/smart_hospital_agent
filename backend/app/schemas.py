from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    session_id: str | None = None
    user_id: str | None = None
    mode: str = "patient"


class RagSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=20)
    department: str | None = None


class LocalAnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=8000)
    instruction: str = "请用简洁的中文做医学文本分析，提炼要点、风险与建议。"


class DrugCheckRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)


class LoginRequest(BaseModel):
    username: str
    password: str


class LockRequest(BaseModel):
    slot_id: str
    patient_id: str | None = None


class OrderRequest(BaseModel):
    lock_id: str
    patient_id: str | None = None


class PayRequest(BaseModel):
    confirm_token: str
    method: str = "mock"


class RuleRequest(BaseModel):
    kind: str
    key: str
    value: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True


class EvalRequest(BaseModel):
    name: str = "smoke"
    limit: int = Field(100, ge=10, le=500)
