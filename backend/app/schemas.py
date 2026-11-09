from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(..., min_length=1, max_length=4000)
    title: str | None = Field(default=None, max_length=80)
    session_id: str | None = None
    member_id: str | None = None
    mode: str = "patient"


class DrugCheckRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=64)
    password: str


class AccountDeleteRequest(BaseModel):
    password: str = Field(..., min_length=1, max_length=200)


class LockRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slot_id: str
    member_id: str | None = None


class OrderRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lock_id: str
    member_id: str | None = None


class PayRequest(BaseModel):
    confirm_token: str
    method: str = "mock"


class ReviewRejectRequest(BaseModel):
    note: str = Field(default="", max_length=500)


class EvalRequest(BaseModel):
    name: str = "smoke"
    limit: int = Field(100, ge=10, le=500)


class AdminUserCreateRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=64)
    password: str = Field(..., min_length=6, max_length=200)
    role: Literal["admin", "patient"] = "patient"
    enabled: bool = True


class AdminUserUpdateRequest(BaseModel):
    enabled: bool | None = None
    role: Literal["admin", "patient"] | None = None
    password: str | None = Field(default=None, min_length=6, max_length=200)


class FamilyMemberRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=40)
    relation: str = "self"
    gender: str | None = None
    birth_date: str | None = None
    phone: str | None = None
    allergies: str | None = None
    conditions: str | None = None
    is_self: bool = False


class MedicationRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)
    member_id: str | None = None
    dose: str | None = None
    frequency: str | None = None
    reminder_times: list[str] = Field(default_factory=list)
    notes: str | None = None


class MedCheckRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)
    member_id: str | None = None


class FollowupRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=120)
    member_id: str | None = None
    due_date: str | None = None
    notes: str | None = None


class AnalyticsEventRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)
    session_id: str | None = None
    props: dict[str, Any] = Field(default_factory=dict)
