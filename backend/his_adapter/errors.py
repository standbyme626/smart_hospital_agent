from __future__ import annotations

from typing import Any

HIS_AUTH_FAILED = "HIS_AUTH_FAILED"
HIS_UNAVAILABLE = "HIS_UNAVAILABLE"
HIS_TIMEOUT = "HIS_TIMEOUT"
HIS_RATE_LIMITED = "HIS_RATE_LIMITED"
HIS_VALIDATION = "HIS_VALIDATION"
HIS_NOT_FOUND = "HIS_NOT_FOUND"
HIS_CONFLICT = "HIS_CONFLICT"
HIS_TENANT_MISMATCH = "HIS_TENANT_MISMATCH"
HIS_PERMISSION_DENIED = "HIS_PERMISSION_DENIED"
HIS_UNSUPPORTED = "HIS_UNSUPPORTED"
HIS_DUPLICATE_PATIENT = "HIS_DUPLICATE_PATIENT"
HIS_DUPLICATE_REGISTRATION = "HIS_DUPLICATE_REGISTRATION"
HIS_PAYMENT_REQUIRED = "HIS_PAYMENT_REQUIRED"
HIS_PAYMENT_MISMATCH = "HIS_PAYMENT_MISMATCH"
ADAPTER_IDEMPOTENCY_CONFLICT = "ADAPTER_IDEMPOTENCY_CONFLICT"
ADAPTER_MAPPING_MISSING = "ADAPTER_MAPPING_MISSING"
ADAPTER_SCOPE_DENIED = "ADAPTER_SCOPE_DENIED"
ADAPTER_OUTBOX_PENDING = "ADAPTER_OUTBOX_PENDING"

RETRYABLE_CODES = frozenset({HIS_TIMEOUT, HIS_UNAVAILABLE, HIS_RATE_LIMITED, ADAPTER_OUTBOX_PENDING})

TRANSIENT_HTTP = frozenset({408, 425, 429, 500, 502, 503, 504})


class AdapterError(Exception):
    def __init__(self, code: str, message: str, retryable: bool | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = code in RETRYABLE_CODES if retryable is None else retryable

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "retryable": self.retryable}


def _message(body: Any) -> str:
    if isinstance(body, dict):
        for key in ("msg", "message", "detail", "error"):
            value = body.get(key)
            if value:
                return str(value)[:300]
        return str(body)[:300]
    return str(body or "")[:300]


def normalize_http(status: int, body: Any = None) -> AdapterError:
    message = _message(body) or f"HTTP {status}"
    if status == 401:
        return AdapterError(HIS_AUTH_FAILED, message, retryable=False)
    if status == 403:
        return AdapterError(HIS_PERMISSION_DENIED, message, retryable=False)
    if status == 404:
        return AdapterError(HIS_NOT_FOUND, message, retryable=False)
    if status == 409:
        return AdapterError(HIS_CONFLICT, message, retryable=False)
    if status == 429:
        return AdapterError(HIS_RATE_LIMITED, message, retryable=True)
    if status == 408:
        return AdapterError(HIS_TIMEOUT, message, retryable=True)
    if status >= 500:
        return AdapterError(HIS_UNAVAILABLE, message, retryable=status in TRANSIENT_HTTP)
    return AdapterError(HIS_VALIDATION, message, retryable=False)


def normalize_business(body: Any, context: str | None = None) -> AdapterError:
    message = _message(body) or "OpenHIS business error"
    text = message.lower()
    code = body.get("code") if isinstance(body, dict) else None

    def has(*needles: str) -> bool:
        return any(needle in message or needle in text for needle in needles)

    if has("租户", "tenant"):
        return AdapterError(HIS_TENANT_MISMATCH, message, retryable=False)
    if has("权限", "未授权", "无权限", "forbidden", "not allowed"):
        return AdapterError(HIS_PERMISSION_DENIED, message, retryable=False)
    if has("登录", "认证", "token", "token无效"):
        return AdapterError(HIS_AUTH_FAILED, message, retryable=False)
    if has("密码", "用户名", "账号"):
        return AdapterError(HIS_AUTH_FAILED, message, retryable=False)
    if context == "patient" and has("身份证", "证件", "已存在", "重复"):
        return AdapterError(HIS_DUPLICATE_PATIENT, message, retryable=False)
    if context == "registration" and has("已挂号", "重复挂号", "已就诊", "重复", "已存在"):
        return AdapterError(HIS_DUPLICATE_REGISTRATION, message, retryable=False)
    if has("未支付", "请先支付", "payment required", "未结算"):
        return AdapterError(HIS_PAYMENT_REQUIRED, message, retryable=False)
    if has("金额", "不一致", "mismatch", "对账"):
        return AdapterError(HIS_PAYMENT_MISMATCH, message, retryable=False)
    if has("不存在", "未找到", "未查询到", "not found"):
        return AdapterError(HIS_NOT_FOUND, message, retryable=False)
    if has("已取消", "已退号", "冲突", "状态不允许", "请勿重复"):
        return AdapterError(HIS_CONFLICT, message, retryable=False)
    if code in (401, "401"):
        return AdapterError(HIS_AUTH_FAILED, message, retryable=False)
    return AdapterError(HIS_VALIDATION, message, retryable=False)
