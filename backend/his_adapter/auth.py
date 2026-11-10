from __future__ import annotations

import asyncio
from typing import Any
from urllib.parse import quote

import httpx

from .errors import AdapterError, normalize_business, normalize_http


def _json(resp: httpx.Response) -> dict[str, Any]:
    try:
        body = resp.json()
    except ValueError as exc:
        raise normalize_http(resp.status_code, resp.text) from exc
    if resp.status_code >= 400:
        raise normalize_http(resp.status_code, body)
    if not isinstance(body, dict):
        raise normalize_http(resp.status_code, body)
    code = body.get("code")
    if code is not None and str(code) not in ("0", "200"):
        raise normalize_business(body)
    return body


def _tenant_list(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        records = data.get("records") or data.get("rows") or []
        if isinstance(records, list):
            return [item for item in records if isinstance(item, dict)]
    return []


class TokenCache:
    def __init__(
        self,
        base_url: str,
        username: str,
        password: str,
        tenant_id: str,
        client: httpx.AsyncClient,
        timeout: float,
    ):
        self.base_url = base_url.rstrip("/")
        self.username = username
        self.password = password
        self.tenant_id = str(tenant_id or "")
        self.client = client
        self.timeout = timeout
        self._token: str | None = None
        self._tenant: str | None = None
        self._lock = asyncio.Lock()

    async def get(self) -> str:
        if self._token:
            return self._token
        async with self._lock:
            if self._token:
                return self._token
            return await self._login()

    async def refresh(self) -> str:
        async with self._lock:
            self._token = None
            return await self._login()

    async def invalidate(self) -> None:
        self._token = None

    async def tenants(self) -> list[dict[str, Any]]:
        url = f"{self.base_url}/system/tenant/user-bind/{quote(self.username, safe='')}"
        try:
            resp = await self.client.get(url, timeout=self.timeout)
        except httpx.TimeoutException as exc:
            raise AdapterError("HIS_TIMEOUT", "tenant lookup timed out", retryable=True) from exc
        except httpx.TransportError as exc:
            raise AdapterError("HIS_UNAVAILABLE", str(exc), retryable=True) from exc
        body = _json(resp)
        return _tenant_list(body.get("data"))

    async def _login(self) -> str:
        if not self.username or not self.password:
            raise AdapterError("HIS_AUTH_FAILED", "OpenHIS credentials are not configured")
        tenants = await self.tenants()
        if not tenants:
            raise AdapterError("HIS_TENANT_MISMATCH", f"user {self.username!r} has no bound tenant")
        chosen = None
        for tenant in tenants:
            if self.tenant_id and str(tenant.get("id") or tenant.get("tenantId")) == self.tenant_id:
                chosen = tenant
                break
        if chosen is None:
            if self.tenant_id:
                raise AdapterError(
                    "HIS_TENANT_MISMATCH",
                    f"tenant {self.tenant_id!r} is not bound to user {self.username!r}",
                )
            chosen = tenants[0]
        tenant_value = chosen.get("id") or chosen.get("tenantId")
        payload = {
            "username": self.username,
            "password": self.password,
            "code": "",
            "uuid": "",
            "tenantId": int(tenant_value) if str(tenant_value).isdigit() else tenant_value,
        }
        try:
            resp = await self.client.post(f"{self.base_url}/login", json=payload, timeout=self.timeout)
        except httpx.TimeoutException as exc:
            raise AdapterError("HIS_TIMEOUT", "login timed out", retryable=True) from exc
        except httpx.TransportError as exc:
            raise AdapterError("HIS_UNAVAILABLE", str(exc), retryable=True) from exc
        body = _json(resp)
        data = body.get("data") if isinstance(body.get("data"), dict) else {}
        token = body.get("token") or data.get("token") or body.get("access_token")
        if not token:
            raise AdapterError("HIS_AUTH_FAILED", "login response did not contain a token")
        self._token = str(token)
        self._tenant = str(tenant_value)
        return self._token
