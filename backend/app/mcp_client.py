from __future__ import annotations

import json
import time
from typing import Any

import httpx

from .config import Settings


class MCPError(RuntimeError):
    pass


class HospitalMCPClient:
    """Minimal MCP JSON-RPC client for the hospital system.

    The mock server in ``backend/mcp_mock/server.py`` speaks the same protocol,
    so the real hospital MCP endpoint can be swapped in by changing the URL.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.url = settings.HOSPITAL_MCP_URL
        self.token = settings.HOSPITAL_MCP_TOKEN
        self.client = httpx.AsyncClient(timeout=20.0)
        self._initialized = False
        self._tools: list[dict[str, Any]] | None = None

    async def close(self) -> None:
        await self.client.aclose()

    async def _rpc(self, method: str, params: dict[str, Any] | None = None, request_id: int | None = None) -> Any:
        payload = {
            "jsonrpc": "2.0",
            "id": request_id or int(time.time() * 1000),
            "method": method,
            "params": params or {},
        }
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        response = await self.client.post(self.url, json=payload, headers=headers)
        if response.status_code >= 400:
            raise MCPError(f"HTTP {response.status_code}: {response.text[:200]}")
        data = response.json()
        if "error" in data:
            error = data["error"]
            raise MCPError(f"{error.get('code')}: {error.get('message')}")
        return data.get("result")

    async def initialize(self) -> dict[str, Any]:
        if self._initialized:
            return {"ok": True}
        result = await self._rpc(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "smart-hospital-agent", "version": "2.0.0"},
            },
        )
        self._initialized = True
        return result or {"ok": True}

    async def list_tools(self) -> list[dict[str, Any]]:
        if self._tools is not None:
            return self._tools
        await self.initialize()
        result = await self._rpc("tools/list", {})
        tools = (result or {}).get("tools", [])
        self._tools = tools
        return tools

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        await self.initialize()
        payload = dict(arguments or {})
        if idempotency_key:
            payload["idempotency_key"] = idempotency_key
        result = await self._rpc("tools/call", {"name": name, "arguments": payload})
        if not isinstance(result, dict):
            return {"text": str(result), "is_error": False}
        content = result.get("content") or []
        texts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                texts.append(str(item.get("text", "")))
        text = "\n".join(texts).strip()
        parsed: Any = None
        if text:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
        return {
            "text": text,
            "data": parsed,
            "is_error": bool(result.get("isError")),
        }

    async def health(self) -> bool:
        try:
            await self.list_tools()
            return True
        except Exception:
            return False
