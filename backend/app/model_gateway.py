from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx

from .config import Settings

CJK = re.compile(r"[\u4e00-\u9fff]")
ASCII = re.compile(r"[a-zA-Z0-9_]+")


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ChatResult:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    provider: str = ""
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0
    latency_ms: int = 0
    raw: dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""


class ModelError(RuntimeError):
    pass


class ModelGateway:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(90.0, connect=10.0),
            limits=httpx.Limits(max_connections=40, max_keepalive_connections=10),
        )
        self._ollama_available: bool | None = None
        self._ollama_checked_at = 0.0
        self.session_id = uuid.uuid4().hex

    async def close(self) -> None:
        await self.client.aclose()

    # ------------------------------------------------------------------ chat
    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        task: str = "diagnosis",
        temperature: float = 0.2,
        max_tokens: int = 2048,
        response_format: dict[str, Any] | None = None,
        session_id: str | None = None,
        reasoning_effort: str | None = None,
    ) -> ChatResult:
        order = await self._provider_order(task)
        errors: list[str] = []
        for provider in order:
            try:
                if provider == "ollama":
                    return await self._chat_ollama(
                        messages,
                        tools=None,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                return await self._chat_opencode(
                    messages,
                    tools=tools,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                    session_id=session_id,
                    reasoning_effort=reasoning_effort,
                )
            except Exception as exc:
                errors.append(f"{provider}: {exc}")
                continue
        raise ModelError("; ".join(errors) or "no model provider available")

    async def _provider_order(self, task: str) -> list[str]:
        local = self.settings.OLLAMA_ENABLED and await self._check_ollama()
        if task == "local":
            return ["ollama"]
        if task == "diagnosis":
            return ["opencode", "ollama"] if local else ["opencode"]
        return ["opencode", "ollama"] if local else ["opencode"]

    async def _chat_opencode(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None,
        temperature: float,
        max_tokens: int,
        response_format: dict[str, Any] | None,
        session_id: str | None,
        reasoning_effort: str | None = None,
    ) -> ChatResult:
        payload: dict[str, Any] = {
            "model": self.settings.LLM_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if response_format:
            payload["response_format"] = response_format
        effort = reasoning_effort or self.settings.LLM_REASONING_EFFORT
        if effort:
            payload["reasoning_effort"] = effort
        started = time.perf_counter()
        data = await self._request_with_retry(
            "POST",
            f"{self.settings.OPENCODE_API_BASE}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.settings.opencode_key}",
                "x-opencode-session": session_id or self.session_id,
            },
            json_body=payload,
            timeout=90.0,
        )
        latency = int((time.perf_counter() - started) * 1000)
        return self._parse_chat(data, "opencode-go", self.settings.LLM_MODEL, latency)

    async def _chat_ollama(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None,
        temperature: float,
        max_tokens: int,
    ) -> ChatResult:
        payload: dict[str, Any] = {
            "model": self.settings.OLLAMA_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
        started = time.perf_counter()
        data = await self._request_with_retry(
            "POST",
            f"{self.settings.OLLAMA_BASE_URL}/chat/completions",
            headers={},
            json_body=payload,
            timeout=self.settings.OLLAMA_TIMEOUT_SECONDS,
        )
        latency = int((time.perf_counter() - started) * 1000)
        return self._parse_chat(data, "ollama", self.settings.OLLAMA_MODEL, latency)

    def _parse_chat(self, data: dict[str, Any], provider: str, model: str, latency: int) -> ChatResult:
        choices = data.get("choices") or []
        if not choices:
            raise ModelError(f"empty choices: {json.dumps(data)[:200]}")
        message = choices[0].get("message") or {}
        content = message.get("content") or ""
        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
        if not content and reasoning:
            text = re.sub(r"<think>.*?</think>", "", reasoning, flags=re.S)
            content = text.strip() or re.sub(r"</?think>", "", reasoning).strip()
        calls: list[ToolCall] = []
        for index, call in enumerate(message.get("tool_calls") or []):
            function = call.get("function") or {}
            name = function.get("name") or ""
            raw_args = function.get("arguments") or "{}"
            if isinstance(raw_args, str):
                try:
                    arguments = json.loads(raw_args)
                except json.JSONDecodeError:
                    arguments = {"raw": raw_args}
            else:
                arguments = raw_args if isinstance(raw_args, dict) else {"raw": raw_args}
            calls.append(ToolCall(id=call.get("id") or f"call_{index}", name=name, arguments=arguments))
        usage = data.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        cost = 0.0
        if provider == "opencode-go":
            cost = (
                prompt_tokens / 1_000_000 * self.settings.LLM_INPUT_COST_PER_M
                + completion_tokens / 1_000_000 * self.settings.LLM_OUTPUT_COST_PER_M
            )
        return ChatResult(
            content=content,
            tool_calls=calls,
            provider=provider,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost=cost,
            latency_ms=latency,
            raw=data,
            reasoning=reasoning,
        )

    # ----------------------------------------------------------------- embed
    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self.settings.SILICONFLOW_API_KEY:
            try:
                return await self._embed_siliconflow(texts)
            except Exception:
                pass
        return [self._hash_embed(text) for text in texts]

    async def _embed_siliconflow(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        batch_size = 16
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            data = await self._request_with_retry(
                "POST",
                f"{self.settings.SILICONFLOW_API_BASE}/embeddings",
                headers={"Authorization": f"Bearer {self.settings.SILICONFLOW_API_KEY}"},
                json_body={
                    "model": self.settings.EMBEDDING_MODEL,
                    "input": batch,
                    "encoding_format": "float",
                },
                timeout=120.0,
            )
            items = sorted(data.get("data") or [], key=lambda item: item.get("index", 0))
            if len(items) != len(batch):
                raise ModelError("embedding response size mismatch")
            for item in items:
                vector = item.get("embedding")
                if not isinstance(vector, list) or not vector:
                    raise ModelError("embedding response missing vector")
                vectors.append([float(x) for x in vector])
        return vectors

    def _hash_embed(self, text: str) -> list[float]:
        dim = self.settings.EMBEDDING_DIM
        vector = [0.0] * dim
        tokens = [token for token in ASCII.findall(text.lower())]
        tokens.extend(CJK.findall(text))
        cjk = "".join(CJK.findall(text))
        tokens.extend(cjk[i : i + 2] for i in range(max(0, len(cjk) - 1)))
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    # ---------------------------------------------------------------- rerank
    async def rerank(self, query: str, documents: list[str], top_n: int = 20) -> list[tuple[int, float]]:
        if not documents:
            return []
        if not self.settings.SILICONFLOW_API_KEY:
            return [(index, 1.0 / (index + 1)) for index in range(len(documents))]
        try:
            data = await self._request_with_retry(
                "POST",
                f"{self.settings.SILICONFLOW_API_BASE}/rerank",
                headers={"Authorization": f"Bearer {self.settings.SILICONFLOW_API_KEY}"},
                json_body={
                    "model": self.settings.RERANK_MODEL,
                    "query": query,
                    "documents": documents,
                    "top_n": min(top_n, len(documents)),
                    "return_documents": False,
                },
                timeout=60.0,
            )
            results = data.get("results") or []
            ranked = [
                (int(item.get("index", 0)), float(item.get("relevance_score", 0.0)))
                for item in results
            ]
            return sorted(ranked, key=lambda item: item[1], reverse=True)
        except Exception:
            return [(index, 1.0 / (index + 1)) for index in range(len(documents))]

    # ---------------------------------------------------------------- health
    async def health(self) -> dict[str, Any]:
        return {
            "opencode": bool(self.settings.opencode_key),
            "siliconflow": bool(self.settings.SILICONFLOW_API_KEY),
            "ollama": await self._check_ollama(),
        }

    async def _check_ollama(self) -> bool:
        if not self.settings.OLLAMA_ENABLED:
            return False
        now = time.time()
        if self._ollama_available is not None and now - self._ollama_checked_at < 30:
            return self._ollama_available
        try:
            response = await self.client.get(
                f"{self.settings.OLLAMA_BASE_URL}/models",
                timeout=3.0,
            )
            self._ollama_available = response.status_code == 200
        except Exception:
            self._ollama_available = False
        self._ollama_checked_at = now
        return bool(self._ollama_available)

    # ----------------------------------------------------------------- utils
    async def _request_with_retry(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        json_body: dict[str, Any],
        timeout: float,
        attempts: int = 3,
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                response = await self.client.request(
                    method,
                    url,
                    headers=headers,
                    json=json_body,
                    timeout=timeout,
                )
                if response.status_code in {429, 500, 502, 503, 504} and attempt < attempts - 1:
                    await asyncio.sleep(2**attempt)
                    continue
                if response.status_code >= 400:
                    raise ModelError(f"HTTP {response.status_code}: {response.text[:300]}")
                return response.json()
            except Exception as exc:
                last_error = exc
                if attempt < attempts - 1:
                    await asyncio.sleep(2**attempt)
                    continue
        raise ModelError(str(last_error) if last_error else "request failed")
