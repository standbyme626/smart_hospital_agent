"""Shared deterministic fake model gateway for v3.1 tests.

Mirrors the OpenAI-facing behaviour used by the real ``ModelGateway``:
triage JSON, ``submit_report`` tool calls, vision JSON, embeddings and rerank.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any

from app.config import get_settings
from app.model_gateway import ChatResult, ToolCall

REPORT_MARKER = "【E2E】"

TRIAGE = {
    "intent": "MEDICAL_CONSULT",
    "department": "心血管内科",
    "urgency": "normal",
    "symptoms": "头痛伴乏力",
    "reason": "fake triage",
    "confidence": 0.86,
}

REPORT = {
    "department": "心血管内科",
    "urgency": "normal",
    "summary": f"{REPORT_MARKER}建议到心血管内科门诊评估，注意休息并监测血压。",
    "advice": ["规律作息", "监测血压"],
    "medications": [],
    "citations": [],
    "confidence": 0.82,
    "followup_question": "最近是否有胸闷或心悸？",
}

VISION_REPORT = {
    "summary": f"{REPORT_MARKER}检查报告未见明显异常，白细胞轻度升高。",
    "abnormal_values": [
        {
            "name": "白细胞计数",
            "value": "10.2 ×10^9/L",
            "reference": "3.5-9.5",
            "status": "high",
            "explanation": "轻度升高，建议结合临床。",
        }
    ],
    "questions": ["需要复查血常规吗？"],
}

TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]")


def _has_image(messages: list[dict[str, Any]]) -> bool:
    for message in messages:
        content = message.get("content")
        if isinstance(content, list) and any(
            isinstance(part, dict) and part.get("type") == "image_url" for part in content
        ):
            return True
    return False


class FakeGateway:
    """Drop-in replacement for ``ModelGateway`` used by integration tests."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> ChatResult:
        self.calls.append({"messages": messages, **kwargs})
        if _has_image(messages):
            return self._result(VISION_REPORT, model="fake-vision")
        task = kwargs.get("task", "diagnosis")
        if task == "triage":
            return self._result(TRIAGE)
        if kwargs.get("tools"):
            return ChatResult(
                content="",
                tool_calls=[ToolCall(id="call_submit_report", name="submit_report", arguments=dict(REPORT))],
                provider="fake",
                model="fake-report",
            )
        return ChatResult(content="建议到心血管内科门诊就诊。", provider="fake", model="fake-text")

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_embed(text) for text in texts]

    async def rerank(self, query: str, documents: list[str], top_n: int = 20) -> list[tuple[int, float]]:
        return [(index, 1.0 - index * 0.01) for index in range(min(top_n, len(documents)))]

    def calls_for(self, task: str) -> list[dict[str, Any]]:
        return [call for call in self.calls if call.get("task") == task]

    def _result(self, payload: dict[str, Any], *, model: str = "fake-triage") -> ChatResult:
        import json

        return ChatResult(
            content=json.dumps(payload, ensure_ascii=False),
            provider="fake",
            model=model,
            prompt_tokens=10,
            completion_tokens=5,
        )

    def _hash_embed(self, text: str) -> list[float]:
        dim = get_settings().EMBEDDING_DIM
        vector = [0.0] * dim
        tokens = TOKEN_RE.findall(text or "") or [text or "empty"]
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % dim
            vector[index] += 1.0 if digest[4] % 2 == 0 else -1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]
