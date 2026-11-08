#!/usr/bin/env python3
"""Deterministic OpenAI-compatible fake model server for E2E tests.

Implements just enough of the OpenAI surface used by the backend:

- POST /v1/chat/completions  (triage JSON, submit_report tool call, vision JSON)
- POST /v1/embeddings        (deterministic 4096-dim vectors)
- POST /v1/rerank            (stable index/relevance pairs)

No network access and no dependencies beyond the Python standard library.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

EMBEDDING_DIM = 4096
MODEL_NAME = "e2e-fake-model"
REPORT_MARKER = "【E2E】"

TRIAGE = {
    "intent": "MEDICAL_CONSULT",
    "department": "心血管内科",
    "urgency": "normal",
    "symptoms": "头痛伴乏力",
    "reason": "E2E fake triage",
    "confidence": 0.86,
}

REPORT = {
    "department": "心血管内科",
    "urgency": "normal",
    "summary": (
        f"{REPORT_MARKER}结合你描述的症状，建议先到心血管内科门诊评估。"
        "注意休息并监测血压；若出现胸痛、呼吸困难或意识不清，请立即前往急诊。"
    ),
    "advice": ["规律作息，避免熬夜", "每日监测血压并记录", "症状加重时及时就医"],
    "medications": [],
    "citations": [],
    "confidence": 0.82,
    "followup_question": "最近是否有胸闷或心悸？",
}

VISION_REPORT = {
    "summary": f"{REPORT_MARKER}检查报告解读：整体未见明显异常，白细胞轻度升高，建议结合临床。",
    "abnormal_values": [
        {
            "name": "白细胞计数",
            "value": "10.2 ×10^9/L",
            "reference": "3.5-9.5",
            "status": "high",
            "explanation": "轻度升高，可能与近期感染或应激有关。",
        }
    ],
    "questions": ["需要复查血常规吗？", "近期是否有发热？"],
}

TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]")


def _vector(text: str) -> list[float]:
    vector = [0.0] * EMBEDDING_DIM
    tokens = TOKEN_RE.findall(text or "") or [text or "empty"]
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "big") % EMBEDDING_DIM
        vector[index] += 1.0 if digest[4] % 2 == 0 else -1.0
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [round(value / norm, 6) for value in vector]


def _has_image(messages: list[dict[str, Any]]) -> bool:
    for message in messages:
        content = message.get("content")
        if isinstance(content, list) and any(
            isinstance(part, dict) and part.get("type") == "image_url" for part in content
        ):
            return True
    return False


def _usage() -> dict[str, int]:
    return {"prompt_tokens": 128, "completion_tokens": 64, "total_tokens": 192}


def _chat_completion(content: str | None, tool_calls: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    finish_reason = "stop"
    if tool_calls:
        message["tool_calls"] = tool_calls
        message["content"] = None
        finish_reason = "tool_calls"
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": _usage(),
    }


def _submit_report_call() -> list[dict[str, Any]]:
    return [
        {
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {
                "name": "submit_report",
                "arguments": json.dumps(REPORT, ensure_ascii=False),
            },
        }
    ]


def _chat_payload(body: dict[str, Any]) -> dict[str, Any]:
    messages = body.get("messages") or []
    if _has_image(messages):
        return _chat_completion(json.dumps(VISION_REPORT, ensure_ascii=False))
    if body.get("tools"):
        return _chat_completion("", _submit_report_call())
    response_format = body.get("response_format") or {}
    if isinstance(response_format, dict) and response_format.get("type") == "json_object":
        return _chat_completion(json.dumps(TRIAGE, ensure_ascii=False))
    return _chat_completion("建议到心血管内科门诊就诊，必要时完善心电图检查。")


def _embeddings_payload(body: dict[str, Any]) -> dict[str, Any]:
    raw = body.get("input")
    inputs = [raw] if isinstance(raw, str) else list(raw or [])
    data = [
        {"object": "embedding", "index": index, "embedding": _vector(str(text))}
        for index, text in enumerate(inputs)
    ]
    return {
        "object": "list",
        "data": data,
        "model": body.get("model") or MODEL_NAME,
        "usage": {"prompt_tokens": len(inputs), "total_tokens": len(inputs)},
    }


def _rerank_payload(body: dict[str, Any]) -> dict[str, Any]:
    documents = body.get("documents") or []
    top_n = int(body.get("top_n") or len(documents) or 0)
    results = [
        {"index": index, "relevance_score": round(1.0 - index * 0.01, 4)}
        for index in range(min(top_n, len(documents)))
    ]
    return {"id": f"rerank-{uuid.uuid4().hex[:12]}", "results": results}


class Handler(BaseHTTPRequestHandler):
    server_version = "E2EFakeModel/1.0"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        print(f"[fake-model] {self.address_string()} {format % args}", flush=True)

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        if path in {"/health", "/api/health"}:
            self._send(200, {"ok": True, "dim": EMBEDDING_DIM})
        elif path == "/v1/models":
            self._send(200, {"object": "list", "data": [{"id": MODEL_NAME, "object": "model"}]})
        else:
            self._send(404, {"error": {"message": f"unknown path: {path}"}})

    def do_POST(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send(400, {"error": {"message": "invalid json"}})
            return
        if path == "/v1/chat/completions":
            self._send(200, _chat_payload(body))
        elif path == "/v1/embeddings":
            self._send(200, _embeddings_payload(body))
        elif path == "/v1/rerank":
            self._send(200, _rerank_payload(body))
        else:
            self._send(404, {"error": {"message": f"unknown path: {path}"}})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9999)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[fake-model] listening on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
