import json
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)

from app.api.v1.endpoints import chat as chat_endpoint


async def _collect_stream_payloads(
    raw_events: list[dict],
    monkeypatch,
    *,
    raise_exc: Exception | None = None,
) -> tuple[list[dict], bool, list[dict]]:
    async def _fake_astream_events(_inputs, config=None, version="v2"):
        del config, version
        for event in raw_events:
            yield event
        if raise_exc is not None:
            raise raise_exc

    monkeypatch.setattr(chat_endpoint.graph_app, "astream_events", _fake_astream_events)
    ensure_calls: list[dict] = []
    monkeypatch.setattr(
        chat_endpoint.langfuse_bridge,
        "ensure_trace",
        lambda *args, **kwargs: ensure_calls.append(kwargs),
    )
    monkeypatch.setattr(chat_endpoint.logger, "error", lambda *args, **kwargs: None)
    monkeypatch.setattr(chat_endpoint.langfuse_bridge, "annotate_trace", lambda *args, **kwargs: None)
    monkeypatch.setattr(chat_endpoint.langfuse_bridge, "finish_trace", lambda *args, **kwargs: None)

    payloads: list[dict] = []
    done_seen = False

    async for sse in chat_endpoint.event_generator(
        message="最近头痛三天，伴恶心",
        session_id="sess-contract-01",
        request_id="req-contract-01",
    ):
        line = sse.strip()
        assert line.startswith("data: ")
        body = line[len("data: ") :]
        if body == "[DONE]":
            done_seen = True
            continue
        payloads.append(json.loads(body))

    return payloads, done_seen, ensure_calls


@pytest.mark.asyncio
async def test_chat_stream_contract_required_order_and_done(monkeypatch):
    monkeypatch.setattr(chat_endpoint.settings, "UPGRADE3_CHAT_SHELL_ENABLED", True, raising=False)
    monkeypatch.setattr(chat_endpoint.settings, "UPGRADE3_DIAGNOSIS_SHELL_ENABLED", True, raising=False)
    monkeypatch.setattr(chat_endpoint.settings, "UPGRADE3_RETRIEVER_PIPELINE_ENABLED", True, raising=False)

    raw_events = [
        {
            "event": "on_chain_start",
            "metadata": {"langgraph_node": "Query_Rewrite"},
            "data": {},
        },
        {
            "event": "on_chain_end",
            "metadata": {"langgraph_node": "Query_Rewrite"},
            "data": {
                "output": {
                    "retrieval_plan": {
                        "rewrite_fallback": True,
                        "rewrite_fallback_reason": "timeout",
                        "crisis_fastlane": False,
                        "effective_runtime_config": {
                            "rewrite_timeout_s": 3.0,
                            "crisis_fastlane": False,
                        },
                    }
                }
            },
        },
        {
            "event": "on_chat_model_stream",
            "metadata": {"langgraph_node": "Diagnosis_Report"},
            "data": {"chunk": SimpleNamespace(content="建议先到神经内科就诊。")},
        },
    ]

    payloads, done_seen, ensure_calls = await _collect_stream_payloads(raw_events, monkeypatch)

    assert done_seen
    assert ensure_calls
    trace_meta = ensure_calls[0]["metadata"]
    assert trace_meta["upgrade3_flags"] == {
        "chat_shell_enabled": True,
        "diagnosis_shell_enabled": True,
        "retriever_pipeline_enabled": True,
    }
    assert payloads[0]["type"] == "status"
    assert payloads[0]["content"] == "stream_opened"

    final_idx = next(i for i, p in enumerate(payloads) if p.get("type") == "final")
    closed_idx = next(
        i
        for i, p in enumerate(payloads)
        if p.get("type") == "status" and p.get("content") == "stream_closed"
    )
    assert final_idx < closed_idx

    seqs = [p["seq"] for p in payloads if "seq" in p]
    assert seqs == sorted(seqs)
    assert len(seqs) == len(set(seqs))


@pytest.mark.asyncio
async def test_chat_stream_contract_optional_rewrite_path(monkeypatch):
    raw_events = [
        {
            "event": "on_chain_end",
            "metadata": {"langgraph_node": "Query_Rewrite"},
            "data": {
                "output": {
                    "retrieval_plan": {
                        "rewrite_fallback": False,
                        "rewrite_fallback_reason": "",
                        "crisis_fastlane": False,
                        "effective_runtime_config": {
                            "rewrite_timeout_s": 4.0,
                            "crisis_fastlane": False,
                        },
                    }
                }
            },
        },
        {
            "event": "on_chat_model_stream",
            "metadata": {"langgraph_node": "Diagnosis_Report"},
            "data": {"chunk": SimpleNamespace(content="可先门诊评估。")},
        },
    ]

    payloads, done_seen, _ = await _collect_stream_payloads(raw_events, monkeypatch)

    assert done_seen

    status_contents = [
        p.get("content")
        for p in payloads
        if p.get("type") == "status" and p.get("node") == "Query_Rewrite"
    ]
    assert "runtime_config_applied" in status_contents
    assert "rewrite_path" not in status_contents


@pytest.mark.asyncio
async def test_chat_stream_contract_error_event_and_done(monkeypatch):
    raw_events = [
        {
            "event": "on_chat_model_stream",
            "metadata": {"langgraph_node": "Diagnosis_Report"},
            "data": {"chunk": SimpleNamespace(content="先进行基础检查。")},
        },
    ]

    payloads, done_seen, _ = await _collect_stream_payloads(
        raw_events,
        monkeypatch,
        raise_exc=RuntimeError("upstream_crash"),
    )

    assert done_seen
    assert any(p.get("type") == "error" for p in payloads)
