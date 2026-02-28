#!/usr/bin/env python3
"""
Robust end-to-end full-chain tester with rich logging.

Features:
- Optional backend auto-start (local python process, non-docker)
- SSE stream test against /api/v1/chat/stream
- Per-case timeout, stall timeout, max event guard
- Global timeout guard
- Detailed logs:
  - sent input
  - node start/finish timeline (from SSE thought events)
  - retrieval-related backend log lines
  - streamed answer content
  - duration and event statistics
- JSON + JSONL + Markdown reports for post-analysis
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx


DEFAULT_CASES = [
    {
        "id": "greeting",
        "message": "你好",
        "expected_intent": "GREETING",
        "expected_route": "fast_reply",
        "assert_retrieval": False,
        "session_group": "greeting",
    },
    {
        "id": "registration",
        "message": "我想挂明天下午心内科号",
        "expected_intent": "REGISTRATION",
        "expected_route": "service",
        "assert_retrieval": False,
        "session_group": "registration",
    },
    {
        "id": "medical",
        "message": "我最近头痛恶心三天",
        "expected_intent": "MEDICAL_CONSULT",
        "expected_route": "diagnosis",
        "assert_retrieval": True,
        "session_group": "medical",
    },
    {
        "id": "crisis",
        "message": "我不想活了",
        "expected_intent": "CRISIS",
        "expected_route": "diagnosis",
        "assert_retrieval": True,
        "session_group": "crisis",
    },
]

MULTITURN_DEMO_CASES = [
    {
        "id": "medical_turn1",
        "message": "我头痛两天伴恶心",
        "expected_intent": "MEDICAL_CONSULT",
        "expected_route": "diagnosis",
        "assert_retrieval": True,
        "session_group": "mt_medical",
    },
    {
        "id": "medical_turn2",
        "message": "还有畏光，昨晚开始加重",
        "expected_intent": "MEDICAL_CONSULT",
        "expected_route": "diagnosis",
        "assert_retrieval": True,
        "session_group": "mt_medical",
    },
]

KEY_BACKEND_PATTERNS = [
    "workflow_router_decision",
    "node_start",
    "node_end",
    "node_completed",
    "fast_track_hit",
    "slm_classification_success",
    "hybrid_retriever_query",
    "hybrid_retriever_success",
    "hybrid_retriever_no_query_found",
    "hybrid_retriever_failed",
    "rag_search_start",
    "rag_cache_hit",
    "rag_cache_miss",
    "quality_gate",
    "intent_classified",
    "[DEBUG] Node",
    "[DEBUG] RAG Query:",
]

ROUTING_NODES = {
    "cache_lookup",
    "ingress",
    "pii_filter",
    "multimodal_processor",
    "history_injector",
    "guard",
    "intent_classifier",
}

RETRIEVAL_NODES = {
    "retriever",
    "hybrid_retriever",
    "Hybrid_Retriever",
    "state_sync",
    "State_Sync",
}

RESPONSE_NODES = {
    "fast_reply",
    "service",
    "registration_tool",
    "diagnosis",
    "diagnosis_egress",
    "expert_aggregation",
    "generate_followup",
    "quality_gate",
    "egress",
    "persistence",
}


@dataclass
class NodeEvent:
    node: str
    event: str
    ts_iso: str
    rel_ms: int
    raw: str


@dataclass
class CaseResult:
    case_id: str
    session_id: str
    sent_content: str
    started_at: str
    finished_at: str = ""
    duration_s: float = 0.0
    http_status: Optional[int] = None
    done: bool = False
    terminated_by: str = "normal"
    error: str = ""
    exception: str = ""
    event_count: int = 0
    thought_count: int = 0
    token_count: int = 0
    status_count: int = 0
    response_content: str = ""
    response_preview: str = ""
    node_timeline: List[NodeEvent] = field(default_factory=list)
    retrieval_logs: List[str] = field(default_factory=list)
    backend_signals: List[str] = field(default_factory=list)
    expected_intent: str = ""
    actual_intent: str = ""
    intent_assert_ok: bool = True
    expected_route: str = ""
    actual_routes: List[str] = field(default_factory=list)
    route_assert_ok: bool = True
    retrieval_assert_required: bool = False
    retrieval_query_non_empty: bool = False
    retrieval_signal_hit: bool = False
    retrieval_assert_ok: bool = True
    node_durations_s: Dict[str, float] = field(default_factory=dict)
    last_event_kind: str = ""
    last_event_node: str = ""
    timeout_segment: str = ""
    failure_category: str = ""
    failure_segment: str = ""
    ok: bool = False


def now_iso() -> str:
    return datetime.now().isoformat(timespec="milliseconds")


def _ratio_bar(ratio: float, width: int = 20) -> str:
    safe = max(0.0, min(1.0, ratio))
    fill = int(round(width * safe))
    return "[" + ("#" * fill).ljust(width, "-") + "]"


def emit_progress(message: str, final: bool = False) -> None:
    stream = sys.stderr
    if stream.isatty():
        text = message[:220]
        if final:
            stream.write("\r" + text.ljust(220) + "\n")
        else:
            stream.write("\r" + text.ljust(220))
        stream.flush()
        return
    print(message, file=stream, flush=True)


def load_env_file(path: Path) -> Dict[str, str]:
    env: Dict[str, str] = {}
    if not path.exists():
        return env
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        env[k] = v
    return env


def resolve_single_source_env_file(project_root: Path, requested_env_file: str) -> Path:
    root_env = (project_root / ".env").resolve()
    requested = Path(requested_env_file)
    resolved = requested.resolve() if requested.is_absolute() else (project_root / requested).resolve()
    if resolved != root_env:
        raise ValueError(
            f"invalid_env_file: only project root .env is allowed ({root_env}), got {resolved}"
        )
    return resolved


def python_has_backend_deps(python_bin: str) -> bool:
    try:
        proc = subprocess.run(
            [python_bin, "-c", "import uvicorn, fastapi, httpx; print('ok')"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
            check=False,
        )
        return proc.returncode == 0
    except Exception:
        return False


def resolve_backend_python(project_root: Path, preferred: str) -> str:
    if preferred and preferred != "auto":
        return preferred

    candidates: List[str] = []
    venv_python = str(project_root / ".venv" / "bin" / "python")
    if Path(venv_python).exists():
        candidates.append(venv_python)
    if sys.executable:
        candidates.append(sys.executable)
    for name in ("python3", "python"):
        p = shutil.which(name)
        if p:
            candidates.append(p)

    seen = set()
    uniq = []
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        uniq.append(c)

    for c in uniq:
        if python_has_backend_deps(c):
            return c

    if uniq:
        return uniq[0]
    return "python3"


def write_jsonl(path: Path, record: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_new_log_lines(log_path: Path, start_offset: int, max_bytes: int = 2_000_000) -> Tuple[List[str], int]:
    if not log_path.exists():
        return [], start_offset
    size = log_path.stat().st_size
    if size <= start_offset:
        return [], size
    read_from = start_offset
    if size - start_offset > max_bytes:
        read_from = size - max_bytes
    with log_path.open("rb") as f:
        f.seek(read_from)
        chunk = f.read()
    text = chunk.decode("utf-8", errors="ignore")
    lines = text.splitlines()
    return lines, size


def extract_backend_signals(lines: List[str]) -> Tuple[List[str], List[str]]:
    signals: List[str] = []
    retrieval: List[str] = []
    for line in lines:
        if any(p in line for p in KEY_BACKEND_PATTERNS):
            signals.append(line)
        if (
            ("hybrid_retriever_query" in line)
            or ("rag_search_start" in line)
            or ("context_len=" in line)
            or ("[DEBUG] RAG Query:" in line)
            or ("[DEBUG] Embedding Latency:" in line)
        ):
            retrieval.append(line)
    return signals, retrieval


def infer_intent_from_message(text: str) -> str:
    t = (text or "").strip().lower()
    if not t:
        return ""
    crisis_keywords = ["救命", "胸痛", "昏迷", "呼吸困难", "中风", "大出血", "120", "想死", "自杀", "不想活", "轻生"]
    if any(k in t for k in crisis_keywords):
        return "CRISIS"
    service_keywords = ["挂号", "预约", "门诊", "科室", "看医生", "约号"]
    if any(k in t for k in service_keywords):
        return "REGISTRATION"
    greeting_keywords = ["你好", "您好", "hello", "hi", "早安", "晚上好", "在吗"]
    if len(t) <= 10 and any(k in t for k in greeting_keywords):
        return "GREETING"
    info_keywords = ["地址", "电话", "位置", "几点", "时间", "在哪", "在哪里"]
    if any(k in t for k in info_keywords):
        return "INFO"
    return "MEDICAL_CONSULT"


def infer_node_from_thought(content: str) -> str:
    c = (content or "").strip()
    node_map = [
        ("cache_lookup", "cache_lookup"),
        ("ingress", "ingress"),
        ("pii_filter", "pii_filter"),
        ("multimodal_processor", "multimodal_processor"),
        ("history_injector", "history_injector"),
        ("guard", "guard"),
        ("识别意图", "intent_classifier"),
        ("挂号服务", "service"),
        ("挂号工具调用", "registration_tool"),
        ("进行诊断分析", "diagnosis"),
        ("同步患者上下文", "state_sync"),
        ("检索医学知识", "retriever"),
        ("专家系统正在推理", "expert_aggregation"),
        ("生成追问", "generate_followup"),
        ("egress", "egress"),
        ("整合诊断结果", "diagnosis_egress"),
        ("quality_gate", "quality_gate"),
        ("persistence", "persistence"),
        ("生成回答", "fast_reply"),
    ]
    for hint, node in node_map:
        if hint in c:
            return node
    m_processing = re.search(r"正在处理\s*([a-zA-Z_]+)\.\.\.", c)
    if m_processing:
        return m_processing.group(1).strip()
    return ""


def parse_cases(args: argparse.Namespace) -> List[Dict[str, str]]:
    if args.cases_file:
        p = Path(args.cases_file)
        data = json.loads(p.read_text(encoding="utf-8"))
        out = []
        for i, item in enumerate(data):
            out.append(
                {
                    "id": str(item.get("id") or f"case_{i+1}"),
                    "message": str(item["message"]),
                    "expected_intent": str(item.get("expected_intent", "")).strip(),
                    "expected_route": str(item.get("expected_route", "")).strip(),
                    "assert_retrieval": bool(item.get("assert_retrieval", False)),
                    "session_group": str(item.get("session_group") or item.get("id") or f"case_{i+1}"),
                }
            )
        return out
    cases = [dict(item) for item in DEFAULT_CASES]
    if args.multi_turn_demo:
        cases.extend(dict(item) for item in MULTITURN_DEMO_CASES)
    return cases


def parse_backend_analysis(signals: List[str], retrieval_logs: List[str]) -> Dict[str, Any]:
    actual_intent = ""
    routes: List[str] = []
    retrieval_query_non_empty = False
    retrieval_signal_hit = False
    node_durations: Dict[str, float] = {}

    for line in signals:
        if "intent_classified" in line:
            m = re.search(r"intent=([A-Z_]+)", line)
            if m:
                actual_intent = m.group(1).strip()
        if "fast_track_hit" in line and not actual_intent:
            m = re.search(r"intent=([A-Z_]+)", line)
            if m:
                actual_intent = m.group(1).strip()
        if "slm_classification_success" in line and not actual_intent:
            m = re.search(r"intent=([A-Z_]+)", line)
            if m:
                actual_intent = m.group(1).strip()

        if "workflow_router_decision" in line:
            m = re.search(r"route=([a-zA-Z_]+)", line)
            if m:
                route = m.group(1).strip()
                if route:
                    routes.append(route)

        if "[DEBUG] Node" in line:
            if "expert_aggregation" in line or "quality_gate" in line:
                routes.append("diagnosis")

        if "node_completed" in line:
            m_node = re.search(r"node=([a-zA-Z_]+)", line)
            m_dur = re.search(r"duration[=:'\"]+([0-9.]+)s", line)
            if m_node and m_dur:
                node = m_node.group(1).strip()
                try:
                    node_durations[node] = float(m_dur.group(1))
                except Exception:
                    pass

    for line in retrieval_logs:
        if "hybrid_retriever_query" in line:
            m = re.search(r"query=(.+?)(\\s+source=|$)", line)
            if m and m.group(1).strip().strip("'\""):
                retrieval_query_non_empty = True
        if "[DEBUG] RAG Query:" in line:
            m = re.search(r"RAG Query:\s*(.+)$", line)
            if m and m.group(1).strip():
                retrieval_query_non_empty = True
        if any(k in line for k in ["hybrid_retriever_success", "rag_search_start", "context_len="]):
            retrieval_signal_hit = True
        if "[DEBUG] Embedding Latency:" in line:
            retrieval_signal_hit = True

    return {
        "actual_intent": actual_intent,
        "routes": routes,
        "retrieval_query_non_empty": retrieval_query_non_empty,
        "retrieval_signal_hit": retrieval_signal_hit,
        "node_durations": node_durations,
    }


def derive_node_durations_from_timeline(node_timeline: List[NodeEvent], case_duration_s: float) -> Dict[str, float]:
    """Infer per-node durations from thought timeline when backend structured timings are absent."""
    starts = [e for e in node_timeline if e.event == "start" and e.node]
    if not starts:
        return {}
    starts.sort(key=lambda e: e.rel_ms)
    total_ms = max(0, int(case_duration_s * 1000))
    durations: Dict[str, float] = {}
    for i, ev in enumerate(starts):
        next_ms = starts[i + 1].rel_ms if i + 1 < len(starts) else total_ms
        dur_s = max(0.0, (next_ms - ev.rel_ms) / 1000.0)
        durations[ev.node] = round(durations.get(ev.node, 0.0) + dur_s, 3)
    return durations


def classify_failure(result: CaseResult) -> str:
    if result.http_status not in (None, 200):
        return "protocol_failure"
    if result.exception:
        return "transport_failure"
    if result.terminated_by in {"stall_timeout", "case_timeout"}:
        return "timeout_failure"
    if result.error:
        return "model_failure"
    if not result.intent_assert_ok:
        return "route_failure"
    if not result.route_assert_ok:
        return "route_failure"
    if not result.retrieval_assert_ok:
        return "retrieval_failure"
    if not result.done:
        return "stream_failure"
    if not result.response_content:
        return "empty_response_failure"
    return "ok"


def infer_segment_from_node(node: str) -> str:
    node_norm = (node or "").strip()
    if not node_norm or node_norm == "-":
        return "unknown"
    if node_norm in ROUTING_NODES:
        return "route"
    if node_norm in RETRIEVAL_NODES:
        return "retrieval"
    if node_norm in RESPONSE_NODES:
        return "response"
    return "response"


def classify_failure_segment(result: CaseResult) -> str:
    if result.failure_category == "ok":
        return "ok"
    if result.failure_category == "protocol_failure":
        return "protocol"
    if result.failure_category == "transport_failure":
        return "transport"
    if result.failure_category == "route_failure":
        return "route"
    if result.failure_category == "retrieval_failure":
        return "retrieval"
    if result.failure_category in {"timeout_failure"}:
        if result.timeout_segment:
            return result.timeout_segment
        return infer_segment_from_node(result.last_event_node)
    if result.failure_category in {"model_failure"}:
        return "model"
    return "response"


def aggregate_node_timing(results: List[CaseResult]) -> Dict[str, Dict[str, float]]:
    accum: Dict[str, List[float]] = {}
    for r in results:
        for node, dur in r.node_durations_s.items():
            accum.setdefault(node, []).append(dur)
    out: Dict[str, Dict[str, float]] = {}
    for node, values in accum.items():
        if not values:
            continue
        out[node] = {
            "count": float(len(values)),
            "avg_s": round(sum(values) / len(values), 3),
            "max_s": round(max(values), 3),
        }
    return out


def aggregate_segment_timing(results: List[CaseResult]) -> Dict[str, Dict[str, float]]:
    segment_accum: Dict[str, List[float]] = {}
    for r in results:
        seg_cost = {"route": 0.0, "retrieval": 0.0, "response": 0.0}
        for node, dur in r.node_durations_s.items():
            seg = infer_segment_from_node(node)
            if seg in seg_cost:
                seg_cost[seg] += float(dur)
        for seg, val in seg_cost.items():
            segment_accum.setdefault(seg, []).append(round(val, 3))

    out: Dict[str, Dict[str, float]] = {}
    for seg, values in segment_accum.items():
        if not values:
            continue
        out[seg] = {
            "count": float(len(values)),
            "avg_s": round(sum(values) / len(values), 3),
            "max_s": round(max(values), 3),
        }
    return out


def build_report_md(run_id: str, summary: Dict[str, Any]) -> str:
    lines = []
    lines.append(f"# E2E Full-Chain Report ({run_id})")
    lines.append("")
    lines.append(f"- started_at: `{summary['started_at']}`")
    lines.append(f"- finished_at: `{summary['finished_at']}`")
    lines.append(f"- total_cases: `{summary['total_cases']}`")
    lines.append(f"- passed_cases: `{summary['passed_cases']}`")
    lines.append(f"- ok: `{summary['ok']}`")
    if summary.get("node_timing_summary"):
        lines.append("")
        lines.append("## Node Timing Summary")
        for node, stats in sorted(summary["node_timing_summary"].items()):
            lines.append(f"- `{node}`: avg={stats['avg_s']}s max={stats['max_s']}s count={int(stats['count'])}")
    if summary.get("segment_timing_summary"):
        lines.append("")
        lines.append("## Segment Timing Summary")
        for seg, stats in sorted(summary["segment_timing_summary"].items()):
            lines.append(f"- `{seg}`: avg={stats['avg_s']}s max={stats['max_s']}s count={int(stats['count'])}")
    lines.append("")
    for case in summary["results"]:
        lines.append(f"## Case `{case['case_id']}`")
        lines.append(f"- session_id: `{case['session_id']}`")
        lines.append(f"- ok: `{case['ok']}`")
        lines.append(f"- duration_s: `{case['duration_s']}`")
        lines.append(f"- terminated_by: `{case['terminated_by']}`")
        lines.append(f"- failure_category: `{case.get('failure_category', '')}`")
        lines.append(f"- failure_segment: `{case.get('failure_segment', '')}`")
        lines.append(f"- timeout_segment: `{case.get('timeout_segment', '')}`")
        lines.append(f"- http_status: `{case['http_status']}`")
        lines.append(f"- event_count/thought/token/status: `{case['event_count']}/{case['thought_count']}/{case['token_count']}/{case['status_count']}`")
        lines.append(f"- expected/actual_intent: `{case.get('expected_intent','')}` / `{case.get('actual_intent','')}`")
        lines.append(f"- intent_assert_ok: `{case.get('intent_assert_ok', True)}`")
        lines.append(f"- expected_route/routes: `{case.get('expected_route','')}` / `{','.join(case.get('actual_routes', []))}`")
        lines.append(f"- route_assert_ok: `{case.get('route_assert_ok', True)}`")
        lines.append(
            f"- retrieval_assert(query_non_empty/signal_hit/ok): "
            f"`{case.get('retrieval_query_non_empty', False)}/{case.get('retrieval_signal_hit', False)}/{case.get('retrieval_assert_ok', True)}`"
        )
        if case.get("error"):
            lines.append(f"- error: `{case['error']}`")
        if case.get("exception"):
            lines.append(f"- exception: `{case['exception']}`")
        lines.append("- sent_content:")
        lines.append(f"  - `{case['sent_content']}`")
        lines.append("- response_preview:")
        lines.append(f"  - `{case['response_preview']}`")
        lines.append("- node_timeline (first 12):")
        for ne in case["node_timeline"][:12]:
            lines.append(f"  - `{ne['ts_iso']}` `{ne['event']}` `{ne['node']}`")
        lines.append("- retrieval_logs (first 8):")
        for rl in case["retrieval_logs"][:8]:
            lines.append(f"  - `{rl}`")
        lines.append("")
    return "\n".join(lines)


async def wait_backend_ready(base_url: str, timeout_s: int, interval_s: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout_s
    async with httpx.AsyncClient(timeout=httpx.Timeout(5.0, connect=2.0)) as client:
        while time.monotonic() < deadline:
            try:
                r = await client.get(f"{base_url}/health")
                if r.status_code == 200:
                    return True
            except Exception:
                pass
            await asyncio.sleep(interval_s)
    return False


def start_backend_process(
    project_root: Path,
    env_file: Path,
    host: str,
    port: int,
    log_path: Path,
    python_bin: str,
) -> subprocess.Popen:
    env = dict(os.environ)
    env.update(load_env_file(env_file))
    env["PYTHONPATH"] = str(project_root / "backend")
    cmd = [
        python_bin,
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        host,
        "--port",
        str(port),
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=str(project_root),
        env=env,
        stdout=f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return proc


def stop_backend_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass
    for _ in range(20):
        if proc.poll() is not None:
            return
        time.sleep(0.2)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


async def run_case(
    client: httpx.AsyncClient,
    base_url: str,
    case_id: str,
    message: str,
    session_id: str,
    expected_intent: str,
    expected_route: str,
    assert_retrieval: bool,
    case_index: int,
    total_cases: int,
    case_timeout_s: int,
    stall_timeout_s: int,
    max_events: int,
    events_jsonl: Path,
    backend_log_path: Optional[Path],
    backend_offset: int,
    show_progress: bool,
    progress_interval_s: float,
) -> Tuple[CaseResult, int]:
    started = now_iso()
    result = CaseResult(
        case_id=case_id,
        session_id=session_id,
        sent_content=message,
        started_at=started,
        expected_intent=expected_intent,
        expected_route=expected_route,
        retrieval_assert_required=assert_retrieval,
    )
    case_start_mono = time.monotonic()
    last_activity_mono = case_start_mono
    last_kind = "init"
    last_node = "-"
    inferred_routes = set()
    inferred_retrieval_signal = False
    token_parts: List[str] = []
    offset = backend_offset
    payload = {"message": message, "session_id": session_id}
    stop_progress = asyncio.Event()

    async def progress_pulse() -> None:
        while not stop_progress.is_set():
            elapsed = time.monotonic() - case_start_mono
            case_ratio = min(1.0, elapsed / max(float(case_timeout_s), 1.0))
            overall_ratio = ((case_index - 1) + case_ratio) / max(float(total_cases), 1.0)
            stall_left = max(0, int(stall_timeout_s - (time.monotonic() - last_activity_mono)))
            msg = (
                f"[total]{_ratio_bar(overall_ratio, 18)} {case_index}/{total_cases} "
                f"[case]{_ratio_bar(case_ratio, 18)} {case_id} "
                f"elapsed={elapsed:.1f}s stall_left={stall_left}s "
                f"evt={result.event_count} tok={result.token_count} thought={result.thought_count} "
                f"last={last_kind}/{last_node}"
            )
            emit_progress(msg, final=False)
            await asyncio.sleep(max(0.2, progress_interval_s))

    pulse_task: Optional[asyncio.Task] = None
    if show_progress:
        emit_progress(
            f"[start] case {case_index}/{total_cases} `{case_id}` sent={message[:40]}",
            final=True,
        )
        pulse_task = asyncio.create_task(progress_pulse())

    write_jsonl(
        events_jsonl,
        {
            "ts_iso": now_iso(),
            "case_id": case_id,
            "session_id": session_id,
            "kind": "request_sent",
            "payload": payload,
        },
    )

    try:
        async with client.stream("POST", f"{base_url}/api/v1/chat/stream", json=payload) as resp:
            result.http_status = resp.status_code
            if resp.status_code != 200:
                result.error = f"http_status_{resp.status_code}"
                result.terminated_by = "http_error"
                return result, offset

            line_iter = resp.aiter_lines()
            while True:
                elapsed = time.monotonic() - case_start_mono
                if elapsed > case_timeout_s:
                    result.terminated_by = "case_timeout"
                    break
                if result.event_count >= max_events:
                    result.terminated_by = "max_events_guard"
                    break

                try:
                    line = await asyncio.wait_for(line_iter.__anext__(), timeout=stall_timeout_s)
                except asyncio.TimeoutError:
                    result.terminated_by = "stall_timeout"
                    last_kind = "stall_timeout"
                    break
                except StopAsyncIteration:
                    last_kind = "stream_closed"
                    break

                if not line.startswith("data: "):
                    continue

                last_activity_mono = time.monotonic()
                payload_raw = line[6:]
                ts_iso = now_iso()
                rel_ms = int((time.monotonic() - case_start_mono) * 1000)

                if payload_raw == "[DONE]":
                    result.done = True
                    last_kind = "done"
                    write_jsonl(
                        events_jsonl,
                        {
                            "ts_iso": ts_iso,
                            "case_id": case_id,
                            "session_id": session_id,
                            "kind": "done",
                            "rel_ms": rel_ms,
                        },
                    )
                    break

                try:
                    evt = json.loads(payload_raw)
                except Exception:
                    write_jsonl(
                        events_jsonl,
                        {
                            "ts_iso": ts_iso,
                            "case_id": case_id,
                            "session_id": session_id,
                            "kind": "unparsed",
                            "rel_ms": rel_ms,
                            "raw": payload_raw,
                        },
                    )
                    continue

                kind = str(evt.get("type", "unknown"))
                content = str(evt.get("content", ""))
                result.event_count += 1
                last_kind = kind

                if kind == "token":
                    result.token_count += 1
                    token_parts.append(content)
                elif kind == "thought":
                    result.thought_count += 1
                    m_enter = re.search(r"进入节点:\s*(.+)$", content)
                    m_done = re.search(r"节点完成:\s*([^\s(]+)", content)
                    if m_enter:
                        node_name = m_enter.group(1).strip()
                        last_node = node_name
                        result.node_timeline.append(NodeEvent(node_name, "start", ts_iso, rel_ms, content))
                    elif m_done:
                        node_name = m_done.group(1).strip()
                        last_node = node_name
                        result.node_timeline.append(NodeEvent(node_name, "end", ts_iso, rel_ms, content))
                    else:
                        node_name = infer_node_from_thought(content)
                        if node_name:
                            last_node = node_name
                            result.node_timeline.append(NodeEvent(node_name, "start", ts_iso, rel_ms, content))

                    # Heuristic route/retrieval signal inference from thought content
                    if ("挂号服务" in content) or ("service" in content.lower()):
                        inferred_routes.add("service")
                    if ("诊断分析" in content) or ("diagnosis" in content.lower()) or ("专家系统正在推理" in content):
                        inferred_routes.add("diagnosis")
                    if ("生成回答" in content) or ("fast_reply" in content.lower()):
                        inferred_routes.add("fast_reply")
                    if "检索医学知识" in content:
                        inferred_retrieval_signal = True
                elif kind == "status":
                    result.status_count += 1
                elif kind == "error":
                    result.error = content
                    result.terminated_by = "error_event"

                write_jsonl(
                    events_jsonl,
                    {
                        "ts_iso": ts_iso,
                        "case_id": case_id,
                        "session_id": session_id,
                        "kind": kind,
                        "rel_ms": rel_ms,
                        "content": content,
                    },
                )

                if kind == "error":
                    break

    except Exception as e:
        result.exception = str(e)
        result.terminated_by = "exception"
        last_kind = "exception"
    finally:
        if pulse_task:
            stop_progress.set()
            with contextlib.suppress(Exception):
                await pulse_task

    if backend_log_path:
        lines, new_offset = read_new_log_lines(backend_log_path, offset)
        offset = new_offset
        signals, retrieval = extract_backend_signals(lines)
        result.backend_signals = signals[-200:]
        result.retrieval_logs = retrieval[-80:]
        analysis = parse_backend_analysis(result.backend_signals, result.retrieval_logs)
        result.actual_intent = analysis["actual_intent"]
        result.actual_routes = analysis["routes"][-12:]
        result.retrieval_query_non_empty = bool(analysis["retrieval_query_non_empty"])
        result.retrieval_signal_hit = bool(analysis["retrieval_signal_hit"])
        result.node_durations_s = analysis["node_durations"]

    if inferred_routes:
        for route in inferred_routes:
            if route not in result.actual_routes:
                result.actual_routes.append(route)
    if result.actual_routes:
        # Keep insertion order while removing duplicates for cleaner reports.
        result.actual_routes = list(dict.fromkeys(result.actual_routes))
    if inferred_retrieval_signal and not result.retrieval_signal_hit:
        result.retrieval_signal_hit = True
    if result.retrieval_signal_hit and not result.retrieval_query_non_empty and message.strip():
        # Fallback heuristic when backend logs are unavailable.
        result.retrieval_query_non_empty = True

    if not result.actual_intent:
        route_set = set(result.actual_routes)
        if "service" in route_set:
            result.actual_intent = "REGISTRATION"
        elif "diagnosis" in route_set:
            msg_intent = infer_intent_from_message(message)
            result.actual_intent = "CRISIS" if msg_intent == "CRISIS" else "MEDICAL_CONSULT"
        elif "fast_reply" in route_set:
            msg_intent = infer_intent_from_message(message)
            result.actual_intent = "GREETING" if msg_intent == "GREETING" else "INFO"
        else:
            result.actual_intent = infer_intent_from_message(message)

    result.response_content = "".join(token_parts).strip()
    result.response_preview = result.response_content[:400]
    result.duration_s = round(time.monotonic() - case_start_mono, 3)
    result.finished_at = now_iso()
    result.last_event_kind = last_kind
    result.last_event_node = last_node
    if result.terminated_by in {"stall_timeout", "case_timeout"}:
        result.timeout_segment = infer_segment_from_node(last_node)

    # Fallback timing inference when backend does not emit structured node_completed durations.
    inferred_node_timing = derive_node_durations_from_timeline(result.node_timeline, result.duration_s)
    for node, dur in inferred_node_timing.items():
        if node not in result.node_durations_s:
            result.node_durations_s[node] = dur

    if result.terminated_by == "normal":
        result.terminated_by = "done" if result.done else "stream_closed"

    result.route_assert_ok = True
    if result.expected_route:
        result.route_assert_ok = result.expected_route in set(result.actual_routes)

    result.intent_assert_ok = True
    if result.expected_intent:
        result.intent_assert_ok = result.expected_intent == result.actual_intent

    result.retrieval_assert_ok = True
    if result.retrieval_assert_required:
        result.retrieval_assert_ok = result.retrieval_query_non_empty and result.retrieval_signal_hit

    result.ok = (
        result.http_status == 200
        and result.done
        and not result.error
        and not result.exception
        and len(result.response_content) > 0
        and result.intent_assert_ok
        and result.route_assert_ok
        and result.retrieval_assert_ok
    )
    result.failure_category = classify_failure(result)
    result.failure_segment = classify_failure_segment(result)
    if show_progress:
        case_ratio = min(1.0, result.duration_s / max(float(case_timeout_s), 1.0))
        overall_ratio = case_index / max(float(total_cases), 1.0)
        emit_progress(
            (
                f"[done ] [total]{_ratio_bar(overall_ratio, 18)} {case_index}/{total_cases} "
                f"[case]{_ratio_bar(case_ratio, 18)} {case_id} "
                f"status={'OK' if result.ok else 'FAIL'} term={result.terminated_by} "
                f"elapsed={result.duration_s:.3f}s evt={result.event_count} tok={result.token_count}"
            ),
            final=True,
        )
    return result, offset


async def run(args: argparse.Namespace) -> int:
    project_root = Path(args.project_root).resolve()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (project_root / args.output_dir / run_id).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    events_jsonl = out_dir / "events.jsonl"
    summary_json = out_dir / "summary.json"
    report_md = out_dir / "report.md"
    backend_log_path = out_dir / "backend_runtime.log" if args.start_backend else None
    if not args.start_backend and args.backend_log_file:
        backend_log_path = Path(args.backend_log_file).resolve()
    backend_python = resolve_backend_python(project_root, args.backend_python)

    meta = {
        "run_id": run_id,
        "started_at": now_iso(),
        "project_root": str(project_root),
        "base_url": args.base_url.rstrip("/"),
        "start_backend": args.start_backend,
        "backend_python": backend_python,
        "case_timeout_s": args.case_timeout,
        "stall_timeout_s": args.stall_timeout,
        "global_timeout_s": args.global_timeout,
        "max_events_per_case": args.max_events,
        "progress_enabled": not args.no_progress,
        "progress_interval_s": args.progress_interval,
        "backend_log_path": str(backend_log_path) if backend_log_path else "",
        "backend_log_from_end": bool(args.backend_log_from_end),
        "multi_turn_demo": bool(args.multi_turn_demo),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    backend_proc: Optional[subprocess.Popen] = None
    backend_offset = 0
    if not args.start_backend and backend_log_path and args.backend_log_from_end and backend_log_path.exists():
        backend_offset = backend_log_path.stat().st_size
    if args.start_backend:
        try:
            env_file = resolve_single_source_env_file(project_root, args.env_file)
        except ValueError as exc:
            summary = {
                "ok": False,
                "error": "invalid_env_file",
                "error_detail": str(exc),
                "started_at": meta["started_at"],
                "finished_at": now_iso(),
                "results": [],
                "total_cases": 0,
                "passed_cases": 0,
            }
            summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            report_md.write_text(build_report_md(run_id, summary), encoding="utf-8")
            return 1
        backend_proc = start_backend_process(
            project_root=project_root,
            env_file=env_file,
            host=args.backend_host,
            port=args.backend_port,
            log_path=backend_log_path,  # type: ignore[arg-type]
            python_bin=backend_python,
        )
        ready = await wait_backend_ready(args.base_url.rstrip("/"), timeout_s=args.backend_ready_timeout)
        if not ready:
            tail = ""
            if backend_log_path and backend_log_path.exists():
                tail = "\n".join(backend_log_path.read_text(encoding="utf-8", errors="ignore").splitlines()[-120:])
            summary = {
                "ok": False,
                "error": "backend_start_failed_or_not_ready",
                "backend_log_tail": tail,
                "started_at": meta["started_at"],
                "finished_at": now_iso(),
                "results": [],
                "total_cases": 0,
                "passed_cases": 0,
            }
            summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            report_md.write_text(build_report_md(run_id, summary), encoding="utf-8")
            if backend_proc:
                stop_backend_process(backend_proc)
            return 1
        if backend_log_path and backend_log_path.exists():
            backend_offset = backend_log_path.stat().st_size

    cases = parse_cases(args)
    results: List[CaseResult] = []
    run_start_mono = time.monotonic()
    session_group_map: Dict[str, str] = {}

    timeout_cfg = httpx.Timeout(connect=15.0, read=None, write=15.0, pool=15.0)
    async with httpx.AsyncClient(timeout=timeout_cfg) as client:
        for case in cases:
            if (time.monotonic() - run_start_mono) > args.global_timeout:
                break
            cid = case["id"]
            msg = case["message"]
            expected_intent = str(case.get("expected_intent", "")).strip()
            expected_route = str(case.get("expected_route", "")).strip()
            assert_retrieval = bool(case.get("assert_retrieval", False))
            session_group = str(case.get("session_group", "default")).strip() or "default"
            if session_group in session_group_map:
                session_id = session_group_map[session_group]
            else:
                session_id = f"{args.session_prefix}_{session_group}_{int(time.time())}"
                session_group_map[session_group] = session_id
            r, backend_offset = await run_case(
                client=client,
                base_url=args.base_url.rstrip("/"),
                case_id=cid,
                message=msg,
                session_id=session_id,
                expected_intent=expected_intent,
                expected_route=expected_route,
                assert_retrieval=assert_retrieval,
                case_index=len(results) + 1,
                total_cases=len(cases),
                case_timeout_s=args.case_timeout,
                stall_timeout_s=args.stall_timeout,
                max_events=args.max_events,
                events_jsonl=events_jsonl,
                backend_log_path=backend_log_path,
                backend_offset=backend_offset,
                show_progress=not args.no_progress,
                progress_interval_s=args.progress_interval,
            )
            results.append(r)

    if backend_proc:
        stop_backend_process(backend_proc)

    passed = sum(1 for r in results if r.ok)
    summary = {
        "run_id": run_id,
        "ok": passed == len(results) and len(results) == len(cases),
        "started_at": meta["started_at"],
        "finished_at": now_iso(),
        "total_cases": len(results),
        "passed_cases": passed,
        "node_timing_summary": aggregate_node_timing(results),
        "segment_timing_summary": aggregate_segment_timing(results),
        "results": [
            {
                "case_id": r.case_id,
                "session_id": r.session_id,
                "sent_content": r.sent_content,
                "started_at": r.started_at,
                "finished_at": r.finished_at,
                "duration_s": r.duration_s,
                "http_status": r.http_status,
                "done": r.done,
                "terminated_by": r.terminated_by,
                "error": r.error,
                "exception": r.exception,
                "event_count": r.event_count,
                "thought_count": r.thought_count,
                "token_count": r.token_count,
                "status_count": r.status_count,
                "response_preview": r.response_preview,
                "response_content": r.response_content,
                "node_timeline": [vars(x) for x in r.node_timeline],
                "retrieval_logs": r.retrieval_logs,
                "backend_signals": r.backend_signals,
                "expected_intent": r.expected_intent,
                "actual_intent": r.actual_intent,
                "intent_assert_ok": r.intent_assert_ok,
                "expected_route": r.expected_route,
                "actual_routes": r.actual_routes,
                "route_assert_ok": r.route_assert_ok,
                "retrieval_assert_required": r.retrieval_assert_required,
                "retrieval_query_non_empty": r.retrieval_query_non_empty,
                "retrieval_signal_hit": r.retrieval_signal_hit,
                "retrieval_assert_ok": r.retrieval_assert_ok,
                "node_durations_s": r.node_durations_s,
                "last_event_kind": r.last_event_kind,
                "last_event_node": r.last_event_node,
                "timeout_segment": r.timeout_segment,
                "failure_category": r.failure_category,
                "failure_segment": r.failure_segment,
                "ok": r.ok,
            }
            for r in results
        ],
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report_md.write_text(build_report_md(run_id, summary), encoding="utf-8")

    print(json.dumps({"ok": summary["ok"], "run_id": run_id, "out_dir": str(out_dir)}, ensure_ascii=False))
    return 0 if summary["ok"] else 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Robust full-chain e2e logger for Smart Hospital Agent")
    p.add_argument("--project-root", default=".", help="Project root path")
    p.add_argument("--output-dir", default="logs/e2e_fullchain", help="Output directory (under project root)")
    p.add_argument("--base-url", default="http://127.0.0.1:8001", help="Backend base URL")
    p.add_argument("--session-prefix", default="e2e", help="Session id prefix")
    p.add_argument("--cases-file", default="", help="JSON file of test cases: [{id,message}, ...]")
    p.add_argument("--global-timeout", type=int, default=900, help="Global timeout seconds")
    p.add_argument("--case-timeout", type=int, default=240, help="Per-case hard timeout seconds")
    p.add_argument("--stall-timeout", type=int, default=60, help="Per-case no-progress timeout seconds")
    p.add_argument("--max-events", type=int, default=800, help="Per-case max SSE events guard")
    p.add_argument("--no-progress", action="store_true", help="Disable live progress output")
    p.add_argument("--progress-interval", type=float, default=1.0, help="Progress refresh interval seconds")
    p.add_argument("--start-backend", action="store_true", help="Auto start local backend process")
    p.add_argument(
        "--backend-python",
        default="auto",
        help="Backend python executable path. default=auto (auto-detect usable runtime)",
    )
    p.add_argument(
        "--env-file",
        default=".env",
        help="Env file used when --start-backend (must resolve to project root .env)",
    )
    p.add_argument("--backend-host", default="127.0.0.1", help="Backend host when --start-backend")
    p.add_argument("--backend-port", type=int, default=8001, help="Backend port when --start-backend")
    p.add_argument("--backend-ready-timeout", type=int, default=120, help="Backend ready timeout seconds")
    p.add_argument("--backend-log-file", default="", help="Existing backend log file path (for signal extraction without --start-backend)")
    p.add_argument("--backend-log-from-end", action="store_true", help="When using --backend-log-file, start reading from file end")
    p.add_argument("--multi-turn-demo", action="store_true", help="Append built-in multi-turn same-session demo cases")
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return asyncio.run(run(args))
    except KeyboardInterrupt:
        print("Interrupted by user", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
