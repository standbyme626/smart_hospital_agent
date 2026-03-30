#!/usr/bin/env python3
"""
Targeted full-chain SSE tester for /api/v1/chat/stream.

Design goals:
- Robust SSE parsing when backend emits multiple "data:" frames in one chunk.
- Explicit terminal contract checks:
  status:stream_opened -> final -> status:stream_closed -> [DONE]
- Multi-turn same-session execution via "session_group".
- Actionable artifacts: events.jsonl, summary.json, report.md.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx


DEFAULT_CASES: list[dict[str, Any]] = [
    {
        "id": "mt_turn1_greeting",
        "message": "你好，我想先咨询一下",
        "session_group": "multi_turn_mix",
    },
    {
        "id": "mt_turn2_medical",
        "message": "我最近头痛恶心三天",
        "session_group": "multi_turn_mix",
    },
    {
        "id": "mt_turn3_registration",
        "message": "那帮我挂明天下午心内科号",
        "session_group": "multi_turn_mix",
    },
]


@dataclass
class CaseSpec:
    case_id: str
    message: str
    session_group: str
    expected_intent: str = ""
    expected_route: str = ""
    assert_retrieval: bool = False


@dataclass
class CaseResult:
    case_id: str
    session_id: str
    request_id: str
    message: str
    started_at: str
    finished_at: str = ""
    duration_s: float = 0.0
    http_status: Optional[int] = None
    terminated_by: str = "normal"
    done: bool = False
    event_count: int = 0
    token_count: int = 0
    status_count: int = 0
    unparsed_count: int = 0
    final_seen: bool = False
    stream_opened_seen: bool = False
    stream_closed_seen: bool = False
    response_content: str = ""
    error_events: List[str] = field(default_factory=list)
    sequence: List[str] = field(default_factory=list)
    contract_issues: List[str] = field(default_factory=list)
    effective_runtime_config: Dict[str, Any] = field(default_factory=dict)
    exception: str = ""
    ok: bool = False


def now_iso() -> str:
    return datetime.now().isoformat(timespec="milliseconds")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        k, v = raw.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def resolve_python_bin(project_root: Path, requested: str) -> str:
    if requested and requested != "auto":
        return requested
    candidates = [
        project_root.parent / ".venv" / "bin" / "python",
        project_root / ".venv" / "bin" / "python",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    py3 = shutil.which("python3")
    if py3:
        return py3
    return sys.executable


def start_backend_process(
    project_root: Path,
    env_file: Path,
    host: str,
    port: int,
    python_bin: str,
    log_path: Path,
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
    fout = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=str(project_root),
        env=env,
        stdout=fout,
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


async def wait_backend_ready(base_url: str, timeout_s: int) -> bool:
    deadline = time.monotonic() + timeout_s
    async with httpx.AsyncClient(timeout=httpx.Timeout(5.0, connect=2.0)) as client:
        while time.monotonic() < deadline:
            try:
                r = await client.get(f"{base_url.rstrip('/')}/health")
                if r.status_code == 200:
                    return True
            except Exception:
                pass
            await asyncio.sleep(1.0)
    return False


def load_cases(path: Optional[Path]) -> List[CaseSpec]:
    if path is None:
        raw_cases = DEFAULT_CASES
    else:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError(f"cases file must be a JSON list: {path}")
        raw_cases = raw
    cases: List[CaseSpec] = []
    for i, item in enumerate(raw_cases):
        if not isinstance(item, dict):
            raise ValueError(f"case[{i}] must be object")
        case_id = str(item.get("id", f"case_{i+1}")).strip()
        message = str(item.get("message", "")).strip()
        session_group = str(item.get("session_group", case_id)).strip() or case_id
        if not message:
            raise ValueError(f"case[{i}] message is empty")
        cases.append(
            CaseSpec(
                case_id=case_id,
                message=message,
                session_group=session_group,
                expected_intent=str(item.get("expected_intent", "") or ""),
                expected_route=str(item.get("expected_route", "") or ""),
                assert_retrieval=bool(item.get("assert_retrieval", False)),
            )
        )
    return cases


def build_session_map(cases: List[CaseSpec], prefix: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    seed = int(time.time())
    for spec in cases:
        if spec.session_group not in mapping:
            nonce = random.randint(1000, 9999)
            mapping[spec.session_group] = f"{prefix}_{spec.session_group}_{seed}_{nonce}"
    return mapping


def _split_sse_frames(buffer: str) -> tuple[list[str], str]:
    frames: list[str] = []
    while True:
        idx = buffer.find("\n\n")
        if idx < 0:
            break
        frame = buffer[:idx]
        buffer = buffer[idx + 2 :]
        if frame.strip():
            frames.append(frame)
    return frames, buffer


def _extract_data_payload(frame: str) -> Optional[str]:
    data_lines: list[str] = []
    for line in frame.split("\n"):
        if not line:
            continue
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    if not data_lines:
        return None
    return "\n".join(data_lines).strip()


def _marker_for_event(payload: dict[str, Any]) -> str:
    event_type = str(payload.get("type", "unknown") or "unknown")
    content = str(payload.get("content", "") or "")
    if event_type == "status":
        return f"status:{content}"
    return event_type


def _first_index(seq: list[str], marker: str) -> int:
    try:
        return seq.index(marker)
    except ValueError:
        return -1


def _evaluate_contract(result: CaseResult) -> list[str]:
    issues: list[str] = []
    if result.http_status != 200:
        issues.append(f"http_status_{result.http_status}")
    if not result.stream_opened_seen:
        issues.append("missing_stream_opened")
    if not result.final_seen:
        issues.append("missing_final")
    if not result.stream_closed_seen:
        issues.append("missing_stream_closed")
    if not result.done:
        issues.append("missing_done")
    if not result.response_content.strip():
        issues.append("empty_response")
    if result.error_events:
        issues.append("error_event_emitted")

    final_idx = _first_index(result.sequence, "final")
    closed_idx = _first_index(result.sequence, "status:stream_closed")
    done_idx = _first_index(result.sequence, "[DONE]")
    if final_idx >= 0 and closed_idx >= 0 and done_idx >= 0:
        if not (final_idx < closed_idx < done_idx):
            issues.append("terminal_order_invalid")

    if result.terminated_by in {
        "stall_timeout",
        "case_timeout",
        "max_events_guard",
        "http_error",
        "exception",
    }:
        issues.append(f"terminated_by_{result.terminated_by}")
    return issues


async def run_case(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    spec: CaseSpec,
    session_id: str,
    events_jsonl: Path,
    case_timeout_s: int,
    stall_timeout_s: int,
    max_events: int,
) -> CaseResult:
    request_id = f"e2e-target-{spec.case_id}-{int(time.time() * 1000)}-{uuid.uuid4().hex[:6]}"
    result = CaseResult(
        case_id=spec.case_id,
        session_id=session_id,
        request_id=request_id,
        message=spec.message,
        started_at=now_iso(),
    )
    payload = {
        "message": spec.message,
        "session_id": session_id,
        "request_id": request_id,
    }
    write_jsonl(
        events_jsonl,
        {
            "ts_iso": now_iso(),
            "case_id": spec.case_id,
            "session_id": session_id,
            "request_id": request_id,
            "kind": "request_sent",
            "payload": payload,
        },
    )

    start_mono = time.monotonic()
    token_parts: list[str] = []

    try:
        async with client.stream("POST", f"{base_url}/api/v1/chat/stream", json=payload) as resp:
            result.http_status = resp.status_code
            if resp.status_code != 200:
                result.terminated_by = "http_error"
                result.finished_at = now_iso()
                result.duration_s = round(time.monotonic() - start_mono, 3)
                result.contract_issues = _evaluate_contract(result)
                result.ok = len(result.contract_issues) == 0
                return result

            chunk_iter = resp.aiter_text()
            buffer = ""
            while True:
                elapsed = time.monotonic() - start_mono
                if elapsed > case_timeout_s:
                    result.terminated_by = "case_timeout"
                    break
                if result.event_count >= max_events:
                    result.terminated_by = "max_events_guard"
                    break

                try:
                    chunk = await asyncio.wait_for(chunk_iter.__anext__(), timeout=stall_timeout_s)
                except asyncio.TimeoutError:
                    result.terminated_by = "stall_timeout"
                    break
                except StopAsyncIteration:
                    result.terminated_by = "stream_eof"
                    break

                if not chunk:
                    continue

                buffer += chunk.replace("\r\n", "\n").replace("\r", "\n")
                frames, buffer = _split_sse_frames(buffer)
                for frame in frames:
                    rel_ms = int((time.monotonic() - start_mono) * 1000)
                    payload_raw = _extract_data_payload(frame)
                    if payload_raw is None:
                        write_jsonl(
                            events_jsonl,
                            {
                                "ts_iso": now_iso(),
                                "case_id": spec.case_id,
                                "session_id": session_id,
                                "request_id": request_id,
                                "kind": "non_data_frame",
                                "rel_ms": rel_ms,
                                "raw": frame[:600],
                            },
                        )
                        continue

                    if payload_raw == "[DONE]":
                        result.done = True
                        result.sequence.append("[DONE]")
                        write_jsonl(
                            events_jsonl,
                            {
                                "ts_iso": now_iso(),
                                "case_id": spec.case_id,
                                "session_id": session_id,
                                "request_id": request_id,
                                "kind": "done",
                                "rel_ms": rel_ms,
                            },
                        )
                        break

                    try:
                        event_obj = json.loads(payload_raw)
                    except Exception:
                        result.unparsed_count += 1
                        result.sequence.append("unparsed")
                        write_jsonl(
                            events_jsonl,
                            {
                                "ts_iso": now_iso(),
                                "case_id": spec.case_id,
                                "session_id": session_id,
                                "request_id": request_id,
                                "kind": "unparsed",
                                "rel_ms": rel_ms,
                                "raw": payload_raw[:600],
                            },
                        )
                        continue

                    result.event_count += 1
                    marker = _marker_for_event(event_obj)
                    result.sequence.append(marker)

                    event_type = str(event_obj.get("type", "unknown") or "unknown")
                    content = str(event_obj.get("content", "") or "")
                    node = str(event_obj.get("node", "") or "")
                    seq = event_obj.get("seq")
                    stage = str(event_obj.get("stage", "") or "")

                    if event_type == "token":
                        result.token_count += 1
                        if content:
                            token_parts.append(content)
                    elif event_type == "status":
                        result.status_count += 1
                        if content == "stream_opened":
                            result.stream_opened_seen = True
                        elif content == "stream_closed":
                            result.stream_closed_seen = True
                    elif event_type == "final":
                        result.final_seen = True
                        if content and not token_parts:
                            token_parts.append(content)
                    elif event_type == "error":
                        if content:
                            result.error_events.append(content[:1000])

                    runtime_cfg = event_obj.get("runtime_config_effective")
                    if not isinstance(runtime_cfg, dict):
                        meta_obj = event_obj.get("meta")
                        if isinstance(meta_obj, dict):
                            runtime_cfg = meta_obj.get("runtime_config_effective")
                    if isinstance(runtime_cfg, dict):
                        result.effective_runtime_config = dict(runtime_cfg)

                    write_jsonl(
                        events_jsonl,
                        {
                            "ts_iso": now_iso(),
                            "case_id": spec.case_id,
                            "session_id": session_id,
                            "request_id": request_id,
                            "kind": event_type,
                            "rel_ms": rel_ms,
                            "content": content,
                            "node": node,
                            "seq": seq,
                            "stage": stage,
                        },
                    )

                if result.done:
                    result.terminated_by = "done"
                    break

            if not result.done and buffer.strip():
                trailing = _extract_data_payload(buffer)
                if trailing:
                    try:
                        obj = json.loads(trailing)
                        write_jsonl(
                            events_jsonl,
                            {
                                "ts_iso": now_iso(),
                                "case_id": spec.case_id,
                                "session_id": session_id,
                                "request_id": request_id,
                                "kind": "trailing_event",
                                "rel_ms": int((time.monotonic() - start_mono) * 1000),
                                "event_type": str(obj.get("type", "unknown")),
                            },
                        )
                    except Exception:
                        write_jsonl(
                            events_jsonl,
                            {
                                "ts_iso": now_iso(),
                                "case_id": spec.case_id,
                                "session_id": session_id,
                                "request_id": request_id,
                                "kind": "trailing_unparsed",
                                "rel_ms": int((time.monotonic() - start_mono) * 1000),
                                "raw": trailing[:600],
                            },
                        )
    except Exception as exc:
        result.terminated_by = "exception"
        result.exception = str(exc)[:1000]

    result.finished_at = now_iso()
    result.duration_s = round(time.monotonic() - start_mono, 3)
    result.response_content = "".join(token_parts).strip()
    result.contract_issues = _evaluate_contract(result)
    result.ok = len(result.contract_issues) == 0
    return result


def render_report(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Targeted Full-chain E2E Report")
    lines.append("")
    lines.append(f"- run_id: `{summary['run_id']}`")
    lines.append(f"- ok: `{summary['ok']}`")
    lines.append(f"- started_at: `{summary['started_at']}`")
    lines.append(f"- finished_at: `{summary['finished_at']}`")
    lines.append(f"- total_cases: `{summary['total_cases']}`")
    lines.append(f"- passed_cases: `{summary['passed_cases']}`")
    lines.append("")
    for case in summary["results"]:
        lines.append(f"## {case['case_id']}")
        lines.append(f"- ok: `{case['ok']}`")
        lines.append(f"- terminated_by: `{case['terminated_by']}`")
        lines.append(f"- http_status: `{case['http_status']}`")
        lines.append(f"- duration_s: `{case['duration_s']}`")
        lines.append(f"- stream flags: `opened={case['stream_opened_seen']}, final={case['final_seen']}, closed={case['stream_closed_seen']}, done={case['done']}`")
        lines.append(f"- counts: `events={case['event_count']}, token={case['token_count']}, status={case['status_count']}, unparsed={case['unparsed_count']}`")
        if case.get("contract_issues"):
            lines.append(f"- contract_issues: `{','.join(case['contract_issues'])}`")
        if case.get("effective_runtime_config"):
            lines.append(
                f"- effective_runtime_config: `{json.dumps(case['effective_runtime_config'], ensure_ascii=False)}`"
            )
        if case.get("error_events"):
            lines.append(f"- error_events: `{json.dumps(case['error_events'], ensure_ascii=False)}`")
        if case.get("exception"):
            lines.append(f"- exception: `{case['exception']}`")
        lines.append("- response_preview:")
        lines.append(f"  - `{(case.get('response_content') or '')[:240]}`")
        lines.append("")
    return "\n".join(lines)


async def run(args: argparse.Namespace) -> int:
    project_root = Path(args.project_root).resolve() if args.project_root else Path(__file__).resolve().parents[1]
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.output_dir:
        out_root = Path(args.output_dir)
        if not out_root.is_absolute():
            out_root = project_root / out_root
    else:
        out_root = project_root / "logs" / "e2e_fullchain_targeted"
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    events_jsonl = run_dir / "events.jsonl"
    summary_json = run_dir / "summary.json"
    report_md = run_dir / "report.md"
    backend_log = run_dir / "backend_runtime.log"

    cases_path: Optional[Path] = None
    if args.cases_file:
        cases_path = Path(args.cases_file)
        if not cases_path.is_absolute():
            cases_path = project_root / cases_path
        if not cases_path.exists():
            raise FileNotFoundError(f"cases file not found: {cases_path}")
    cases = load_cases(cases_path)
    session_map = build_session_map(cases, args.session_prefix)

    if args.base_url:
        base_url = args.base_url.rstrip("/")
    else:
        base_url = f"http://{args.backend_host}:{args.backend_port}"

    backend_proc: Optional[subprocess.Popen] = None
    started_at = now_iso()
    if args.start_backend:
        python_bin = resolve_python_bin(project_root, args.backend_python)
        env_file = Path(args.env_file)
        if not env_file.is_absolute():
            env_file = project_root / env_file
        backend_proc = start_backend_process(
            project_root=project_root,
            env_file=env_file,
            host=args.backend_host,
            port=args.backend_port,
            python_bin=python_bin,
            log_path=backend_log,
        )
        ready = await wait_backend_ready(base_url, args.backend_ready_timeout)
        if not ready:
            if backend_proc is not None:
                stop_backend_process(backend_proc)
            summary = {
                "run_id": run_id,
                "ok": False,
                "started_at": started_at,
                "finished_at": now_iso(),
                "error": f"backend_not_ready_within_{args.backend_ready_timeout}s",
                "total_cases": len(cases),
                "passed_cases": 0,
                "results": [],
            }
            write_json(summary_json, summary)
            report_md.write_text(render_report(summary), encoding="utf-8")
            print(json.dumps({"ok": False, "run_id": run_id, "out_dir": str(run_dir)}, ensure_ascii=False))
            return 2

    meta = {
        "run_id": run_id,
        "project_root": str(project_root),
        "base_url": base_url,
        "cases_file": str(cases_path) if cases_path else "",
        "start_backend": bool(args.start_backend),
        "backend_host": args.backend_host,
        "backend_port": args.backend_port,
        "case_timeout_s": args.case_timeout,
        "stall_timeout_s": args.stall_timeout,
        "max_events": args.max_events,
        "started_at": started_at,
    }
    write_json(run_dir / "meta.json", meta)

    results: List[CaseResult] = []
    timeout = httpx.Timeout(connect=10.0, read=None, write=30.0, pool=30.0)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            for spec in cases:
                session_id = session_map[spec.session_group]
                case_result = await run_case(
                    client,
                    base_url=base_url,
                    spec=spec,
                    session_id=session_id,
                    events_jsonl=events_jsonl,
                    case_timeout_s=args.case_timeout,
                    stall_timeout_s=args.stall_timeout,
                    max_events=args.max_events,
                )
                results.append(case_result)
    finally:
        if backend_proc is not None:
            stop_backend_process(backend_proc)

    result_rows = [asdict(r) for r in results]
    passed = sum(1 for r in results if r.ok)
    summary = {
        "run_id": run_id,
        "ok": passed == len(results) and len(results) > 0,
        "started_at": started_at,
        "finished_at": now_iso(),
        "total_cases": len(results),
        "passed_cases": passed,
        "results": result_rows,
    }
    write_json(summary_json, summary)
    report_md.write_text(render_report(summary), encoding="utf-8")
    print(json.dumps({"ok": summary["ok"], "run_id": run_id, "out_dir": str(run_dir)}, ensure_ascii=False))
    return 0 if summary["ok"] else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Targeted full-chain SSE tester robust to multi-frame chunks."
    )
    parser.add_argument("--project-root", default="", help="Project root path")
    parser.add_argument("--output-dir", default="logs/e2e_fullchain_targeted", help="Output root directory")
    parser.add_argument("--cases-file", default="scripts/e2e_cases_multiturn.json", help="JSON case list file")
    parser.add_argument("--base-url", default="", help="Backend base URL, e.g. http://127.0.0.1:8001")
    parser.add_argument("--session-prefix", default="e2e_targeted", help="Session id prefix")
    parser.add_argument("--case-timeout", type=int, default=120, help="Per-case hard timeout seconds")
    parser.add_argument("--stall-timeout", type=int, default=30, help="Per-case no-progress timeout seconds")
    parser.add_argument("--max-events", type=int, default=1200, help="Per-case max event guard")

    parser.add_argument("--start-backend", action="store_true", help="Auto start local backend uvicorn")
    parser.add_argument("--backend-python", default="auto", help="Backend python executable")
    parser.add_argument("--env-file", default=".env", help="Env file path when --start-backend")
    parser.add_argument("--backend-host", default="127.0.0.1", help="Backend host")
    parser.add_argument("--backend-port", type=int, default=8001, help="Backend port")
    parser.add_argument(
        "--backend-ready-timeout",
        type=int,
        default=90,
        help="Timeout seconds waiting backend /health",
    )
    return parser.parse_args()


def main() -> None:
    code = asyncio.run(run(parse_args()))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
