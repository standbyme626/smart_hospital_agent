#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "test_sample_100_v4.jsonl"

BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.department_normalization import (
    extract_department_mentions,
    normalize_department_candidates,
    normalize_department_name,
)


@dataclass
class ChainEvalRecord:
    chain: str
    idx: int
    session_id: str
    request_id: str
    question: str
    expected_raw: str
    expected_canonical: str
    latency_s: float
    status: str
    done: bool
    stream_error: str
    timeout: bool
    response_excerpt: str
    predicted_raw_top3: List[str]
    predicted_canonical_top3: List[str]
    predicted_confidence: float
    used_structured: bool
    effective_runtime_config: Dict[str, Any]


def load_jsonl(path: Path, max_samples: int = 0) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_samples > 0 and len(rows) >= max_samples:
                break
    return rows


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    vals = sorted(values)
    k = (len(vals) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(vals) - 1)
    if lo == hi:
        return vals[lo]
    return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)


def baseline_match(expected_raw: str, predicted_raw: str) -> bool:
    e = str(expected_raw or "").strip()
    p = str(predicted_raw or "").strip()
    if not e or not p:
        return False
    return e in p or p in e


def lenient_match(expected_canonical: str, predicted_canonical: str) -> bool:
    e = str(expected_canonical or "").strip()
    p = str(predicted_canonical or "").strip()
    if not e or not p:
        return False
    if e == p:
        return True
    e2 = e.replace("内科", "").replace("外科", "").replace("科", "")
    p2 = p.replace("内科", "").replace("外科", "").replace("科", "")
    return bool(e2 and p2 and (e2 in p2 or p2 in e2))


def classify_failure_bucket(
    status: str,
    stream_error: str,
    timeout: bool,
    pred_top1_hit: bool,
    pred_top3_hit: bool,
    has_prediction: bool,
) -> str:
    if timeout:
        return "timeout"
    if status != "ok":
        return "request_error"
    if stream_error:
        return "stream_error"
    if not has_prediction:
        return "no_department_extracted"
    if not pred_top1_hit and pred_top3_hit:
        return "top1_miss_top3_hit"
    if not pred_top3_hit:
        return "top3_miss"
    return "ok"


def _coerce_confidence(value: Any) -> float:
    try:
        conf = float(value)
    except Exception:
        return 0.0
    if conf < 0:
        return 0.0
    if conf > 1:
        if conf <= 100:
            return conf / 100.0
        return 1.0
    return conf


def _extract_structured_department_result(event_obj: Dict[str, Any]) -> Tuple[List[str], float]:
    candidates: List[str] = []
    confidence = 0.0

    def _from_container(container: Any) -> None:
        nonlocal confidence
        if not isinstance(container, dict):
            return
        top1 = container.get("department_top1")
        top3 = container.get("department_top3")
        if isinstance(top1, str) and top1.strip():
            candidates.append(top1.strip())
        if isinstance(top3, list):
            for item in top3:
                if isinstance(item, str) and item.strip():
                    candidates.append(item.strip())
        if confidence <= 0:
            confidence = _coerce_confidence(container.get("confidence"))

    meta = event_obj.get("meta") if isinstance(event_obj.get("meta"), dict) else {}
    data = event_obj.get("data") if isinstance(event_obj.get("data"), dict) else {}

    _from_container(event_obj)
    _from_container(meta)
    _from_container(data)
    _from_container(meta.get("department_result"))
    _from_container(meta.get("data"))

    canonical_top3 = normalize_department_candidates(candidates, top_k=3)
    return canonical_top3, confidence


async def request_sse(
    client: httpx.AsyncClient,
    url: str,
    payload: Dict[str, Any],
    timeout_s: float,
    stop_on_first_dept: bool = True,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    status = "ok"
    done = False
    timeout = False
    stream_error = ""
    parsed_events = 0
    early_stopped = False
    text_parts: List[str] = []
    raw_lines: List[str] = []
    structured_top3: List[str] = []
    structured_confidence = 0.0
    seen_request_id = str(payload.get("request_id") or "").strip()
    effective_runtime_config: Dict[str, Any] = {}

    try:
        async with client.stream("POST", url, json=payload) as resp:
            if resp.status_code >= 400:
                status = f"http_{resp.status_code}"
            async for line in resp.aiter_lines():
                if time.perf_counter() - t0 > timeout_s:
                    timeout = True
                    status = "timeout"
                    break
                if not line:
                    continue
                if line.startswith("event:"):
                    raw_lines.append(line)
                    continue
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                raw_lines.append(line[:400])
                if data == "[DONE]":
                    done = True
                    break
                try:
                    obj = json.loads(data)
                except Exception:
                    continue
                parsed_events += 1
                event_type = str(obj.get("type", "") or "")
                evt_request_id = str(obj.get("request_id", "") or "").strip()
                if evt_request_id:
                    seen_request_id = evt_request_id
                runtime_cfg = obj.get("runtime_config_effective")
                if not isinstance(runtime_cfg, dict):
                    meta_obj = obj.get("meta")
                    if isinstance(meta_obj, dict):
                        runtime_cfg = meta_obj.get("runtime_config_effective")
                if isinstance(runtime_cfg, dict):
                    effective_runtime_config = dict(runtime_cfg)
                if event_type == "department_result":
                    canonical_top3, conf = _extract_structured_department_result(obj)
                    if canonical_top3:
                        structured_top3 = canonical_top3
                        if conf > 0:
                            structured_confidence = conf
                        if stop_on_first_dept:
                            early_stopped = True
                            done = True
                            break
                if event_type in {"token", "final"}:
                    content = str(obj.get("content", "") or "").strip()
                    if content:
                        text_parts.append(content)
                        if stop_on_first_dept and not structured_top3:
                            _, canon = extract_department_mentions("\n".join(text_parts), top_k=3)
                            if canon:
                                early_stopped = True
                                done = True
                                break
                if event_type == "error":
                    stream_error = str(obj.get("content", "") or "")[:800]
    except httpx.TimeoutException as exc:
        status = "timeout"
        timeout = True
        stream_error = str(exc)[:300]
    except Exception as exc:
        status = "exception"
        stream_error = str(exc)[:300]

    latency_s = time.perf_counter() - t0
    text = "\n".join(text_parts).strip()
    return {
        "status": status,
        "done": done,
        "timeout": timeout,
        "stream_error": stream_error,
        "latency_s": latency_s,
        "parsed_events": parsed_events,
        "response_text": text,
        "raw_lines": raw_lines,
        "early_stopped": early_stopped,
        "structured_top3": structured_top3,
        "structured_confidence": round(structured_confidence, 4),
        "request_id": seen_request_id,
        "effective_runtime_config": effective_runtime_config,
    }


async def eval_one_chain(
    client: httpx.AsyncClient,
    chain: str,
    base_url: str,
    idx: int,
    question: str,
    expected_raw: str,
    expected_canonical: str,
    timeout_s: float,
    prompt_suffix: str,
) -> ChainEvalRecord:
    sid = f"real_eval_{chain}_{idx:04d}"
    request_id = f"weekly-{chain}-{idx:04d}-{int(time.time() * 1000)}"
    eval_message = question if not prompt_suffix else f"{question}\n{prompt_suffix}"
    if chain == "triage":
        url = f"{base_url}/api/v1/chat/stream"
        payload = {
            "message": eval_message,
            "session_id": sid,
            "request_id": request_id,
            "rag": {"top_k": 3, "use_rerank": True, "rerank_threshold": 0.15},
        }
    elif chain == "doctor":
        url = f"{base_url}/api/v1/doctor/workflow?schema_mode=unified"
        payload = {"message": eval_message, "session_id": sid, "request_id": request_id}
    else:
        raise ValueError(f"unsupported chain: {chain}")

    try:
        # Hard timeout to guard against silent stream stalls with no incoming lines.
        resp = await asyncio.wait_for(
            request_sse(client=client, url=url, payload=payload, timeout_s=timeout_s),
            timeout=timeout_s + 8.0,
        )
    except asyncio.TimeoutError:
        resp = {
            "status": "timeout",
            "done": False,
            "timeout": True,
            "stream_error": "hard_timeout",
            "latency_s": timeout_s + 8.0,
            "response_text": "",
        }
    raw_top3_text, canonical_top3_text = extract_department_mentions(resp["response_text"], top_k=3)
    structured_top3 = list(resp.get("structured_top3") or [])
    used_structured = bool(structured_top3)

    canonical_top3 = structured_top3 if structured_top3 else canonical_top3_text
    raw_top3 = raw_top3_text if raw_top3_text else list(canonical_top3)
    predicted_confidence = float(resp.get("structured_confidence") or 0.0)

    return ChainEvalRecord(
        chain=chain,
        idx=idx,
        session_id=sid,
        request_id=str(resp.get("request_id") or request_id),
        question=question,
        expected_raw=expected_raw,
        expected_canonical=expected_canonical,
        latency_s=round(float(resp["latency_s"]), 6),
        status=str(resp["status"]),
        done=bool(resp["done"]),
        stream_error=str(resp["stream_error"] or ""),
        timeout=bool(resp["timeout"]),
        response_excerpt=str(resp["response_text"][:500]),
        predicted_raw_top3=raw_top3,
        predicted_canonical_top3=canonical_top3,
        predicted_confidence=round(predicted_confidence, 4),
        used_structured=used_structured,
        effective_runtime_config=dict(resp.get("effective_runtime_config") or {}),
    )


def summarize_chain(records: List[ChainEvalRecord]) -> Dict[str, Any]:
    if not records:
        return {}

    baseline_hits_top1 = 0
    baseline_hits_top3 = 0
    current_hits_top1_strict = 0
    current_hits_top3_strict = 0
    current_hits_top1_lenient = 0
    current_hits_top3_lenient = 0

    baseline_buckets = Counter()
    current_buckets = Counter()
    latencies = []
    structured_used = 0

    for r in records:
        latencies.append(r.latency_s)
        structured_used += int(r.used_structured)
        has_pred = bool(r.predicted_canonical_top3)
        base_top1 = baseline_match(r.expected_raw, r.predicted_raw_top3[0] if r.predicted_raw_top3 else "")
        base_top3 = any(baseline_match(r.expected_raw, p) for p in r.predicted_raw_top3)
        cur_top1_strict = (
            bool(r.predicted_canonical_top3) and (r.expected_canonical == r.predicted_canonical_top3[0])
        )
        cur_top3_strict = any(r.expected_canonical == p for p in r.predicted_canonical_top3)
        cur_top1_lenient = (
            bool(r.predicted_canonical_top3)
            and lenient_match(r.expected_canonical, r.predicted_canonical_top3[0])
        )
        cur_top3_lenient = any(lenient_match(r.expected_canonical, p) for p in r.predicted_canonical_top3)

        baseline_hits_top1 += int(base_top1)
        baseline_hits_top3 += int(base_top3)
        current_hits_top1_strict += int(cur_top1_strict)
        current_hits_top3_strict += int(cur_top3_strict)
        current_hits_top1_lenient += int(cur_top1_lenient)
        current_hits_top3_lenient += int(cur_top3_lenient)

        baseline_buckets[
            classify_failure_bucket(
                status=r.status,
                stream_error=r.stream_error,
                timeout=r.timeout,
                pred_top1_hit=base_top1,
                pred_top3_hit=base_top3,
                has_prediction=has_pred,
            )
        ] += 1
        current_buckets[
            classify_failure_bucket(
                status=r.status,
                stream_error=r.stream_error,
                timeout=r.timeout,
                pred_top1_hit=cur_top1_lenient,
                pred_top3_hit=cur_top3_lenient,
                has_prediction=has_pred,
            )
        ] += 1

    total = len(records)
    baseline = {
        "sample_n": total,
        "top1": baseline_hits_top1 / total if total else 0.0,
        "top3": baseline_hits_top3 / total if total else 0.0,
        "error_rate": 1.0 - (baseline_hits_top1 / total if total else 0.0),
        "latency_s": {
            "avg": statistics.mean(latencies) if latencies else 0.0,
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "max": max(latencies) if latencies else 0.0,
        },
        "failure_buckets": dict(baseline_buckets),
    }
    current = {
        "sample_n": total,
        "structured_output_hits": structured_used,
        "structured_output_ratio": structured_used / total if total else 0.0,
        "top1_strict": current_hits_top1_strict / total if total else 0.0,
        "top3_strict": current_hits_top3_strict / total if total else 0.0,
        "top1": current_hits_top1_lenient / total if total else 0.0,
        "top3": current_hits_top3_lenient / total if total else 0.0,
        "error_rate": 1.0 - (current_hits_top1_lenient / total if total else 0.0),
        "latency_s": {
            "avg": statistics.mean(latencies) if latencies else 0.0,
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "max": max(latencies) if latencies else 0.0,
        },
        "failure_buckets": dict(current_buckets),
    }
    delta = {
        "top1_delta": current["top1"] - baseline["top1"],
        "top3_delta": current["top3"] - baseline["top3"],
        "error_rate_delta": current["error_rate"] - baseline["error_rate"],
        "p50_delta_s": current["latency_s"]["p50"] - baseline["latency_s"]["p50"],
        "p95_delta_s": current["latency_s"]["p95"] - baseline["latency_s"]["p95"],
    }
    return {"baseline": baseline, "current": current, "delta": delta}


def _langfuse_host() -> str:
    return str(os.getenv("LANGFUSE_HOST", "http://127.0.0.1:3000") or "http://127.0.0.1:3000").rstrip("/")


def _write_trace_mapping_artifacts(
    *,
    outdir: Path,
    records: List[ChainEvalRecord],
    summary_json: Path,
    report_md: Path,
    records_jsonl: Path,
) -> Tuple[Path, Path]:
    host = _langfuse_host()
    mapping_jsonl = outdir / "trace_request_map.jsonl"
    mapping_md = outdir / "trace_request_map.md"

    if mapping_jsonl.exists():
        mapping_jsonl.unlink()

    lines = [
        "# Trace ↔ Request Mapping (Weekly Baseline)",
        "",
        "| request_id | chain | idx | langfuse | summary | report | records |",
        "|---|---|---:|---|---|---|---|",
    ]

    for item in records:
        entry = {
            "source": "weekly_baseline",
            "request_id": item.request_id,
            "trace_id": item.request_id,
            "langfuse_host": host,
            "langfuse_lookup_hint": f"search trace id `{item.request_id}` in Langfuse",
            "chain": item.chain,
            "idx": item.idx,
            "session_id": item.session_id,
            "effective_runtime_config": item.effective_runtime_config,
            "artifacts": {
                "summary_json": str(summary_json),
                "report_md": str(report_md),
                "records_jsonl": str(records_jsonl),
                "out_dir": str(outdir),
            },
        }
        with mapping_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        lines.append(
            f"| `{item.request_id}` | `{item.chain}` | `{item.idx}` | `{host}` (search `{item.request_id}`) | `{summary_json}` | `{report_md}` | `{records_jsonl}` |"
        )

    mapping_md.write_text("\n".join(lines), encoding="utf-8")
    global_dir = PROJECT_ROOT / "logs" / "trace_replay"
    global_dir.mkdir(parents=True, exist_ok=True)
    global_jsonl = global_dir / "replay_index.jsonl"
    with mapping_jsonl.open("r", encoding="utf-8") as src, global_jsonl.open("a", encoding="utf-8") as dst:
        for line in src:
            dst.write(line)
    return mapping_jsonl, mapping_md


async def run_eval(args: argparse.Namespace) -> Dict[str, Any]:
    rows = load_jsonl(Path(args.dataset), max_samples=args.max_samples)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    records: List[ChainEvalRecord] = []
    client_timeout = httpx.Timeout(connect=10.0, read=args.timeout_s + 5.0, write=10.0, pool=10.0)
    async with httpx.AsyncClient(timeout=client_timeout) as client:
        for idx, row in enumerate(rows, start=1):
            question = str(row.get("question", "")).strip()
            expected_raw = str(row.get("department", "")).strip()
            expected_canonical, _ = normalize_department_name(expected_raw)
            if not question:
                continue
            for chain in ("triage", "doctor"):
                rec = await eval_one_chain(
                    client=client,
                    chain=chain,
                    base_url=args.base_url,
                    idx=idx,
                    question=question,
                    expected_raw=expected_raw,
                    expected_canonical=expected_canonical,
                    timeout_s=args.timeout_s,
                    prompt_suffix=args.prompt_suffix,
                )
                records.append(rec)

    triage_records = [r for r in records if r.chain == "triage"]
    doctor_records = [r for r in records if r.chain == "doctor"]

    summary = {
        "meta": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "dataset": str(args.dataset),
            "sample_n": len(rows),
            "base_url": args.base_url,
            "timeout_s": args.timeout_s,
            "max_samples": args.max_samples,
            "prompt_suffix": args.prompt_suffix,
        },
        "triage": summarize_chain(triage_records),
        "doctor": summarize_chain(doctor_records),
    }

    records_jsonl = outdir / "records.jsonl"
    summary_json = outdir / "summary.json"
    compare_report_md = outdir / "compare_report.md"

    with records_jsonl.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(
                json.dumps(
                    {
                        "chain": r.chain,
                        "idx": r.idx,
                        "session_id": r.session_id,
                        "request_id": r.request_id,
                        "question": r.question,
                        "expected_raw": r.expected_raw,
                        "expected_canonical": r.expected_canonical,
                        "latency_s": r.latency_s,
                        "status": r.status,
                        "done": r.done,
                        "timeout": r.timeout,
                        "stream_error": r.stream_error,
                        "response_excerpt": r.response_excerpt,
                        "predicted_raw_top3": r.predicted_raw_top3,
                        "predicted_canonical_top3": r.predicted_canonical_top3,
                        "predicted_confidence": r.predicted_confidence,
                        "used_structured": r.used_structured,
                        "effective_runtime_config": r.effective_runtime_config,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    report_lines = [
        "# Real Triage/Doctor Chain Eval",
        "",
        f"- dataset: `{args.dataset}`",
        f"- sample_n: `{len(rows)}`",
        f"- base_url: `{args.base_url}`",
        f"- timeout_s: `{args.timeout_s}`",
        f"- records_jsonl: `{records_jsonl}`",
        "",
    ]
    for chain in ("triage", "doctor"):
        block = summary.get(chain, {})
        baseline = block.get("baseline", {})
        current = block.get("current", {})
        delta = block.get("delta", {})
        report_lines.extend(
            [
                f"## {chain}",
                "",
                "| metric | baseline | current | delta |",
                "|---|---:|---:|---:|",
                f"| error_rate | {baseline.get('error_rate', 0.0):.4f} | {current.get('error_rate', 0.0):.4f} | {delta.get('error_rate_delta', 0.0):+.4f} |",
                f"| Top1 | {baseline.get('top1', 0.0):.4f} | {current.get('top1', 0.0):.4f} | {delta.get('top1_delta', 0.0):+.4f} |",
                f"| Top3 | {baseline.get('top3', 0.0):.4f} | {current.get('top3', 0.0):.4f} | {delta.get('top3_delta', 0.0):+.4f} |",
                f"| StructuredRatio | 0.0000 | {current.get('structured_output_ratio', 0.0):.4f} | {current.get('structured_output_ratio', 0.0):+.4f} |",
                f"| P50(s) | {baseline.get('latency_s', {}).get('p50', 0.0):.4f} | {current.get('latency_s', {}).get('p50', 0.0):.4f} | {delta.get('p50_delta_s', 0.0):+.4f} |",
                f"| P95(s) | {baseline.get('latency_s', {}).get('p95', 0.0):.4f} | {current.get('latency_s', {}).get('p95', 0.0):.4f} | {delta.get('p95_delta_s', 0.0):+.4f} |",
                "",
                f"- baseline_failure_buckets: `{json.dumps(baseline.get('failure_buckets', {}), ensure_ascii=False)}`",
                f"- current_failure_buckets: `{json.dumps(current.get('failure_buckets', {}), ensure_ascii=False)}`",
                "",
            ]
        )
    trace_map_jsonl, trace_map_md = _write_trace_mapping_artifacts(
        outdir=outdir,
        records=records,
        summary_json=summary_json,
        report_md=compare_report_md,
        records_jsonl=records_jsonl,
    )
    summary["meta"]["trace_request_map_jsonl"] = str(trace_map_jsonl)
    summary["meta"]["trace_request_map_md"] = str(trace_map_md)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    report_lines.extend(
        [
            "",
            "## Trace Mapping",
            "",
            f"- trace_request_map_jsonl: `{trace_map_jsonl}`",
            f"- trace_request_map_md: `{trace_map_md}`",
        ]
    )
    compare_report_md.write_text("\n".join(report_lines), encoding="utf-8")

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate real triage/doctor chain with department normalization mapping.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--timeout-s", type=float, default=45.0)
    parser.add_argument(
        "--prompt-suffix",
        default="请仅输出推荐就诊科室（1到3个），直接给出科室名，不要解释。",
    )
    parser.add_argument("--outdir", default="")
    args = parser.parse_args()

    if not args.outdir:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.outdir = str(PROJECT_ROOT / "logs" / "rca" / "triage_doctor_real_eval" / ts)
    return args


def main() -> None:
    args = parse_args()
    summary = asyncio.run(run_eval(args))
    print(json.dumps({"outdir": args.outdir, "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
