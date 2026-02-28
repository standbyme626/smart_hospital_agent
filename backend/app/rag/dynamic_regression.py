from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence
import json

_ALLOWED_EVENTS = {"request_test", "provide_result", "diagnosis_ready", "escalate"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_dynamic_scenarios() -> List[Dict[str, Any]]:
    return [
        {
            "scenario_id": "base-01",
            "risk_level": "baseline",
            "expected_outcome": "diagnosis_ready",
            "events": [
                {"event": "request_test", "payload": {"symptom": "胃痛"}},
                {"event": "provide_result", "payload": {"lab": "normal"}},
                {"event": "diagnosis_ready", "payload": {"department": "消化内科"}},
            ],
        },
        {
            "scenario_id": "risk-01",
            "risk_level": "high",
            "expected_outcome": "escalate",
            "events": [
                {"event": "request_test", "payload": {"symptom": "胸痛"}},
                {"event": "provide_result", "payload": {"ecg": "abnormal"}},
                {"event": "escalate", "payload": {"target": "human_review"}},
            ],
        },
    ]


def _normalize_events(events: Any) -> List[str]:
    if not isinstance(events, Sequence):
        return []
    normalized: List[str] = []
    for item in events:
        if isinstance(item, Mapping):
            evt = str(item.get("event") or "").strip()
        else:
            evt = str(item or "").strip()
        if evt:
            normalized.append(evt)
    return normalized


def evaluate_dynamic_regression_scenarios(
    *,
    scenarios: Iterable[Mapping[str, Any]],
) -> Dict[str, Any]:
    scenario_results: List[Dict[str, Any]] = []
    event_coverage: Dict[str, int] = {event: 0 for event in sorted(_ALLOWED_EVENTS)}
    failure_types: Dict[str, int] = {}

    for raw in scenarios:
        scenario_id = str(raw.get("scenario_id") or "").strip() or f"scenario-{len(scenario_results)+1}"
        risk_level = str(raw.get("risk_level") or "baseline").strip() or "baseline"
        expected_outcome = str(raw.get("expected_outcome") or "diagnosis_ready").strip() or "diagnosis_ready"
        events = _normalize_events(raw.get("events"))

        reasons: List[str] = []
        if not events:
            reasons.append("empty_events")
        else:
            unknown = [event for event in events if event not in _ALLOWED_EVENTS]
            if unknown:
                reasons.append("unknown_event")
            if events and events[0] != "request_test":
                reasons.append("invalid_start_event")
            if "diagnosis_ready" not in events and "escalate" not in events:
                reasons.append("missing_terminal_event")
            if expected_outcome == "diagnosis_ready" and "diagnosis_ready" not in events:
                reasons.append("missing_expected_diagnosis_ready")
            if expected_outcome == "escalate" and "escalate" not in events:
                reasons.append("missing_expected_escalate")

        for event in set(events):
            if event in event_coverage:
                event_coverage[event] += 1

        passed = len(reasons) == 0
        for reason in reasons:
            failure_types[reason] = failure_types.get(reason, 0) + 1

        scenario_results.append(
            {
                "scenario_id": scenario_id,
                "risk_level": risk_level,
                "expected_outcome": expected_outcome,
                "events": events,
                "status": "ok" if passed else "failed",
                "reasons": reasons,
            }
        )

    total = len(scenario_results)
    passed_count = sum(1 for item in scenario_results if item.get("status") == "ok")
    failed_count = total - passed_count

    return {
        "generated_at": _utc_now(),
        "total": total,
        "passed": passed_count,
        "failed": failed_count,
        "pass_rate": round((passed_count / total), 6) if total else 0.0,
        "event_coverage": event_coverage,
        "failure_types": failure_types,
        "scenario_results": scenario_results,
    }


def write_dynamic_regression_baseline(
    *,
    report: Mapping[str, Any],
    output_dir: str | Path,
    report_prefix: str = "dynamic_regression_baseline",
) -> Dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rows = report.get("scenario_results") if isinstance(report.get("scenario_results"), list) else []

    jsonl_path = output_path / f"{report_prefix}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    md_path = output_path / f"{report_prefix}.md"
    with md_path.open("w", encoding="utf-8") as fh:
        fh.write("# Dynamic Regression Baseline\n\n")
        fh.write(f"- total: {report.get('total', 0)}\n")
        fh.write(f"- passed: {report.get('passed', 0)}\n")
        fh.write(f"- failed: {report.get('failed', 0)}\n")
        fh.write(f"- pass_rate: {report.get('pass_rate', 0.0)}\n\n")

        fh.write("## Event Coverage\n\n")
        for event, cnt in (report.get("event_coverage") or {}).items():
            fh.write(f"- {event}: {cnt}\n")
        fh.write("\n")

        fh.write("| scenario_id | risk_level | expected_outcome | status | reasons |\n")
        fh.write("| --- | --- | --- | --- | --- |\n")
        for row in rows:
            reasons = ",".join(row.get("reasons") or []) or "none"
            fh.write(
                f"| {row.get('scenario_id')} | {row.get('risk_level')} | "
                f"{row.get('expected_outcome')} | {row.get('status')} | {reasons} |\n"
            )

    return {
        "jsonl_path": str(jsonl_path),
        "md_path": str(md_path),
        "total": report.get("total", 0),
        "passed": report.get("passed", 0),
        "failed": report.get("failed", 0),
    }
