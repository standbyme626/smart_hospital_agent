from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
import json


_DEFAULT_RETRIEVAL_METRICS: tuple[str, ...] = ("mrr", "map", "ndcg", "recall@k")
_DEFAULT_GENERATION_METRICS: tuple[str, ...] = ("faithfulness", "citation_coverage", "refusal_accuracy")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _normalize_metric_rows(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for item in rows:
        row = {
            "request_id": str(item.get("request_id") or "").strip(),
            "dataset_version": str(item.get("dataset_version") or "").strip(),
            "model_version": str(item.get("model_version") or "").strip(),
            "task_type": str(item.get("task_type") or "").strip(),
            "status": str(item.get("status") or "ok").strip().lower() or "ok",
            "instance_metrics": dict(item.get("instance_metrics") or {}),
            "summary_metrics": dict(item.get("summary_metrics") or {}),
            "failure_stats": dict(item.get("failure_stats") or {}),
            "generated_at": str(item.get("generated_at") or _utc_now()),
        }
        normalized.append(row)
    return normalized


def _metric_value(row: Mapping[str, Any], metric_name: str) -> Optional[float]:
    summary_metrics = row.get("summary_metrics") if isinstance(row.get("summary_metrics"), Mapping) else {}
    instance_metrics = row.get("instance_metrics") if isinstance(row.get("instance_metrics"), Mapping) else {}

    if metric_name in summary_metrics:
        return _safe_float(summary_metrics.get(metric_name))
    if metric_name in instance_metrics:
        return _safe_float(instance_metrics.get(metric_name))

    lowered = metric_name.lower()
    for source in (summary_metrics, instance_metrics):
        for key, value in source.items():
            if str(key).lower() == lowered:
                return _safe_float(value)
    return None


def _metric_average(rows: Sequence[Mapping[str, Any]], metric_name: str) -> Optional[float]:
    values: List[float] = []
    for row in rows:
        candidate = _metric_value(row, metric_name)
        if candidate is not None:
            values.append(candidate)
    if not values:
        return None
    return sum(values) / len(values)


def _compare_metric_group(
    *,
    current_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    metrics: Sequence[str],
) -> Dict[str, Any]:
    comparisons: Dict[str, Any] = {}
    passed = True
    comparable_count = 0

    for metric in metrics:
        current_val = _metric_average(current_rows, metric)
        baseline_val = _metric_average(baseline_rows, metric)
        if current_val is None or baseline_val is None:
            comparisons[metric] = {
                "comparable": False,
                "current": current_val,
                "baseline": baseline_val,
                "delta": None,
                "passed": None,
            }
            continue

        comparable_count += 1
        delta = current_val - baseline_val
        metric_passed = delta >= 0.0
        passed = passed and metric_passed
        comparisons[metric] = {
            "comparable": True,
            "current": round(current_val, 6),
            "baseline": round(baseline_val, 6),
            "delta": round(delta, 6),
            "passed": metric_passed,
        }

    return {
        "passed": passed if comparable_count > 0 else True,
        "comparable_count": comparable_count,
        "comparisons": comparisons,
    }


def evaluate_release_gate(
    *,
    metric_results: Iterable[Mapping[str, Any]],
    baseline_results: Iterable[Mapping[str, Any]] | None = None,
    redline_request_ids: Iterable[str] | None = None,
    retrieval_metrics: Sequence[str] = _DEFAULT_RETRIEVAL_METRICS,
    generation_metrics: Sequence[str] = _DEFAULT_GENERATION_METRICS,
    enforce: bool = True,
) -> Dict[str, Any]:
    current_rows = _normalize_metric_rows(metric_results)
    baseline_rows = _normalize_metric_rows(baseline_results or [])
    redline_set = {str(item).strip() for item in (redline_request_ids or []) if str(item).strip()}

    sample_count = len(current_rows)
    ok_count = sum(1 for row in current_rows if row.get("status") == "ok")
    fail_count = sample_count - ok_count

    redline_failed: List[str] = []
    for row in current_rows:
        req_id = row.get("request_id")
        if req_id not in redline_set:
            continue

        failure_stats = row.get("failure_stats") if isinstance(row.get("failure_stats"), Mapping) else {}
        failed_cnt = _safe_float(failure_stats.get("failed")) or 0.0
        if row.get("status") != "ok" or failed_cnt > 0:
            redline_failed.append(req_id)

    redline_check = {
        "passed": len(redline_failed) == 0,
        "required": sorted(redline_set),
        "failed": sorted(redline_failed),
    }

    bootstrap_mode = len(baseline_rows) == 0
    retrieval_check = _compare_metric_group(
        current_rows=current_rows,
        baseline_rows=baseline_rows,
        metrics=tuple(retrieval_metrics),
    )
    generation_check = _compare_metric_group(
        current_rows=current_rows,
        baseline_rows=baseline_rows,
        metrics=tuple(generation_metrics),
    )

    if bootstrap_mode:
        status = "bootstrap"
    elif redline_check["passed"] and retrieval_check["passed"] and generation_check["passed"]:
        status = "passed"
    else:
        status = "failed"

    gate_passed = status in {"passed", "bootstrap"}
    if status == "failed" and not enforce:
        gate_passed = True

    reasons: List[str] = []
    if not redline_check["passed"]:
        reasons.append("redline_failed")
    if not bootstrap_mode and not retrieval_check["passed"]:
        reasons.append("retrieval_regressed")
    if not bootstrap_mode and not generation_check["passed"]:
        reasons.append("generation_regressed")
    if bootstrap_mode:
        reasons.append("bootstrap_missing_baseline")

    return {
        "generated_at": _utc_now(),
        "status": status,
        "gate_passed": gate_passed,
        "enforce": bool(enforce),
        "bootstrap_mode": bootstrap_mode,
        "summary": {
            "sample_count": sample_count,
            "ok_count": ok_count,
            "fail_count": fail_count,
        },
        "checks": {
            "redline_zero_failure": redline_check,
            "retrieval_non_regression": retrieval_check,
            "generation_non_regression": generation_check,
        },
        "reasons": reasons,
    }


def write_gate_report(
    *,
    gate_result: Mapping[str, Any],
    output_dir: str | Path,
    report_prefix: str = "release_gate",
) -> Dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    json_path = output_path / f"{report_prefix}.json"
    md_path = output_path / f"{report_prefix}.md"

    payload = dict(gate_result)
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    summary = payload.get("summary") if isinstance(payload.get("summary"), Mapping) else {}
    checks = payload.get("checks") if isinstance(payload.get("checks"), Mapping) else {}
    redline = checks.get("redline_zero_failure") if isinstance(checks.get("redline_zero_failure"), Mapping) else {}

    with md_path.open("w", encoding="utf-8") as fh:
        fh.write("# Release Gate Report\n\n")
        fh.write(f"- status: {payload.get('status')}\n")
        fh.write(f"- gate_passed: {payload.get('gate_passed')}\n")
        fh.write(f"- bootstrap_mode: {payload.get('bootstrap_mode')}\n")
        fh.write(f"- sample_count: {summary.get('sample_count', 0)}\n")
        fh.write(f"- ok_count: {summary.get('ok_count', 0)}\n")
        fh.write(f"- fail_count: {summary.get('fail_count', 0)}\n")
        fh.write(f"- reasons: {', '.join(payload.get('reasons') or []) or 'none'}\n\n")

        fh.write("## Redline\n\n")
        fh.write(f"- passed: {redline.get('passed')}\n")
        fh.write(f"- failed: {', '.join(redline.get('failed') or []) or 'none'}\n\n")

        for check_name in ("retrieval_non_regression", "generation_non_regression"):
            check = checks.get(check_name) if isinstance(checks.get(check_name), Mapping) else {}
            fh.write(f"## {check_name}\n\n")
            fh.write(f"- passed: {check.get('passed')}\n")
            fh.write(f"- comparable_count: {check.get('comparable_count', 0)}\n\n")
            fh.write("| metric | comparable | current | baseline | delta | passed |\n")
            fh.write("| --- | --- | --- | --- | --- | --- |\n")
            comparisons = check.get("comparisons") if isinstance(check.get("comparisons"), Mapping) else {}
            for metric_name, detail in comparisons.items():
                detail_map = detail if isinstance(detail, Mapping) else {}
                fh.write(
                    "| "
                    f"{metric_name} | {detail_map.get('comparable')} | {detail_map.get('current')} | "
                    f"{detail_map.get('baseline')} | {detail_map.get('delta')} | {detail_map.get('passed')} |\n"
                )
            fh.write("\n")

    return {
        "json_path": str(json_path),
        "md_path": str(md_path),
        "status": payload.get("status"),
        "gate_passed": payload.get("gate_passed"),
    }
