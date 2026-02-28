from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping
import json

_REQUIRED_JOB_FIELDS = ("request_id", "dataset_version", "model_version", "task_type")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _coerce_metrics(metrics: Any) -> Dict[str, float]:
    if not isinstance(metrics, Mapping):
        return {}
    normalized: Dict[str, float] = {}
    for key, value in metrics.items():
        try:
            normalized[str(key)] = float(value)
        except Exception:
            continue
    return normalized


def _validate_metric_job(job: Mapping[str, Any]) -> Dict[str, Any]:
    normalized = {key: str(job.get(key) or "").strip() for key in _REQUIRED_JOB_FIELDS}
    missing = [key for key, value in normalized.items() if not value]
    if missing:
        raise ValueError(f"metric_job missing required fields: {', '.join(missing)}")

    return {
        "request_id": normalized["request_id"],
        "dataset_version": normalized["dataset_version"],
        "model_version": normalized["model_version"],
        "task_type": normalized["task_type"],
    }


def build_metric_result(
    *,
    metric_job: Mapping[str, Any],
    instance_metrics: Mapping[str, Any] | None = None,
    summary_metrics: Mapping[str, Any] | None = None,
    failure_stats: Mapping[str, Any] | None = None,
    status: str = "ok",
) -> Dict[str, Any]:
    """
    Stage E minimal contract:
    - Input must include request_id / dataset_version / model_version / task_type.
    - Output must include instance_metrics / summary_metrics / failure_stats.
    """
    base = _validate_metric_job(metric_job)
    failures = dict(failure_stats or {})
    failures.setdefault("total", 0)
    failures.setdefault("failed", 0)

    return {
        **base,
        "status": str(status or "ok"),
        "instance_metrics": _coerce_metrics(instance_metrics),
        "summary_metrics": _coerce_metrics(summary_metrics),
        "failure_stats": failures,
        "generated_at": _utc_now(),
    }


def write_weekly_baseline(
    *,
    metric_results: Iterable[Mapping[str, Any]],
    output_dir: str | Path,
) -> Dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for item in metric_results:
        row = {
            "request_id": str(item.get("request_id") or "").strip(),
            "dataset_version": str(item.get("dataset_version") or "").strip(),
            "model_version": str(item.get("model_version") or "").strip(),
            "task_type": str(item.get("task_type") or "").strip(),
            "instance_metrics": _coerce_metrics(item.get("instance_metrics")),
            "summary_metrics": _coerce_metrics(item.get("summary_metrics")),
            "failure_stats": dict(item.get("failure_stats") or {}),
            "status": str(item.get("status") or "ok"),
            "generated_at": str(item.get("generated_at") or _utc_now()),
        }
        if not all(row[field] for field in _REQUIRED_JOB_FIELDS):
            raise ValueError("metric_result missing required identity fields")
        rows.append(row)

    jsonl_path = output_path / "weekly_baseline.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    sample_count = len(rows)
    ok_count = sum(1 for row in rows if row.get("status") == "ok")
    fail_count = sample_count - ok_count

    md_path = output_path / "weekly_baseline.md"
    with md_path.open("w", encoding="utf-8") as fh:
        fh.write("# Weekly Baseline\n\n")
        fh.write(f"- sample_count: {sample_count}\n")
        fh.write(f"- ok_count: {ok_count}\n")
        fh.write(f"- fail_count: {fail_count}\n\n")
        fh.write("| request_id | dataset_version | model_version | task_type | status |\n")
        fh.write("| --- | --- | --- | --- | --- |\n")
        for row in rows:
            fh.write(
                f"| {row['request_id']} | {row['dataset_version']} | {row['model_version']} | {row['task_type']} | {row['status']} |\n"
            )

    return {
        "jsonl_path": str(jsonl_path),
        "md_path": str(md_path),
        "sample_count": sample_count,
        "ok_count": ok_count,
        "fail_count": fail_count,
    }
