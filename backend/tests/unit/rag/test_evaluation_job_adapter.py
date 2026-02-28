import json
from pathlib import Path

import pytest

from app.rag.evaluation_job_adapter import build_metric_result, write_weekly_baseline


def test_build_metric_result_requires_stage_e_identity_fields():
    with pytest.raises(ValueError):
        build_metric_result(metric_job={"request_id": "req-1"})


def test_write_weekly_baseline_outputs_jsonl_and_md(tmp_path: Path):
    result_ok = build_metric_result(
        metric_job={
            "request_id": "req-1",
            "dataset_version": "gold-v1",
            "model_version": "model-v1",
            "task_type": "diagnosis",
        },
        instance_metrics={"mrr": 0.8, "recall@5": 0.9},
        summary_metrics={"faithfulness": 0.86},
        failure_stats={"total": 10, "failed": 1},
    )
    result_fail = build_metric_result(
        metric_job={
            "request_id": "req-2",
            "dataset_version": "gold-v1",
            "model_version": "model-v1",
            "task_type": "diagnosis",
        },
        instance_metrics={"mrr": 0.0},
        summary_metrics={"faithfulness": 0.0},
        failure_stats={"total": 10, "failed": 10},
        status="failed",
    )

    summary = write_weekly_baseline(metric_results=[result_ok, result_fail], output_dir=tmp_path)

    assert summary["sample_count"] == 2
    assert summary["ok_count"] == 1
    assert summary["fail_count"] == 1

    jsonl_path = Path(summary["jsonl_path"])
    md_path = Path(summary["md_path"])
    assert jsonl_path.exists()
    assert md_path.exists()

    rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 2
    assert rows[0]["request_id"] == "req-1"
    assert set(rows[0].keys()) >= {
        "request_id",
        "dataset_version",
        "model_version",
        "task_type",
        "instance_metrics",
        "summary_metrics",
        "failure_stats",
    }

    md = md_path.read_text(encoding="utf-8")
    assert "Weekly Baseline" in md
    assert "req-1" in md and "req-2" in md
