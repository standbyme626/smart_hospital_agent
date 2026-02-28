import json
from pathlib import Path

from app.rag.dynamic_regression import (
    evaluate_dynamic_regression_scenarios,
    write_dynamic_regression_baseline,
)


def test_dynamic_regression_protocol_summary():
    scenarios = [
        {
            "scenario_id": "ok-1",
            "risk_level": "baseline",
            "expected_outcome": "diagnosis_ready",
            "events": [
                {"event": "request_test"},
                {"event": "provide_result"},
                {"event": "diagnosis_ready"},
            ],
        },
        {
            "scenario_id": "bad-1",
            "risk_level": "high",
            "expected_outcome": "escalate",
            "events": [
                {"event": "provide_result"},
                {"event": "diagnosis_ready"},
            ],
        },
    ]

    report = evaluate_dynamic_regression_scenarios(scenarios=scenarios)

    assert report["total"] == 2
    assert report["passed"] == 1
    assert report["failed"] == 1
    assert report["event_coverage"]["request_test"] == 1
    assert "invalid_start_event" in report["failure_types"]


def test_dynamic_regression_baseline_outputs_jsonl_and_md(tmp_path: Path):
    report = evaluate_dynamic_regression_scenarios(
        scenarios=[
            {
                "scenario_id": "ok-1",
                "risk_level": "baseline",
                "expected_outcome": "diagnosis_ready",
                "events": [
                    {"event": "request_test"},
                    {"event": "diagnosis_ready"},
                ],
            }
        ]
    )

    summary = write_dynamic_regression_baseline(report=report, output_dir=tmp_path)
    jsonl_path = Path(summary["jsonl_path"])
    md_path = Path(summary["md_path"])

    assert jsonl_path.exists()
    assert md_path.exists()

    rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["scenario_id"] == "ok-1"

    md = md_path.read_text(encoding="utf-8")
    assert "Dynamic Regression Baseline" in md
    assert "ok-1" in md
