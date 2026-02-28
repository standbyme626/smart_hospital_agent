import json
from pathlib import Path

from app.rag.evaluation_gate import evaluate_release_gate, write_gate_report


def _row(request_id: str, *, mrr: float, faithfulness: float, status: str = "ok", failed: int = 0):
    return {
        "request_id": request_id,
        "dataset_version": "gold-v1",
        "model_version": "model-v1",
        "task_type": "diagnosis",
        "status": status,
        "instance_metrics": {"mrr": mrr},
        "summary_metrics": {"faithfulness": faithfulness},
        "failure_stats": {"total": 1, "failed": failed},
    }


def test_release_gate_pass_with_non_regression_and_redline_clean():
    current = [_row("req-1", mrr=0.82, faithfulness=0.9), _row("req-2", mrr=0.78, faithfulness=0.88)]
    baseline = [_row("base-1", mrr=0.8, faithfulness=0.85), _row("base-2", mrr=0.75, faithfulness=0.84)]

    gate = evaluate_release_gate(
        metric_results=current,
        baseline_results=baseline,
        redline_request_ids=["req-1"],
        enforce=True,
    )

    assert gate["status"] == "passed"
    assert gate["gate_passed"] is True
    assert gate["checks"]["redline_zero_failure"]["passed"] is True
    assert gate["checks"]["retrieval_non_regression"]["passed"] is True
    assert gate["checks"]["generation_non_regression"]["passed"] is True


def test_release_gate_fails_when_redline_failed():
    current = [_row("req-1", mrr=0.82, faithfulness=0.9, status="failed", failed=1)]
    baseline = [_row("base-1", mrr=0.8, faithfulness=0.85)]

    gate = evaluate_release_gate(
        metric_results=current,
        baseline_results=baseline,
        redline_request_ids=["req-1"],
        enforce=True,
    )

    assert gate["status"] == "failed"
    assert gate["gate_passed"] is False
    assert "redline_failed" in gate["reasons"]


def test_release_gate_bootstrap_when_baseline_missing():
    current = [_row("req-1", mrr=0.82, faithfulness=0.9)]

    gate = evaluate_release_gate(
        metric_results=current,
        baseline_results=[],
        redline_request_ids=[],
        enforce=True,
    )

    assert gate["status"] == "bootstrap"
    assert gate["bootstrap_mode"] is True
    assert gate["gate_passed"] is True


def test_write_gate_report_outputs_json_and_md(tmp_path: Path):
    gate = evaluate_release_gate(
        metric_results=[_row("req-1", mrr=0.82, faithfulness=0.9)],
        baseline_results=[],
    )

    summary = write_gate_report(gate_result=gate, output_dir=tmp_path)
    json_path = Path(summary["json_path"])
    md_path = Path(summary["md_path"])

    assert json_path.exists()
    assert md_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["status"] == "bootstrap"

    md = md_path.read_text(encoding="utf-8")
    assert "Release Gate Report" in md
    assert "bootstrap_mode" in md
