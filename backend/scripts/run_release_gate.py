from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from app.rag.evaluation_gate import evaluate_release_gate, write_gate_report


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _read_redline(path: Path) -> List[str]:
    ids: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        item = line.strip()
        if not item or item.startswith("#"):
            continue
        ids.append(item)
    return ids


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate release gate from weekly baseline metric results")
    parser.add_argument("--metric-results", required=True, help="Path to weekly_baseline.jsonl")
    parser.add_argument("--baseline-results", help="Path to baseline weekly_baseline.jsonl")
    parser.add_argument("--redline", help="Path to redline request_id list")
    parser.add_argument("--output-dir", required=True, help="Directory to write gate reports")
    parser.add_argument("--enforce", action="store_true", help="Fail command when gate fails")
    args = parser.parse_args()

    metric_rows = _read_jsonl(Path(args.metric_results))
    baseline_rows = _read_jsonl(Path(args.baseline_results)) if args.baseline_results else []
    redline_ids = _read_redline(Path(args.redline)) if args.redline else []

    gate = evaluate_release_gate(
        metric_results=metric_rows,
        baseline_results=baseline_rows,
        redline_request_ids=redline_ids,
        enforce=args.enforce,
    )
    report = write_gate_report(gate_result=gate, output_dir=args.output_dir)

    print(json.dumps({"gate": gate, "report": report}, ensure_ascii=False, indent=2))

    if args.enforce and gate.get("status") == "failed":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
