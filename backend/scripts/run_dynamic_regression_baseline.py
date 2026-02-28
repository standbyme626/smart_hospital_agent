from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from app.rag.dynamic_regression import (
    default_dynamic_scenarios,
    evaluate_dynamic_regression_scenarios,
    write_dynamic_regression_baseline,
)


def _read_scenarios(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [dict(item) for item in payload if isinstance(item, dict)]
        return []

    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        if isinstance(data, dict):
            rows.append(data)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Build dynamic regression baseline report")
    parser.add_argument("--input", help="Path to dynamic scenarios (.json or .jsonl)")
    parser.add_argument("--output-dir", required=True, help="Directory to write baseline reports")
    parser.add_argument("--enforce", action="store_true", help="Exit non-zero when failed scenarios exist")
    args = parser.parse_args()

    scenarios: List[Dict[str, Any]]
    if args.input:
        scenarios = _read_scenarios(Path(args.input))
    else:
        scenarios = default_dynamic_scenarios()

    report = evaluate_dynamic_regression_scenarios(scenarios=scenarios)
    output = write_dynamic_regression_baseline(report=report, output_dir=args.output_dir)

    print(json.dumps({"report": report, "output": output}, ensure_ascii=False, indent=2))

    if args.enforce and int(report.get("failed", 0)) > 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
