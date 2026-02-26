#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

CATS = ["protocol", "model", "retrieval", "routing"]


def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize(rows: List[Dict], sample_limit: int = 5) -> Dict:
    counts = {c: 0 for c in CATS}
    samples = {c: [] for c in CATS}

    for row in rows:
        ft = row.get("failure_type")
        if not ft:
            continue
        if ft not in counts:
            continue
        counts[ft] += 1
        if len(samples[ft]) < sample_limit:
            samples[ft].append(
                {
                    "text": row.get("question") or row.get("query"),
                    "label": row.get("label") or row.get("label_intent"),
                    "pred": row.get("pred") or row.get("pred_intent"),
                    "expected_path": row.get("expected_path"),
                    "pred_path": row.get("pred_path"),
                    "detail": row.get("detail", ""),
                }
            )

    return {"counts": counts, "samples": samples}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="logs/intent_benchmark/<run_dir>")
    parser.add_argument("--output", default="failure_taxonomy_report.json")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)

    before_rows = []
    before_rows += load_jsonl(run_dir / "before_intent_details.jsonl")
    before_rows += load_jsonl(run_dir / "before_triage_details.jsonl")

    after_rows = []
    after_rows += load_jsonl(run_dir / "after_intent_details.jsonl")
    after_rows += load_jsonl(run_dir / "after_triage_details.jsonl")

    before = summarize(before_rows)
    after = summarize(after_rows)

    delta = {
        c: after["counts"][c] - before["counts"][c]
        for c in CATS
    }

    report = {
        "categories": CATS,
        "before": before,
        "after": after,
        "delta": delta,
    }

    out_path = run_dir / args.output
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[OK] {out_path}")
    print(json.dumps(report["delta"], ensure_ascii=False))


if __name__ == "__main__":
    main()
