#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX = PROJECT_ROOT / "logs" / "trace_replay" / "replay_index.jsonl"


def load_entries(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lookup request_id -> logs/e2e/weekly artifacts.")
    parser.add_argument("--request-id", required=True, help="request_id / trace_id")
    parser.add_argument("--index", default=str(DEFAULT_INDEX), help="Path to replay_index.jsonl")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    index_path = Path(args.index).resolve()
    entries = load_entries(index_path)
    target = str(args.request_id or "").strip()
    hits = [entry for entry in entries if str(entry.get("request_id", "")).strip() == target]

    result = {
        "index_path": str(index_path),
        "request_id": target,
        "hit_count": len(hits),
        "matches": hits,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if hits else 2


if __name__ == "__main__":
    raise SystemExit(main())

