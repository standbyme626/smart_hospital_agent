from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from app.config import REPO_ROOT, get_settings
from app.db import Database
from app.model_gateway import ModelGateway
from app.retrieval import RetrievalService

DEFAULT_FILES = [
    REPO_ROOT / "data" / "medical_gold_hardened_v5.jsonl",
    REPO_ROOT / "data" / "medical_gold_final_v4.jsonl",
    REPO_ROOT / "data" / "medical_gold_v3_raw_54k.jsonl",
]


async def main() -> None:
    parser = argparse.ArgumentParser(description="Seed the retrieval index from a medical JSONL file.")
    parser.add_argument("--path", default="", help="JSONL path; defaults to the first available medical_gold file.")
    parser.add_argument("--limit", type=int, default=None, help="max documents to ingest.")
    args = parser.parse_args()

    path = Path(args.path) if args.path else next((item for item in DEFAULT_FILES if item.exists()), None)
    if not path or not path.exists():
        raise SystemExit("no seed file found; pass --path")

    settings = get_settings()
    db = Database(settings)
    await db.init_schema()
    gateway = ModelGateway(settings)
    retrieval = RetrievalService(settings, db, gateway)
    try:
        inserted = await retrieval.seed_from_jsonl(path, args.limit)
        stats = await retrieval.stats()
        print(json.dumps({"file": str(path), "inserted": inserted, **stats}, ensure_ascii=False, indent=2))
    finally:
        await gateway.close()
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
