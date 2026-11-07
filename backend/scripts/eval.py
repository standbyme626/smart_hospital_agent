from __future__ import annotations

import argparse
import asyncio
import json

from app.config import get_settings
from app.db import Database
from app.eval_runner import EvalRunner
from app.model_gateway import ModelGateway
from app.retrieval import RetrievalService


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval + crisis smoke evaluation.")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    settings = get_settings()
    db = Database(settings)
    await db.init_schema()
    gateway = ModelGateway(settings)
    retrieval = RetrievalService(settings, db, gateway)
    runner = EvalRunner(settings, db, retrieval, gateway)
    try:
        metrics = await runner.run(limit=args.limit)
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
    finally:
        await gateway.close()
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
