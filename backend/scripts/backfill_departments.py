from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from typing import Any

from app.config import get_settings
from app.db import Database
from app.policy import normalize_department


async def backfill(*, dry_run: bool) -> dict[str, Any]:
    settings = get_settings()
    db = Database(settings)
    await db.connect()
    try:
        rows = await db.fetch("SELECT id, department FROM documents ORDER BY id")
        before = Counter(row["department"] for row in rows)
        planned = [(row["id"], normalize_department(row["department"])) for row in rows]
        after = Counter(new for _, new in planned)
        changes = [
            (row["id"], new)
            for row, (_, new) in zip(rows, planned)
            if row["department"] != new
        ]
        if changes and not dry_run:
            async with db.pool.acquire() as conn:
                async with conn.transaction():
                    await conn.executemany(
                        "UPDATE documents SET department=$2 WHERE id=$1",
                        changes,
                    )
        return {
            "total": len(rows),
            "changed": len(changes),
            "dry_run": dry_run,
            "before": dict(before.most_common()),
            "after": dict(after.most_common()),
        }
    finally:
        await db.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize documents.department to canonical names.")
    parser.add_argument("--dry-run", action="store_true", help="report changes without writing")
    args = parser.parse_args()
    result = asyncio.run(backfill(dry_run=args.dry_run))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
