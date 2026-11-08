from __future__ import annotations

import argparse
import asyncio
import getpass

from app.config import get_settings
from app.db import Database
from app.security import hash_password


async def main() -> None:
    parser = argparse.ArgumentParser(description="Rotate the password hash for a user.")
    parser.add_argument("username", help="user id/username to update")
    args = parser.parse_args()

    password = getpass.getpass("new password: ")
    if len(password) < 6:
        raise SystemExit("password must be at least 6 characters")

    settings = get_settings()
    db = Database(settings)
    try:
        await db.init_schema()
        status = await db.execute(
            "UPDATE users SET password_hash=$2, updated_at=now() WHERE id=$1",
            args.username,
            hash_password(password),
        )
    finally:
        await db.close()

    if status == "UPDATE 0":
        raise SystemExit(f"user not found: {args.username}")
    print(f"password updated for {args.username}")


if __name__ == "__main__":
    asyncio.run(main())
