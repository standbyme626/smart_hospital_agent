from __future__ import annotations

from typing import Any


STORAGE_FIELDS: tuple[str, ...] = (
    "DATABASE_URL",
    "POSTGRES_SERVER",
    "POSTGRES_USER",
    "POSTGRES_DB",
    "POSTGRES_PORT",
    "REDIS_URL",
    "REDIS_HOST",
    "REDIS_PORT",
    "MILVUS_HOST",
    "MILVUS_PORT",
)


def snapshot(settings_obj: Any) -> dict[str, Any]:
    return {name: getattr(settings_obj, name, None) for name in STORAGE_FIELDS}
