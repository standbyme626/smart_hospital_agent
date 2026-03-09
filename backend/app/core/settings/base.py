from __future__ import annotations

from typing import Any


BASE_FIELDS: tuple[str, ...] = (
    "PROJECT_NAME",
    "VERSION",
    "API_V1_STR",
    "PROJECT_ROOT",
)


def snapshot(settings_obj: Any) -> dict[str, Any]:
    return {name: getattr(settings_obj, name, None) for name in BASE_FIELDS}
