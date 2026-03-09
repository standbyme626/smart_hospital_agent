from __future__ import annotations

from typing import Any, Callable


def apply_sql_prefilter(
    *,
    query: str,
    extract_fn: Callable[[str], dict[str, Any] | None],
) -> dict[str, Any] | None:
    return extract_fn(query)
