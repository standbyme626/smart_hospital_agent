from __future__ import annotations

from typing import Any


MODEL_FIELDS: tuple[str, ...] = (
    "OPENAI_API_BASE",
    "OPENAI_MODEL_NAME",
    "OPENAI_MODEL_FAST",
    "OPENAI_MODEL_SMART",
    "MODEL_SMART",
    "MODEL_FAST",
    "MODEL_CODER",
    "MODEL_LONG",
    "PREFER_LOCAL",
    "ENABLE_LOCAL_FALLBACK",
    "LOCAL_SLM_URL",
    "LOCAL_SLM_MODEL",
)


def snapshot(settings_obj: Any) -> dict[str, Any]:
    return {name: getattr(settings_obj, name, None) for name in MODEL_FIELDS}
