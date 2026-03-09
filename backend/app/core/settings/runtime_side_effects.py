from __future__ import annotations

import os
from typing import Any


def is_valid_key(key: str) -> bool:
    if not key or len(key) < 20:
        return False
    placeholders = ["sk-placeholder", "sk-example"]
    return not any(p in key for p in placeholders)


def mask_key(key: str) -> str:
    if not key:
        return "empty"
    if len(key) <= 12:
        return "****"
    return f"{key[:8]}...{key[-4:]}"


def build_key_candidates(
    *,
    openai_api_key: str,
    dashscope_api_key: str,
    dashscope_api_key_pool: str,
    api_key_rotation_list: str,
) -> list[str]:
    candidates = set()

    if openai_api_key and is_valid_key(openai_api_key):
        candidates.add(openai_api_key)

    if dashscope_api_key and is_valid_key(dashscope_api_key):
        candidates.add(dashscope_api_key)

    if dashscope_api_key_pool:
        pool_keys = [
            k.strip()
            for k in dashscope_api_key_pool.split(",")
            if is_valid_key(k.strip())
        ]
        candidates.update(pool_keys)

    if api_key_rotation_list:
        rotation_keys = [
            k.strip()
            for k in api_key_rotation_list.split(",")
            if is_valid_key(k.strip())
        ]
        candidates.update(rotation_keys)

    if openai_api_key and len(openai_api_key) > 10:
        candidates.add(openai_api_key)

    if not candidates:
        env_key = os.getenv("OPENAI_API_KEY")
        if env_key and len(env_key) > 10:
            candidates.add(env_key)

    return list(candidates)


def export_runtime_env(settings_obj: Any) -> None:
    os.environ["LANGCHAIN_TRACING_V2"] = str(getattr(settings_obj, "LANGCHAIN_TRACING_V2", "false"))
    os.environ["LANGCHAIN_ENDPOINT"] = str(getattr(settings_obj, "LANGCHAIN_ENDPOINT", ""))
    os.environ["LANGCHAIN_API_KEY"] = str(getattr(settings_obj, "LANGCHAIN_API_KEY", ""))
    os.environ["LANGCHAIN_PROJECT"] = str(getattr(settings_obj, "LANGCHAIN_PROJECT", ""))
    os.environ["OPENAI_API_BASE"] = str(getattr(settings_obj, "OPENAI_API_BASE", ""))
    os.environ["OPENAI_BASE_URL"] = str(getattr(settings_obj, "OPENAI_API_BASE", ""))
    os.environ["OPENAI_API_KEY"] = str(getattr(settings_obj, "OPENAI_API_KEY", ""))
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


__all__ = [
    "build_key_candidates",
    "export_runtime_env",
    "is_valid_key",
    "mask_key",
]
