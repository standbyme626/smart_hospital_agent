from __future__ import annotations

import ast
import json
import re
from typing import Any, List


UI_SLOTS_PATTERN = re.compile(
    r"<ui_slots>\s*(?P<body>.*?)\s*</ui_slots>",
    flags=re.IGNORECASE | re.DOTALL,
)

UI_PAYMENT_PATTERN = re.compile(
    r"<ui_payment>\s*(?P<body>.*?)\s*</ui_payment>",
    flags=re.IGNORECASE | re.DOTALL,
)


def _parse_structured_body(body: str) -> Any:
    if not body:
        return body
    try:
        return json.loads(body)
    except Exception:
        try:
            # Some legacy prompts may emit single-quoted pseudo-JSON.
            return ast.literal_eval(body)
        except Exception:
            return body


def extract_doctor_slots(text: str) -> List[Any]:
    """
    Parse `<ui_slots>...</ui_slots>` blocks and return structured payloads.
    Returns an empty list when no valid slots block is found.
    """
    if not text:
        return []

    slots_payloads: List[Any] = []
    for match in UI_SLOTS_PATTERN.finditer(text):
        body = (match.group("body") or "").strip()
        if not body:
            continue
        slots_payloads.append(_parse_structured_body(body))
    return slots_payloads


def extract_ui_payment(text: str) -> List[Any]:
    """
    Parse `<ui_payment>...</ui_payment>` blocks and return structured payloads.
    Returns an empty list when no valid payment block is found.
    """
    if not text:
        return []

    payment_payloads: List[Any] = []
    for match in UI_PAYMENT_PATTERN.finditer(text):
        body = (match.group("body") or "").strip()
        if not body:
            continue
        payment_payloads.append(_parse_structured_body(body))
    return payment_payloads
    return slots_payloads
