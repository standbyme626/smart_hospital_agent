from __future__ import annotations

from typing import Any


def build_diagnosis_output(
    *,
    department_top1: str,
    department_top3: list[str],
    confidence: float,
    reasoning: str,
    citations: list[dict[str, str]],
) -> dict[str, Any]:
    from app.core.graph.sub_graphs import diagnosis as legacy_diagnosis

    return legacy_diagnosis._build_diagnosis_output(
        department_top1=department_top1,
        department_top3=department_top3,
        confidence=confidence,
        reasoning=reasoning,
        citations=citations,
    )


__all__ = [
    "build_diagnosis_output",
]
