from __future__ import annotations

from .contracts import DiagnosisStateContract


async def attach_guarded_output(
    *,
    state: DiagnosisStateContract,
    output: dict,
    diagnosis_output: dict,
) -> dict:
    from app.core.graph.sub_graphs import diagnosis as legacy_diagnosis

    return await legacy_diagnosis._attach_guarded_diagnosis_output(
        state=state,
        output=output,
        diagnosis_output=diagnosis_output,
    )


__all__ = [
    "attach_guarded_output",
]
