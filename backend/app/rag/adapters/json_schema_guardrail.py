from __future__ import annotations

import inspect
from typing import Any, Awaitable, Callable, Dict, List, Optional

import structlog

logger = structlog.get_logger(__name__)

RepairFn = Callable[[Dict[str, Any]], Dict[str, Any] | Awaitable[Dict[str, Any]]]


class JsonSchemaGuardrail:
    def __init__(self, *, schema_version: str = "v1", enabled: bool = False) -> None:
        self.schema_version = str(schema_version or "v1")
        self.enabled = bool(enabled)

    async def validate_and_repair(
        self,
        diagnosis_output: Dict[str, Any],
        *,
        repair_fn: Optional[RepairFn] = None,
    ) -> Dict[str, Any]:
        payload = dict(diagnosis_output or {})
        payload.setdefault("diagnosis_schema_version", self.schema_version)

        if not self.enabled:
            return {
                "diagnosis_output": payload,
                "validated": True,
                "validation_error": None,
                "repair_attempted": False,
            }

        errors = self._validate(payload)
        if not errors:
            return {
                "diagnosis_output": payload,
                "validated": True,
                "validation_error": None,
                "repair_attempted": False,
            }

        error_summary = "; ".join(errors)
        repair_attempted = True
        logger.warning("diagnosis_output_schema_invalid", validation_error=error_summary)

        repair_input = {
            "previous_json": payload,
            "validation_error_summary": error_summary,
            "schema_key_fields": self._schema_key_fields(),
        }

        repaired = await self._run_repair(repair_input, repair_fn)
        repaired.setdefault("diagnosis_schema_version", self.schema_version)
        repaired_errors = self._validate(repaired)
        if not repaired_errors:
            return {
                "diagnosis_output": repaired,
                "validated": True,
                "validation_error": error_summary,
                "repair_attempted": repair_attempted,
            }

        safe_output = self._safe_template(reason="schema_validation_failed")
        return {
            "diagnosis_output": safe_output,
            "validated": False,
            "validation_error": "; ".join(repaired_errors),
            "repair_attempted": repair_attempted,
        }

    async def _run_repair(self, repair_input: Dict[str, Any], repair_fn: Optional[RepairFn]) -> Dict[str, Any]:
        if repair_fn is not None:
            try:
                candidate = repair_fn(repair_input)
                if inspect.isawaitable(candidate):
                    candidate = await candidate
                if isinstance(candidate, dict):
                    return dict(candidate)
            except Exception as exc:
                logger.warning("diagnosis_output_repair_fn_failed", error=str(exc))

        return self._deterministic_repair(repair_input.get("previous_json"))

    def _deterministic_repair(self, payload: Any) -> Dict[str, Any]:
        candidate = dict(payload or {}) if isinstance(payload, dict) else {}

        diagnosis_schema_version = str(candidate.get("diagnosis_schema_version") or self.schema_version)
        department_top1 = str(candidate.get("department_top1") or "Unknown")

        raw_top3 = candidate.get("department_top3")
        if isinstance(raw_top3, list):
            department_top3 = [str(item) for item in raw_top3 if str(item or "").strip()]
        else:
            department_top3 = []
        if not department_top3 and department_top1 and department_top1 != "Unknown":
            department_top3 = [department_top1]

        try:
            confidence = float(candidate.get("confidence", 0.0))
        except Exception:
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        reasoning = str(candidate.get("reasoning") or "Insufficient validated evidence.")

        citations = self._normalize_citations(candidate.get("citations"))

        return {
            "diagnosis_schema_version": diagnosis_schema_version,
            "department_top1": department_top1,
            "department_top3": department_top3,
            "confidence": confidence,
            "reasoning": reasoning,
            "citations": citations,
        }

    def _safe_template(self, *, reason: str) -> Dict[str, Any]:
        return {
            "diagnosis_schema_version": self.schema_version,
            "department_top1": "Unknown",
            "department_top3": [],
            "confidence": 0.0,
            "reasoning": f"Guardrail fallback: {reason}.",
            "citations": [],
        }

    def _normalize_citations(self, citations: Any) -> List[Dict[str, str]]:
        normalized: List[Dict[str, str]] = []
        if not isinstance(citations, list):
            return normalized

        for item in citations:
            if not isinstance(item, dict):
                continue
            doc_id = str(item.get("doc_id") or "").strip()
            chunk_id = str(item.get("chunk_id") or "").strip()
            span = str(item.get("span") or "").strip()
            if not doc_id or not chunk_id or not span:
                continue
            normalized.append({"doc_id": doc_id, "chunk_id": chunk_id, "span": span})
        return normalized

    def _validate(self, payload: Dict[str, Any]) -> List[str]:
        errors: List[str] = []

        schema_version = payload.get("diagnosis_schema_version")
        if not isinstance(schema_version, str) or not schema_version.strip():
            errors.append("diagnosis_schema_version must be non-empty string")

        department_top1 = payload.get("department_top1")
        if not isinstance(department_top1, str) or not department_top1.strip():
            errors.append("department_top1 must be non-empty string")

        department_top3 = payload.get("department_top3")
        if not isinstance(department_top3, list) or any(not isinstance(item, str) for item in department_top3):
            errors.append("department_top3 must be list[str]")

        confidence = payload.get("confidence")
        if not isinstance(confidence, (int, float)):
            errors.append("confidence must be number")
        else:
            confidence_val = float(confidence)
            if confidence_val < 0 or confidence_val > 1:
                errors.append("confidence must be within [0, 1]")

        reasoning = payload.get("reasoning")
        if not isinstance(reasoning, str):
            errors.append("reasoning must be string")

        citations = payload.get("citations")
        if not isinstance(citations, list):
            errors.append("citations must be list")
        else:
            for idx, item in enumerate(citations):
                if not isinstance(item, dict):
                    errors.append(f"citations[{idx}] must be object")
                    continue
                for key in ("doc_id", "chunk_id", "span"):
                    if not isinstance(item.get(key), str) or not str(item.get(key)).strip():
                        errors.append(f"citations[{idx}].{key} must be non-empty string")

        return errors

    def _schema_key_fields(self) -> Dict[str, Any]:
        return {
            "required": [
                "diagnosis_schema_version",
                "department_top1",
                "department_top3",
                "confidence",
                "reasoning",
                "citations",
            ],
            "citations_item": {"doc_id": "string", "chunk_id": "string", "span": "string"},
        }
