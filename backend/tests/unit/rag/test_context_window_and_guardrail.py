import pytest

from app.rag.adapters.context_window_adapter import ContextWindowAdapter
from app.rag.adapters.json_schema_guardrail import JsonSchemaGuardrail


def test_context_window_merge_and_diversity_filter():
    adapter = ContextWindowAdapter(
        window_size=1,
        max_evidence=4,
        max_per_source=1,
        ordering_strategy="lost_in_middle_mitigate",
        enable_window=True,
        enable_merge=True,
        enable_diversity=True,
        max_context_chars=800,
    )

    docs = [
        {
            "id": "a1",
            "doc_id": "doc-a",
            "chunk_id": "a1",
            "source_id": "src-a",
            "split_id": "split-a",
            "split_idx": 1,
            "score": 0.93,
            "content": "A1 主段",
            "child_ids": ["a2"],
            "source_type": "guideline",
        },
        {
            "id": "a2",
            "doc_id": "doc-a",
            "chunk_id": "a2",
            "source_id": "src-a",
            "split_id": "split-a",
            "split_idx": 2,
            "score": 0.90,
            "content": "A2 邻近段",
            "parent_id": "a1",
            "source_type": "guideline",
        },
        {
            "id": "a3",
            "doc_id": "doc-a",
            "chunk_id": "a3",
            "source_id": "src-a",
            "split_id": "split-a",
            "split_idx": 3,
            "score": 0.82,
            "content": "A3 远端段",
            "source_type": "guideline",
        },
        {
            "id": "b1",
            "doc_id": "doc-b",
            "chunk_id": "b1",
            "source_id": "src-b",
            "split_id": "split-b",
            "split_idx": 1,
            "score": 0.88,
            "content": "B1 其他来源",
            "source_type": "paper",
        },
    ]

    pack = adapter.build_context_pack(
        docs=docs,
        fusion_method="weighted_rrf",
        fallback_context="",
        ordering_strategy="lost_in_middle_mitigate",
    )

    assert pack["ordering"] == "lost_in_middle_mitigate"
    assert pack["fusion_method"] == "weighted_rrf"
    assert isinstance(pack["evidence"], list)
    assert len(pack["evidence"]) <= 4
    source_ids = {item.get("source_id") for item in pack["evidence"] if isinstance(item, dict)}
    assert "src-a" in source_ids and "src-b" in source_ids
    assert any("A1 主段" in item.get("content", "") and "A2 邻近段" in item.get("content", "") for item in pack["evidence"])


@pytest.mark.asyncio
async def test_json_schema_guardrail_repair_success():
    guardrail = JsonSchemaGuardrail(schema_version="v1", enabled=True)

    invalid_payload = {
        "diagnosis_schema_version": "",
        "department_top1": "消化内科",
        "department_top3": "消化内科",
        "confidence": 1.2,
        "reasoning": "",
        "citations": "bad",
    }

    result = await guardrail.validate_and_repair(invalid_payload)

    assert result["validated"] is True
    assert result["repair_attempted"] is True
    output = result["diagnosis_output"]
    assert output["diagnosis_schema_version"] == "v1"
    assert isinstance(output["department_top3"], list)
    assert isinstance(output["citations"], list)


@pytest.mark.asyncio
async def test_json_schema_guardrail_safe_fallback_when_repair_invalid():
    guardrail = JsonSchemaGuardrail(schema_version="v1", enabled=True)

    invalid_payload = {
        "department_top1": "",
        "department_top3": "bad",
        "confidence": "high",
        "reasoning": 123,
        "citations": "bad",
    }

    async def broken_repair(_):
        return {"foo": "bar"}

    result = await guardrail.validate_and_repair(invalid_payload, repair_fn=broken_repair)

    assert result["repair_attempted"] is True
    assert result["validated"] is False
    output = result["diagnosis_output"]
    assert output["department_top1"] == "Unknown"
    assert output["diagnosis_schema_version"] == "v1"
    assert output["citations"] == []
