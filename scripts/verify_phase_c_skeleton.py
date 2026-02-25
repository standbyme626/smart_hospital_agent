#!/usr/bin/env python3
import asyncio
import json

from app.core.stream_schema import build_stream_payload
from app.rag.hierarchical_index import hierarchical_index_gateway
from app.rag.retrieval_planner import build_retrieval_plan


async def main() -> None:
    plan = await build_retrieval_plan("我胃疼还拉肚子两天了怎么办", intent="MEDICAL_CONSULT")
    print("=== retrieval_plan ===")
    print(json.dumps(plan.to_state_dict(), ensure_ascii=False, indent=2))

    print("\n=== hierarchical_index_placeholder ===")
    hits = await hierarchical_index_gateway.search(
        query=plan.primary_query,
        level=plan.index_scope,
        top_k=plan.top_k,
        force_enable=True,
    )
    print(json.dumps({"hit_count": len(hits), "hits": hits}, ensure_ascii=False, indent=2))

    print("\n=== stream_schema_legacy ===")
    legacy = build_stream_payload(
        event_type="token",
        content="示例输出",
        session_id="phase-c-demo",
        node="diagnosis_node",
        force_unified=False,
    )
    print(json.dumps(legacy, ensure_ascii=False, indent=2))

    print("\n=== stream_schema_unified ===")
    unified = build_stream_payload(
        event_type="token",
        content="示例输出",
        session_id="phase-c-demo",
        node="diagnosis_node",
        force_unified=True,
    )
    print(json.dumps(unified, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
