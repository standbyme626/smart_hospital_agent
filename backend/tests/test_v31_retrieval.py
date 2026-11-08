import json

import pytest

from app.config import get_settings
from app.db import Database
from app.policy import DEPARTMENT_NAMES, normalize_department
from app.retrieval import RetrievalService
from app.tools import ToolContext, ToolRegistry

SOURCE = "v31_test"
UNAVAILABLE_DEPARTMENT = "康复医学科"


class FakeGateway:
    async def embed(self, texts: list[str]) -> list[list[float]]:
        dim = get_settings().EMBEDDING_DIM
        vectors = []
        for text in texts:
            vector = [0.0] * dim
            for index, char in enumerate(text):
                vector[index % dim] += (ord(char) % 89) + 1.0
            vectors.append(vector)
        return vectors

    async def rerank(self, query: str, documents: list[str], top_n: int = 20) -> list[tuple[int, float]]:
        return [(index, 1.0 - index * 0.01) for index in range(min(top_n, len(documents)))]


async def _cleanup(db: Database) -> None:
    await db.execute("DELETE FROM documents WHERE source=$1", SOURCE)


@pytest.fixture()
async def retrieval_env():
    settings = get_settings()
    db = Database(settings)
    await db.init_schema()
    await _cleanup(db)
    service = RetrievalService(settings, db, FakeGateway())
    try:
        yield service, db
    finally:
        await _cleanup(db)
        await db.close()


def test_normalize_department_aliases():
    assert normalize_department("内科") == "全科"
    assert normalize_department("妇科") == "妇产科"
    assert normalize_department("中医") == "中医科"
    assert normalize_department("中药学") == "中医科"
    assert normalize_department("泌尿科") == "泌尿外科"
    assert normalize_department("心内科") == "心血管内科"
    assert normalize_department("心血管内科") == "心血管内科"
    assert normalize_department(None) == "全科"
    assert normalize_department("不存在的科室") == "全科"


async def test_seed_from_jsonl_normalizes_departments(retrieval_env, tmp_path):
    service, db = retrieval_env
    rows = [
        {"question": "高血压如何管理", "answer": "限盐、规律服药与监测血压。", "cate1": "内科"},
        {"question": "孕期产检时间", "answer": "按孕周定期产检。", "cate1": "妇科"},
        {"question": "针灸适应症", "answer": "可用于慢性疼痛调理。", "cate1": "中医"},
        {"question": "前列腺增生", "answer": "尿频尿急需泌尿外科评估。", "cate1": "泌尿科"},
        {"question": "心悸如何评估", "answer": "心电图与动态心电图。", "cate1": "心内科"},
        {"question": "湿疹护理", "answer": "保湿与避免刺激。", "department": "皮肤科"},
    ]
    path = tmp_path / "docs.jsonl"
    path.write_text(
        "\n".join(json.dumps({**row, "source": SOURCE}, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )

    inserted = await service.seed_from_jsonl(path)

    assert inserted == len(rows)
    stored = await db.fetch("SELECT department FROM documents WHERE source=$1 ORDER BY id", SOURCE)
    assert [row["department"] for row in stored] == [
        "全科",
        "妇产科",
        "中医科",
        "泌尿外科",
        "心血管内科",
        "皮肤科",
    ]


async def _seed_canonical(service: RetrievalService, departments: list[str]) -> None:
    rows = [
        {
            "doc_key": f"v31test-{index:02d}",
            "title": f"{department} 常见问题",
            "content": f"{department} 的常见症状、检查与处理建议。",
            "department": department,
            "source": SOURCE,
            "metadata": {"department": department},
        }
        for index, department in enumerate(departments)
    ]
    await service.ingest(rows)


async def test_filtered_search_returns_department_docs(retrieval_env):
    service, _ = retrieval_env
    await _seed_canonical(service, ["心血管内科", "内分泌科"])

    results = await service.search("心悸", department="心血管内科")

    assert results
    assert service.last_filtered is True
    assert service.last_dept_unavailable is False
    assert {item["department"] for item in results} == {"心血管内科"}


async def test_empty_department_filter_falls_back(retrieval_env):
    service, _ = retrieval_env
    await _seed_canonical(service, ["心血管内科", "内分泌科"])

    results = await service.search("心悸", department=UNAVAILABLE_DEPARTMENT)

    assert results
    assert service.last_filtered is False
    assert service.last_dept_unavailable is True
    assert {item["department"] for item in results} == {"心血管内科", "内分泌科"}


async def test_every_canonical_department_non_empty_or_unavailable(retrieval_env):
    service, _ = retrieval_env
    seeded = [department for department in DEPARTMENT_NAMES if department != UNAVAILABLE_DEPARTMENT]
    await _seed_canonical(service, seeded)

    unavailable = []
    for department in DEPARTMENT_NAMES:
        results = await service.search("常见症状与就诊建议", department=department)
        assert results, f"{department} 检索结果为空"
        assert service.last_filtered or service.last_dept_unavailable, department
        if service.last_dept_unavailable:
            unavailable.append(department)

    assert unavailable == [UNAVAILABLE_DEPARTMENT]


async def test_retrieve_evidence_reports_filter_metadata(retrieval_env):
    service, db = retrieval_env
    await _seed_canonical(service, ["心血管内科"])
    tools = ToolRegistry(service, None, db)  # type: ignore[arg-type]

    hit = await tools.retrieve_evidence(
        {"query": "心悸", "department": "心血管内科"},
        ToolContext(session_id="s1", request_id="r1", user_text="心悸"),
    )
    assert hit["ok"] is True and hit["count"] >= 1
    assert hit["filtered"] is True and hit["dept_unavailable"] is False

    miss = await tools.retrieve_evidence(
        {"query": "心悸", "department": UNAVAILABLE_DEPARTMENT},
        ToolContext(session_id="s2", request_id="r2", user_text="心悸"),
    )
    assert miss["ok"] is True and miss["count"] >= 1
    assert miss["filtered"] is False and miss["dept_unavailable"] is True
