"""管理端 Wave E 第二批：知识库 / 检索 / 未命中 / 模型 / 回放 / 审计 / 验收 / HIS / 多院区 / 实验。

覆盖：权限门禁、真实库读取形状、检索调试与 RAG 链路自检、未命中问题转知识条目、
HIS 连通性留痕、院区与实验的创建/校验/灰度规则。
"""
from __future__ import annotations

import asyncio
import os
import uuid

import asyncpg
import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app

ADMIN_PASSWORD = "admin123456"
PATIENT_PASSWORD = "patient123456"

ADMIN_GET_ENDPOINTS = (
    "/api/v1/admin/kb",
    "/api/v1/admin/rag/documents",
    "/api/v1/admin/unmatched",
    "/api/v1/admin/model-calls/summary",
    "/api/v1/admin/sessions",
    "/api/v1/admin/audit/aggregate",
    "/api/v1/admin/acceptance",
    "/api/v1/admin/his",
    "/api/v1/admin/tenants",
    "/api/v1/admin/experiments",
    "/api/v1/admin/flow",
    "/api/v1/admin/states",
)


class FakeRetrieval:
    """只替换检索，不碰数据库：验证接口契约与错误处理。"""

    last_filtered = True
    last_dept_unavailable = False

    def __init__(self, hits: list[dict] | None = None, fail: bool = False) -> None:
        self.hits = hits if hits is not None else [
            {
                "id": 1,
                "doc_key": "doc_htn",
                "title": "高血压诊疗指南",
                "snippet": "成人高血压定义为收缩压≥140mmHg",
                "department": "心内科",
                "source": "guide",
                "score": 0.91,
            }
        ]
        self.fail = fail
        self.ingested: list[dict] = []

    async def search(self, query: str, *, top_k: int | None = None, department: str | None = None):
        if self.fail:
            raise RuntimeError("embedding service unavailable")
        return self.hits

    async def ingest(self, rows: list[dict], batch_size: int = 8) -> int:
        self.ingested.extend(rows)
        return len(rows)


class FakeMCP:
    def __init__(self, ok: bool = True) -> None:
        self.ok = ok

    async def health(self) -> bool:
        return self.ok


@pytest.fixture()
def client():
    with TestClient(app) as test_client:
        yield test_client


def _sql(query: str, *args):
    async def run():
        conn = await asyncpg.connect(os.environ["DATABASE_URL"])
        try:
            return await conn.fetch(query, *args)
        finally:
            await conn.close()

    return asyncio.run(run())


@pytest.fixture(autouse=True)
def cleanup_admin_ext_rows():
    """本文件会往真库写样本行（文档 / 会话 / 模型调用 / 院区 / 实验），用完必须收干净，
    否则会污染检索类用例（tests/test_v31_retrieval.py 依赖 documents 的科室集合）。"""
    yield
    statements = (
        "DELETE FROM documents WHERE doc_key LIKE 'doc_ext_%' OR source='admin-faq'",
        "DELETE FROM unmatched_resolutions WHERE answer LIKE '%儿童多动症%'",
        "DELETE FROM unmatched_questions WHERE question LIKE '%儿童多动症%'",
        "DELETE FROM model_calls WHERE provider='test'",
        "DELETE FROM messages WHERE session_id IN (SELECT id FROM sessions WHERE title='回放测试会话')",
        "DELETE FROM sessions WHERE title='回放测试会话'",
        "DELETE FROM tenants WHERE code LIKE 't-%'",
        "DELETE FROM experiments WHERE key LIKE 'exp-%'",
    )
    for statement in statements:
        try:
            _sql(statement)
        except Exception:
            continue


def _token(client: TestClient, username: str, password: str) -> str:
    response = client.post("/api/v1/auth/login", json={"username": username, "password": password})
    assert response.status_code == 200, response.text
    return response.json()["access_token"]


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture()
def admin(client: TestClient) -> dict[str, str]:
    get_settings().AUTH_ENABLED = True
    return _auth(_token(client, "admin", ADMIN_PASSWORD))


def _uid() -> str:
    return uuid.uuid4().hex[:8]


# ------------------------------------------------------------------ 权限
def test_admin_ext_requires_admin(client: TestClient) -> None:
    get_settings().AUTH_ENABLED = True
    assert client.get("/api/v1/admin/acceptance").status_code == 401

    patient = _auth(_token(client, "patient", PATIENT_PASSWORD))
    for path in ADMIN_GET_ENDPOINTS:
        assert client.get(path, headers=patient).status_code == 403, path
    assert client.post("/api/v1/admin/his/check", headers=patient).status_code == 403
    assert (
        client.post(
            "/api/v1/admin/retrieval/debug", json={"query": "高血压"}, headers=patient
        ).status_code
        == 403
    )


# ------------------------------------------------------------------ 知识库 / 语料
def test_kb_overview(client: TestClient, admin: dict[str, str]) -> None:
    payload = client.get("/api/v1/admin/kb", headers=admin).json()
    assert payload["coverage"]["total"] >= 20, "科室目录口径应有 21 个科室"
    assert 0 <= payload["coverage"]["covered"] <= payload["coverage"]["total"]
    assert isinstance(payload["departments"], list)
    assert payload["documents"] >= 0
    assert "unmatched" in payload and "resolved" in payload


def test_rag_documents_list_and_filter(client: TestClient, admin: dict[str, str]) -> None:
    _sql(
        "INSERT INTO documents (doc_key, title, content, department, source) "
        "VALUES ($1, $2, $3, $4, $5) ON CONFLICT (doc_key) DO NOTHING",
        "doc_ext_" + _uid(),
        "接口测试文档",
        "内容",
        "心内科",
        "test",
    )
    listed = client.get("/api/v1/admin/rag/documents?limit=5", headers=admin)
    assert listed.status_code == 200
    assert listed.json()["total"] >= 1
    filtered = client.get("/api/v1/admin/rag/documents?department=心内科", headers=admin)
    assert filtered.status_code == 200
    assert all(item["department"] == "心内科" for item in filtered.json()["items"])


# ------------------------------------------------------------------ 检索调试 / RAG 链路
def test_retrieval_debug(client: TestClient, admin: dict[str, str], monkeypatch) -> None:
    monkeypatch.setattr(app.state, "retrieval", FakeRetrieval())

    response = client.post(
        "/api/v1/admin/retrieval/debug",
        json={"query": "高血压怎么分级", "department": "心内科", "top_k": 3},
        headers=admin,
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["total"] == 1
    assert payload["filtered"] is True
    assert payload["hits"][0]["doc_key"] == "doc_htn"
    assert payload["settings"]["top_k"] >= 0
    assert payload["latency_ms"] >= 0

    empty = client.post("/api/v1/admin/retrieval/debug", json={"query": "  "}, headers=admin)
    assert empty.status_code == 422


def test_pipeline_check_ok_and_error(client: TestClient, admin: dict[str, str], monkeypatch) -> None:
    monkeypatch.setattr(app.state, "retrieval", FakeRetrieval())
    ok = client.post("/api/v1/admin/rag/pipeline-check", json={"query": "高血压"}, headers=admin)
    assert ok.status_code == 200, ok.text
    stages = ok.json()["stages"]
    assert len(stages) == 4
    assert all(stage["ok"] for stage in stages)

    monkeypatch.setattr(app.state, "retrieval", FakeRetrieval(fail=True))
    broken = client.post("/api/v1/admin/rag/pipeline-check", json={"query": "高血压"}, headers=admin)
    assert broken.status_code == 200, "链路自检要把失败暴露成阶段失败，而不是 500"
    payload = broken.json()
    assert payload["error"]
    assert payload["hits"] == 0
    assert any(not stage["ok"] for stage in payload["stages"])


# ------------------------------------------------------------------ 未命中问题
def test_unmatched_list_and_resolve(client: TestClient, admin: dict[str, str], monkeypatch) -> None:
    question = "儿童多动症要不要挂心理科 " + _uid()
    _sql(
        "INSERT INTO unmatched_questions (question_key, question, department) VALUES ($1, $2, $3)",
        "qk_" + _uid(),
        question,
        "儿科",
    )
    listed = client.get("/api/v1/admin/unmatched?limit=50", headers=admin).json()
    target = next(item for item in listed["items"] if item["question"] == question)
    assert target["resolved"] is False
    assert target["hits"] >= 1

    fake = FakeRetrieval()
    monkeypatch.setattr(app.state, "retrieval", fake)
    resolved = client.post(
        "/api/v1/admin/unmatched/resolve",
        json={"question": question, "answer": "儿童多动症首诊可挂儿科或儿童精神科。", "department": "儿科"},
        headers=admin,
    )
    assert resolved.status_code == 200, resolved.text
    payload = resolved.json()
    assert payload["written"] == 1
    assert payload["doc_key"].startswith("admin-faq-")
    assert fake.ingested and fake.ingested[0]["source"] == "admin-faq"

    hidden = client.get("/api/v1/admin/unmatched?limit=50", headers=admin).json()
    assert all(item["question"] != question for item in hidden["items"]), "已处理的问题默认不再出现在待处理列表"

    after = client.get("/api/v1/admin/unmatched?limit=50&only_open=false", headers=admin).json()
    assert next(item for item in after["items"] if item["question"] == question)["resolved"] is True

    bad = client.post(
        "/api/v1/admin/unmatched/resolve",
        json={"question": question, "answer": "  "},
        headers=admin,
    )
    assert bad.status_code == 422


# ------------------------------------------------------------------ 模型 / 会话 / 审计
def test_model_call_summary(client: TestClient, admin: dict[str, str]) -> None:
    _sql(
        "INSERT INTO model_calls (provider, model, task, prompt_tokens, completion_tokens, cost) "
        "VALUES ('test', 'model-a', 'triage', 10, 5, 0.001)",
    )
    payload = client.get("/api/v1/admin/model-calls/summary?days=7", headers=admin).json()
    assert payload["totals"]["calls"] >= 1
    assert payload["items"], "应至少有一条聚合记录"
    assert {"provider", "model", "task", "calls", "cost"} <= set(payload["items"][0])


def test_session_trace(client: TestClient, admin: dict[str, str]) -> None:
    session_id = "sess_" + _uid()
    _sql(
        "INSERT INTO sessions (id, user_id, title) VALUES ($1, 'tester', '回放测试会话')",
        session_id,
    )
    _sql(
        "INSERT INTO messages (session_id, user_id, role, content) VALUES ($1, 'tester', 'user', '头痛三天'), ($1, 'tester', 'assistant', '建议神经内科就诊')",
        session_id,
    )
    listed = client.get("/api/v1/admin/sessions?limit=10", headers=admin).json()
    assert any(item["id"] == session_id for item in listed["items"])

    trace = client.get(f"/api/v1/admin/sessions/{session_id}/trace", headers=admin)
    assert trace.status_code == 200, trace.text
    payload = trace.json()
    assert payload["session"]["id"] == session_id
    assert len(payload["messages"]) == 2
    assert payload["messages"][0]["role"] == "user"

    assert client.get("/api/v1/admin/sessions/no-such-session/trace", headers=admin).status_code == 404


def test_audit_aggregate(client: TestClient, admin: dict[str, str]) -> None:
    payload = client.get("/api/v1/admin/audit/aggregate?days=7", headers=admin).json()
    for key in ("total", "by_action", "by_actor", "recent", "config_changes"):
        assert key in payload
    assert isinstance(payload["by_action"], list)


def test_acceptance_board(client: TestClient, admin: dict[str, str]) -> None:
    payload = client.get("/api/v1/admin/acceptance", headers=admin).json()
    assert len(payload["playbooks"]) == 25
    assert payload["playbook_summary"] == {"ready": 24, "partial": 0, "blocked": 1}
    assert payload["checks"], "自检项不能为空"
    assert payload["summary"]["total"] == len(payload["checks"])
    names = {check["name"] for check in payload["checks"]}
    assert {"数据库连接", "业务表数量", "鉴权开关"} <= names
    assert all(check["status"] in {"ok", "warn", "fail"} for check in payload["checks"])


# ------------------------------------------------------------------ HIS
def test_his_overview_and_check(client: TestClient, admin: dict[str, str], monkeypatch) -> None:
    overview = client.get("/api/v1/admin/his", headers=admin)
    assert overview.status_code == 200, overview.text
    payload = overview.json()
    assert "mcp_url" in payload["config"]
    assert payload["capabilities"], "能力矩阵不能为空"

    monkeypatch.setattr(app.state, "mcp", FakeMCP(ok=True))
    checked = client.post("/api/v1/admin/his/check", headers=admin)
    assert checked.status_code == 200, checked.text
    assert checked.json()["ok"] is True
    assert checked.json()["latency_ms"] >= 0

    history = client.get("/api/v1/admin/his", headers=admin).json()["checks"]
    assert history and history[0]["ok"] is True


# ------------------------------------------------------------------ 多院区 / 实验
def test_tenants_crud(client: TestClient, admin: dict[str, str]) -> None:
    code = "t-" + _uid()
    created = client.post(
        "/api/v1/admin/tenants",
        json={"code": code, "name": "测试院区", "isolation": "physical"},
        headers=admin,
    )
    assert created.status_code == 200, created.text
    tenant_id = created.json()["tenant"]["id"]

    listed = client.get("/api/v1/admin/tenants", headers=admin).json()
    assert any(item["code"] == "default" for item in listed["items"]), "默认租户应存在"
    assert any(item["id"] == tenant_id for item in listed["items"])

    updated = client.patch(
        f"/api/v1/admin/tenants/{tenant_id}", json={"status": "suspended"}, headers=admin
    )
    assert updated.status_code == 200
    assert updated.json()["tenant"]["status"] == "suspended"

    assert (
        client.post("/api/v1/admin/tenants", json={"code": code, "name": "重复"}, headers=admin).status_code
        == 409
    )
    assert (
        client.post(
            "/api/v1/admin/tenants", json={"code": "bad code!", "name": "非法"}, headers=admin
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/api/v1/admin/tenants",
            json={"code": "x-" + _uid(), "name": "隔离模式错", "isolation": "magic"},
            headers=admin,
        ).status_code
        == 422
    )
    assert client.patch("/api/v1/admin/tenants/nope", json={"status": "active"}, headers=admin).status_code == 404


def test_experiments_crud_and_rollout_rules(client: TestClient, admin: dict[str, str]) -> None:
    key = "exp-" + _uid()
    created = client.post(
        "/api/v1/admin/experiments",
        json={"key": key, "name": "提示词灰度", "variants": ["control", "v2"]},
        headers=admin,
    )
    assert created.status_code == 200, created.text
    assert created.json()["experiment"]["status"] == "draft"

    running = client.patch(
        f"/api/v1/admin/experiments/{key}", json={"status": "running", "rollout": 20}, headers=admin
    )
    assert running.status_code == 200, running.text
    assert running.json()["experiment"]["rollout"] == 20
    assert running.json()["experiment"]["started_at"], "进入运行态要记录开始时间"

    no_rollout = client.patch(
        f"/api/v1/admin/experiments/{key}", json={"status": "running", "rollout": 0}, headers=admin
    )
    assert no_rollout.status_code == 422, "运行态灰度比例必须大于 0"

    too_big = client.patch(f"/api/v1/admin/experiments/{key}", json={"rollout": 120}, headers=admin)
    assert too_big.status_code == 422

    bad_variants = client.patch(
        f"/api/v1/admin/experiments/{key}", json={"variants": ["only-one"]}, headers=admin
    )
    assert bad_variants.status_code == 422

    listed = client.get("/api/v1/admin/experiments", headers=admin).json()
    assert any(item["key"] == key for item in listed["items"])
    assert "statuses" in listed

    assignments = client.get(f"/api/v1/admin/experiments/{key}/assignments", headers=admin)
    assert assignments.status_code == 200
    assert assignments.json()["by_variant"] == []

    assert (
        client.post(
            "/api/v1/admin/experiments", json={"key": key, "name": "重复"}, headers=admin
        ).status_code
        == 409
    )


def test_flow_and_states(client: TestClient, admin: dict[str, str]) -> None:
    flow = client.get("/api/v1/admin/flow", headers=admin).json()
    assert len(flow["stages"]) >= 8
    assert all(stage["count"] >= 0 for stage in flow["stages"])

    states = client.get("/api/v1/admin/states", headers=admin).json()
    assert len(states["items"]) >= 8
    for item in states["items"]:
        assert "label" in item and "table" in item and "column" in item
        assert isinstance(item["values"], list)
