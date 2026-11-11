"""管理端配置注册表（a-rules / a-notify / a-prompt / a-agent 四屏共用后端）。

覆盖：权限门禁、草稿与发布分离、版本递增、回滚、停用、校验、模板预览、审计留痕。
"""
from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app

ADMIN_PASSWORD = "admin123456"
PATIENT_PASSWORD = "patient123456"


@pytest.fixture()
def client():
    with TestClient(app) as test_client:
        yield test_client


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


def _key(prefix: str = "k") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _rule(value: str = "critical") -> dict:
    return {"severity": value, "keywords": ["胸痛伴大汗"], "action": "立即 120"}


# ------------------------------------------------------------------ 权限
def test_config_requires_admin(client: TestClient) -> None:
    get_settings().AUTH_ENABLED = True

    anonymous = client.get("/api/v1/admin/config/kinds")
    assert anonymous.status_code == 401

    patient = _auth(_token(client, "patient", PATIENT_PASSWORD))
    assert client.get("/api/v1/admin/config/kinds", headers=patient).status_code == 403
    assert (
        client.put(
            f"/api/v1/admin/config/rule/{_key()}",
            json={"value": _rule()},
            headers=patient,
        ).status_code
        == 403
    )


def test_kinds_have_seeds(client: TestClient, admin: dict[str, str]) -> None:
    response = client.get("/api/v1/admin/config/kinds", headers=admin)
    assert response.status_code == 200, response.text
    items = {item["kind"]: item for item in response.json()["items"]}

    assert set(items) == {"rule", "notify_template", "prompt", "agent_config"}
    assert items["rule"]["name"] == "规则库"
    assert items["notify_template"]["name"] == "通知模板"
    for kind in ("rule", "notify_template", "prompt"):
        assert items[kind]["total"] >= 3, f"{kind} 缺少默认条目"


# ------------------------------------------------------- 草稿 / 发布 / 版本
def test_draft_publish_and_effective_value(client: TestClient, admin: dict[str, str]) -> None:
    key = _key("rule")
    url = f"/api/v1/admin/config/rule/{key}"

    saved = client.put(url, json={"value": _rule(), "note": "草稿"}, headers=admin)
    assert saved.status_code == 200, saved.text
    entry = saved.json()["entry"]
    assert entry["version"] == 0
    assert entry["published"] is None
    assert entry["effective"] is None, "未发布时不应有生效值"
    assert entry["draft"]["severity"] == "critical"

    published = client.post(f"{url}/publish", json={"note": "上线"}, headers=admin)
    assert published.status_code == 200, published.text
    entry = published.json()["entry"]
    assert entry["version"] == 1
    assert entry["effective"] == entry["published"]
    assert entry["dirty"] is False

    again = client.post(f"{url}/publish", json={}, headers=admin)
    assert again.status_code == 409, "草稿与已发布一致时应拒绝重复发布"


def test_rollback_restores_previous_version(client: TestClient, admin: dict[str, str]) -> None:
    key = _key("rule")
    url = f"/api/v1/admin/config/rule/{key}"

    client.put(url, json={"value": _rule("high")}, headers=admin)
    assert client.post(f"{url}/publish", json={}, headers=admin).json()["entry"]["version"] == 1

    client.put(url, json={"value": _rule("critical")}, headers=admin)
    v2 = client.post(f"{url}/publish", json={}, headers=admin).json()["entry"]
    assert v2["version"] == 2 and v2["effective"]["severity"] == "critical"

    rolled = client.post(f"{url}/rollback", json={"version": 1}, headers=admin)
    assert rolled.status_code == 200, rolled.text
    entry = rolled.json()["entry"]
    assert entry["version"] == 3, "回滚应产生新版本号，不覆盖历史"
    assert entry["effective"]["severity"] == "high"

    history = client.get(f"{url}/history", headers=admin).json()["items"]
    assert [item["version"] for item in history] == [3, 2, 1]
    assert history[0]["action"] == "rollback"

    missing = client.post(f"{url}/rollback", json={"version": 99}, headers=admin)
    assert missing.status_code == 404


def test_disable_hides_effective_value(client: TestClient, admin: dict[str, str]) -> None:
    key = _key("rule")
    url = f"/api/v1/admin/config/rule/{key}"
    client.put(url, json={"value": _rule()}, headers=admin)
    client.post(f"{url}/publish", json={}, headers=admin)

    off = client.post(f"{url}/enabled", json={"enabled": False}, headers=admin)
    assert off.status_code == 200, off.text
    assert off.json()["entry"]["enabled"] is False
    assert off.json()["entry"]["effective"] is None, "停用后页面应显示未启用，而不是沿用旧值"

    on = client.post(f"{url}/enabled", json={"enabled": True}, headers=admin)
    assert on.json()["entry"]["effective"] is not None


# ------------------------------------------------------------------ 校验
def test_validation_and_not_found(client: TestClient, admin: dict[str, str]) -> None:
    assert client.get("/api/v1/admin/config/unknown", headers=admin).status_code == 404

    missing_title = client.put(
        "/api/v1/admin/config/notify_template/notify_x",
        json={"value": {"body": "只有正文"}},
        headers=admin,
    )
    assert missing_title.status_code == 422

    empty_rule = client.put(
        f"/api/v1/admin/config/rule/{_key()}",
        json={"value": {"note": "没有力度字段"}},
        headers=admin,
    )
    assert empty_rule.status_code == 422

    bad_key = client.put("/api/v1/admin/config/rule/BAD-KEY", json={"value": _rule()}, headers=admin)
    assert bad_key.status_code == 422

    missing_prompt = client.put(
        f"/api/v1/admin/config/prompt/{_key('p')}",
        json={"value": {"label": "空提示词"}},
        headers=admin,
    )
    assert missing_prompt.status_code == 422

    not_found = client.get("/api/v1/admin/config/rule/never_created", headers=admin)
    assert not_found.status_code == 404


# ------------------------------------------------------------------ 模板预览
def test_notify_template_preview(client: TestClient, admin: dict[str, str]) -> None:
    key = _key("tpl")
    url = f"/api/v1/admin/config/notify_template/{key}"
    template = {
        "title": "{{member}}的报告解读完成",
        "body": "{{member}}的{{kind}}报告已完成，请在{{due}}前查看。",
        "channels": ["inbox"],
        "retry": {"max": 3, "backoff_s": 30},
    }
    client.put(url, json={"value": template}, headers=admin)
    client.post(f"{url}/publish", json={}, headers=admin)

    full = client.post(
        f"/api/v1/admin/config/notify_template/{key}/preview",
        json={"vars": {"member": "张三", "kind": "血常规", "due": "11-20"}},
        headers=admin,
    )
    assert full.status_code == 200, full.text
    payload = full.json()
    assert payload["title"] == "张三的报告解读完成"
    assert payload["body"] == "张三的血常规报告已完成，请在11-20前查看。"
    assert payload["missing_vars"] == []
    assert payload["source"] == "published"

    partial = client.post(
        f"/api/v1/admin/config/notify_template/{key}/preview",
        json={"vars": {"member": "张三"}},
        headers=admin,
    )
    assert set(partial.json()["missing_vars"]) == {"kind", "due"}

    assert (
        client.post(
            "/api/v1/admin/config/notify_template/absent_tpl/preview",
            json={"vars": {}},
            headers=admin,
        ).status_code
        == 404
    )


# ------------------------------------------------------------------ 审计
def test_publish_writes_audit_and_changes(client: TestClient, admin: dict[str, str]) -> None:
    key = _key("rule")
    url = f"/api/v1/admin/config/rule/{key}"
    client.put(url, json={"value": _rule()}, headers=admin)
    client.post(f"{url}/publish", json={"note": "审计验证"}, headers=admin)

    changes = client.get("/api/v1/admin/config/changes?limit=5", headers=admin)
    assert changes.status_code == 200, changes.text
    actions = [item["action"] for item in changes.json()["items"]]
    assert "publish" in actions

    audit = client.get("/api/v1/admin/audit", headers=admin)
    assert audit.status_code == 200, audit.text
    dumped = str(audit.json())
    assert "admin.config.publish" in dumped
    assert key in dumped
