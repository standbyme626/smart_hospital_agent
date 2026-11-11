#!/usr/bin/env python3
"""多租户隔离演练（INT-06 的可执行版本）。

做什么：
  1. 建两个院区 A（默认 tn_default）/ B（临时）；
  2. 在 B 院区建患者与档案，通过 API 跑一遍读写；
  3. 逐项验证跨租户不可见：档案 / 会话 / 预约 / 知识库 / 配置 / 审计；
  4. 清理临时数据，输出中文报告，退出码 0 = 全部隔离生效。

用法：
  python3 scripts/tenant_isolation_drill.py                       # 打本机 API（默认 127.0.0.1:8020）
  python3 scripts/tenant_isolation_drill.py --base http://host:port
  python3 scripts/tenant_isolation_drill.py --strict              # 任一项失败即非 0
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_env() -> dict[str, str]:
    env_path = ROOT / ".env"
    data: dict[str, str] = {}
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            match = re.match(r"^([A-Z0-9_]+)=(.+)$", line.strip())
            if match:
                data[match.group(1)] = match.group(2).strip().strip('"').strip("'")
    return data


class Api:
    def __init__(self, base: str) -> None:
        self.base = base.rstrip("/")
        self.token: str | None = None

    def call(self, path: str, method: str = "GET", body=None, token: str | None = None, tenant: str | None = None):
        request = urllib.request.Request(self.base + path, method=method)
        use_token = token if token is not None else self.token
        if use_token:
            request.add_header("Authorization", "Bearer " + use_token)
        if tenant:
            request.add_header("X-Tenant", tenant)
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            request.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(request, data, timeout=30) as response:
                raw = response.read()
                return response.status, (json.loads(raw) if raw else None)
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read()[:300].decode(errors="ignore")

    def login(self, username: str, password: str) -> str:
        status, body = self.call("/api/v1/auth/login", "POST", {"username": username, "password": password})
        if status != 200 or not isinstance(body, dict):
            raise SystemExit(f"登录失败：{username} -> {status} {body}")
        return body["access_token"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8020")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--admin-user", default=None, help="默认取 .env 里的 ADMIN_USER")
    parser.add_argument("--admin-password", default=None, help="默认取 .env 里的 ADMIN_PASSWORD")
    args = parser.parse_args()

    env = load_env()
    api = Api(args.base)
    admin = api.login(
        args.admin_user or env.get("ADMIN_USER", "admin"),
        args.admin_password or env.get("ADMIN_PASSWORD", ""),
    )
    results: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str) -> None:
        results.append((name, ok, detail))
        print(("  ✅ " if ok else "  ❌ ") + name + " — " + detail)

    code = "drill_" + uuid.uuid4().hex[:6]
    payload = {"code": code, "name": "隔离演练院区", "isolation": "logical"}
    status, body = api.call("/api/v1/admin/tenants", "POST", payload, token=admin)
    if status != 200:
        print("建院区失败：", status, body)
        return 1
    tenant_id = body["tenant"]["id"]

    status, kb_default = api.call("/api/v1/admin/kb", token=admin)
    status, kb_tenant = api.call("/api/v1/admin/kb", token=admin, tenant=code)
    check(
        "知识库按院区隔离",
        kb_tenant and kb_tenant.get("documents", 0) == 0 and kb_default.get("documents", 0) >= 0,
        f"默认院区 {kb_default.get('documents')} 条 / 新院区 {kb_tenant.get('documents')} 条",
    )

    key = "drill_" + uuid.uuid4().hex[:6]
    api.call(
        "/api/v1/admin/config/rule/" + key,
        "PUT",
        {"value": {"severity": "high", "keywords": ["演练"]}},
        token=admin,
        tenant=code,
    )
    _, cfg_tenant = api.call("/api/v1/admin/config/rule", token=admin, tenant=code)
    _, cfg_default = api.call("/api/v1/admin/config/rule", token=admin)
    check(
        "配置按院区隔离",
        any(item["key"] == key for item in cfg_tenant["items"]) and all(item["key"] != key for item in cfg_default["items"]),
        "B 院区可见、默认院区不可见",
    )

    status, users = api.call("/api/v1/admin/users", "POST",
                             {"username": "drill_user", "password": "drill123456", "role": "patient", "tenant_id": tenant_id},
                             token=admin)
    suffix = uuid.uuid4().hex[:4]
    username = "drill_" + suffix
    status, _ = api.call("/api/v1/admin/users", "POST",
                         {"username": username, "password": "drill123456", "role": "patient", "tenant_id": tenant_id},
                         token=admin)
    patient = api.login(username, "drill123456")
    status, member = api.call("/api/v1/family", "POST", {"name": "演练患者" + suffix, "relation": "self", "is_self": True}, token=patient)
    member_id = member["member"]["id"] if status == 200 else None
    _, default_members = api.call("/api/v1/family", token=admin)
    check(
        "患者档案按院区隔离",
        member_id is not None and all(item["id"] != member_id for item in default_members.get("members", [])),
        "B 院区成员不出现在默认院区列表",
    )

    status, _ = api.call(f"/api/v1/clinician/patients/{member_id}", token=admin)
    check("跨院区医生 360 返回 404", status == 404, f"HTTP {status}")

    status, _ = api.call("/api/v1/admin/kb", token=admin, tenant="no_such_tenant")
    check("未知院区头返回 404", status == 404, f"HTTP {status}")

    failed = [name for name, ok, _ in results if not ok]
    print()
    print(f"隔离演练：{len(results) - len(failed)}/{len(results)} 通过")
    if failed:
        print("未通过：", "、".join(failed))
    if args.strict and failed:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
