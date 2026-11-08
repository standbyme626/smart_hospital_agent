import re
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app

ROOT = Path(__file__).resolve().parents[2]
OLD_DEMO_PASSWORD = "wasd" + "1234"

REMOVED_ROUTES = {
    "/api/v1/admin/rules",
    "/api/v1/admin/rules/{rule_id}",
    "/api/v1/evolution/run",
    "/api/v1/evolution/results",
    "/api/v1/doctor/consult",
    "/api/v1/local/analyze",
    "/api/v1/rag/search",
}


def _scan_files() -> list[Path]:
    files: list[Path] = [ROOT / ".env.example", ROOT / "docs" / "DEPLOYMENT-106.md"]
    for folder, suffix in (
        ("frontend/app", "*.tsx"),
        ("frontend/components", "*.tsx"),
        ("frontend/lib", "*.ts"),
    ):
        files.extend((ROOT / folder).rglob(suffix))
    build_out = ROOT / "frontend" / "out"
    if build_out.exists():
        files.extend(path for path in build_out.rglob("*") if path.is_file())
    return files


def test_no_old_demo_password_in_build_or_env_example_or_deployment_doc():
    offenders = []
    for path in _scan_files():
        if OLD_DEMO_PASSWORD in path.read_text(encoding="utf-8", errors="ignore"):
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == [], f"old demo password found in: {offenders}"


def test_env_example_uses_placeholders():
    env = (ROOT / ".env.example").read_text()
    for key in ("ADMIN_PASSWORD", "PATIENT_PASSWORD", "HITL_CONFIRM_TOKEN"):
        assert f"{key}=change-me" in env
    assert "postgresql://sha:change-me@" in env


def test_frontend_services_has_no_payment_token():
    source = (ROOT / "frontend" / "app" / "services" / "page.tsx").read_text()
    assert "confirmToken" not in source
    assert "confirm_token" not in source
    assert "payOrder" not in source
    assert re.search(r'<button[^>]*\bdisabled\b[^>]*>\s*<CreditCard', source)


def test_removed_endpoints_not_in_openapi():
    paths = set(app.openapi()["paths"])
    assert REMOVED_ROUTES.isdisjoint(paths)


def test_removed_endpoints_return_404():
    with TestClient(app) as client:
        assert client.get("/api/v1/admin/rules").status_code == 404
        assert client.post("/api/v1/rag/search", json={"query": "x"}).status_code == 404
        assert client.post("/api/v1/local/analyze", json={"text": "x"}).status_code == 404
        assert client.post("/api/v1/doctor/consult", json={"message": "x"}).status_code == 404
        assert client.post("/api/v1/evolution/run", json={"name": "x", "limit": 10}).status_code == 404
