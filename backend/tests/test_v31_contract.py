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
    assert "requestPayment" in source


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


API_TS = ROOT / "frontend" / "lib" / "api.ts"
API_PATH_RE = re.compile(r"/api/v1/[^\"'`\s]+")
TEMPLATE_RE = re.compile(r"(\$\{[^}]*\})")


def _frontend_api_paths() -> list[str]:
    text = API_TS.read_text(encoding="utf-8")
    paths = set()
    for raw in API_PATH_RE.findall(text):
        path = raw.split("?", 1)[0].rstrip(".,;")
        path = re.sub(r"(?<!/)\$\{[^}]*\}", "", path)
        if path:
            paths.add(path)
    return sorted(paths)


def _call_pattern(path: str) -> re.Pattern[str]:
    parts = TEMPLATE_RE.split(path)
    pattern = "".join("[^/]+" if TEMPLATE_RE.fullmatch(part) else re.escape(part) for part in parts)
    return re.compile(pattern)


def test_frontend_api_calls_all_match_openapi_paths():
    paths = _frontend_api_paths()
    assert len(paths) >= 30, paths
    openapi_paths = list(app.openapi()["paths"])
    missing = [
        path
        for path in paths
        if not any(_call_pattern(path).fullmatch(candidate) for candidate in openapi_paths)
    ]
    assert missing == [], f"frontend calls without backend route: {missing}"


def test_frontend_api_calls_cover_v31_flows():
    paths = set(_frontend_api_paths())
    expected = {
        "/api/v1/auth/login",
        "/api/v1/chat/stream",
        "/api/v1/sessions/{session_id}",
        "/api/v1/reports/interpret",
        "/api/v1/booking/lock",
        "/api/v1/booking/orders",
        "/api/v1/booking/orders/{order_id}/request-payment",
        "/api/v1/admin/reviews/{review_id}/approve",
        "/api/v1/family",
        "/api/v1/services/visit-guide",
    }
    for path in expected:
        assert any(_call_pattern(candidate).fullmatch(path) for candidate in paths), path
