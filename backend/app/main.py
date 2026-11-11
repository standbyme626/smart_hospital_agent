from __future__ import annotations

import asyncio
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram

from .agent import MedicalAgent
from .booking_service import BookingService
from .config import get_settings
from .db import Database
from .eval_runner import EvalRunner
from .mcp_client import HospitalMCPClient
from .model_gateway import ModelGateway
from .retrieval import RetrievalService
from .routes import router
from .routes_admin_config import router as admin_config_router
from .routes_admin_ext import router as admin_ext_router
from .routes_clinician_ext import router as clinician_ext_router
from .routes_panel import router as panel_router
from .schema_admin import ensure_admin_schema
from .schema_admin_ext import ensure_admin_ext_schema
from .schema_clinician_ext import ensure_clinician_ext_schema
from .security import bootstrap_users
from .security_hardening import (
    client_ip,
    deny_all_networks,
    ip_allowed,
    parse_networks,
    rate_tier,
    tier_limit,
)
from .session_store import SessionStore
from .tools import ToolRegistry

REQUEST_COUNT = Counter("sha_http_requests_total", "HTTP requests", ["method", "path", "status"])
REQUEST_LATENCY = Histogram("sha_http_request_duration_seconds", "HTTP request latency", ["method", "path"])
LLM_CALLS = Counter("sha_llm_calls_total", "LLM calls", ["provider", "task"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.validate_security()
    app.state.settings = settings
    db = Database(settings)
    try:
        await db.init_schema()
        await ensure_admin_schema(db)
        await ensure_admin_ext_schema(db)
        await ensure_clinician_ext_schema(db)
        await bootstrap_users(settings, db)
    except Exception as exc:  # keep the app bootable for diagnostics
        print(f"[startup] database init failed: {exc}")
    gateway = ModelGateway(settings)
    retrieval = RetrievalService(settings, db, gateway)
    mcp = HospitalMCPClient(settings)
    tools = ToolRegistry(retrieval, mcp, db)
    booking = BookingService(settings, db, mcp)
    sessions = SessionStore(db)
    agent = MedicalAgent(settings, gateway, retrieval, tools, sessions)
    eval_runner = EvalRunner(settings, db, retrieval, gateway)
    app.state.db = db
    app.state.gateway = gateway
    app.state.retrieval = retrieval
    app.state.mcp = mcp
    app.state.tools = tools
    app.state.booking = booking
    app.state.sessions = sessions
    app.state.agent = agent
    app.state.eval_runner = eval_runner
    try:
        yield
    finally:
        await gateway.close()
        await mcp.close()
        await db.close()


app = FastAPI(
    title="Smart Hospital Agent",
    version="3.1.0",
    description="Medical triage agent with hybrid retrieval, MCP booking/payment and SSE.",
    lifespan=lifespan,
)

_settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.cors_origins or ["http://localhost:3000"],
    allow_credentials="*" not in _settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

_rate_buckets: dict[str, deque[float]] = defaultdict(deque)
_rate_lock = asyncio.Lock()


@app.middleware("http")
async def observability(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    request.state.request_id = request_id
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
    finally:
        elapsed = time.perf_counter() - start
        route = request.scope.get("route")
        route_path = getattr(route, "path", None) or "__unmatched__"
        REQUEST_COUNT.labels(request.method, route_path, str(status)).inc()
        REQUEST_LATENCY.labels(request.method, route_path).observe(elapsed)
    response.headers["X-Request-ID"] = request_id
    return response


_allowlist_cache: dict[str, list] = {}


@app.middleware("http")
async def admin_ip_guard(request: Request, call_next):
    """管理接口入口白名单：ADMIN_IP_ALLOWLIST 留空时不生效（保持现状）。"""
    settings = request.app.state.settings
    raw = (getattr(settings, "ADMIN_IP_ALLOWLIST", "") or "").strip()
    if raw and request.url.path.startswith("/api/v1/admin"):
        networks = _allowlist_cache.get(raw)
        if networks is None:
            try:
                networks = parse_networks(raw)
            except ValueError:
                networks = deny_all_networks()  # 配置写错就全拒，避免"配错反而放开"
            _allowlist_cache[raw] = networks
        ip = client_ip(request, bool(getattr(settings, "TRUST_PROXY_HEADERS", True)))
        if not ip_allowed(ip, networks):
            return JSONResponse(
                status_code=403,
                content={
                    "detail": "管理接口来源不在白名单内",
                    "request_id": getattr(request.state, "request_id", ""),
                },
            )
    return await call_next(request)


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    settings = request.app.state.settings
    if request.url.path in {"/health", "/api/health", "/metrics", "/docs", "/openapi.json"}:
        return await call_next(request)
    client = client_ip(request, bool(getattr(settings, "TRUST_PROXY_HEADERS", True)))
    tier, bucket_name = rate_tier(request.url.path)
    limit = tier_limit(settings, tier)
    key = bucket_name + ":" + client
    now = time.time()
    async with _rate_lock:
        bucket = _rate_buckets[key]
        while bucket and now - bucket[0] > 60:
            bucket.popleft()
        if len(bucket) >= limit:
            return JSONResponse(
                status_code=429,
                content={"detail": "rate limit exceeded", "tier": tier, "limit_per_minute": limit},
            )
        bucket.append(now)
    return await call_next(request)


def reset_rate_buckets() -> None:
    """测试用：清空限流桶。"""
    _rate_buckets.clear()


@app.exception_handler(Exception)
async def unhandled_exception(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={
            "detail": "internal server error",
            "request_id": getattr(request.state, "request_id", ""),
        },
    )


app.include_router(router)
app.include_router(admin_config_router)
app.include_router(admin_ext_router)
app.include_router(clinician_ext_router)
app.include_router(panel_router)


@app.get("/")
async def root() -> dict:
    return {
        "app": "smart-hospital-agent",
        "version": "3.1.0",
        "docs": "/docs",
        "health": "/health",
        "chat": "/api/v1/chat/stream",
    }
