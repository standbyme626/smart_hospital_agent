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
from .config import get_settings
from .db import Database
from .eval_runner import EvalRunner
from .mcp_client import HospitalMCPClient
from .model_gateway import ModelGateway
from .retrieval import RetrievalService
from .routes import router
from .security import bootstrap_users
from .session_store import SessionStore
from .tools import ToolRegistry

REQUEST_COUNT = Counter("sha_http_requests_total", "HTTP requests", ["method", "path", "status"])
REQUEST_LATENCY = Histogram("sha_http_request_duration_seconds", "HTTP request latency", ["method", "path"])
LLM_CALLS = Counter("sha_llm_calls_total", "LLM calls", ["provider", "task"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings
    db = Database(settings)
    try:
        await db.init_schema()
        await bootstrap_users(settings, db)
    except Exception as exc:  # keep the app bootable for diagnostics
        print(f"[startup] database init failed: {exc}")
    gateway = ModelGateway(settings)
    retrieval = RetrievalService(settings, db, gateway)
    mcp = HospitalMCPClient(settings)
    tools = ToolRegistry(retrieval, mcp, db)
    sessions = SessionStore(db)
    agent = MedicalAgent(settings, gateway, retrieval, tools, sessions)
    eval_runner = EvalRunner(settings, db, retrieval, gateway)
    app.state.db = db
    app.state.gateway = gateway
    app.state.retrieval = retrieval
    app.state.mcp = mcp
    app.state.tools = tools
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
    version="2.0.0",
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
    path = request.url.path
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
    finally:
        elapsed = time.perf_counter() - start
        REQUEST_COUNT.labels(request.method, path, str(status)).inc()
        REQUEST_LATENCY.labels(request.method, path).observe(elapsed)
    response.headers["X-Request-ID"] = request_id
    return response


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    settings = request.app.state.settings
    if request.url.path in {"/health", "/api/health", "/metrics", "/docs", "/openapi.json"}:
        return await call_next(request)
    forwarded = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
    client = request.headers.get("x-real-ip") or forwarded or (request.client.host if request.client else "unknown")
    now = time.time()
    async with _rate_lock:
        bucket = _rate_buckets[client]
        while bucket and now - bucket[0] > 60:
            bucket.popleft()
        if len(bucket) >= settings.RATE_LIMIT_PER_MINUTE:
            return JSONResponse(status_code=429, content={"detail": "rate limit exceeded"})
        bucket.append(now)
    return await call_next(request)


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


@app.get("/")
async def root() -> dict:
    return {
        "app": "smart-hospital-agent",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
        "chat": "/api/v1/chat/stream",
    }
