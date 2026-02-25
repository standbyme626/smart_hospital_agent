from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.core.config import settings
from app.api.v1.api import api_router
from app.core.logging.setup import setup_logging
from app.core.middleware.error_handler import RequestLogMiddleware, GlobalExceptionHandlerMiddleware
import structlog
import logging
import sys

# 1. 初始化全局日志系统 (Setup Global Logging)
setup_logging()
logger = structlog.get_logger()

from app.core.infra import lifespan

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan
)

# 2. 注册中间件 (Register Middlewares)
# 注意：中间件执行顺序为 洋葱模型，后注册的先执行 (Last Added, First Executed)
# 顺序：Prometheus -> CORS -> ExceptionHandler -> RequestLog -> ...

# [P4] 注册全局异常捕获中间件 (最外层，捕获所有未处理异常)
app.add_middleware(GlobalExceptionHandlerMiddleware)

# [P4] 注册请求日志中间件 (绑定 Request ID)
app.add_middleware(RequestLogMiddleware)

# 注册 Prometheus 监控中间件
from app.core.middleware.instrumentation import PrometheusMiddleware, metrics
app.add_middleware(PrometheusMiddleware)
app.add_route("/metrics", metrics)  # 暴露 /metrics 供 Prometheus 拉取

# [P4] 注册特定异常处理器 (Handle Specific Exceptions)
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from app.core.exceptions import AppException
from app.core.handlers import (
    app_exception_handler,
    validation_exception_handler,
    http_exception_handler,
    # global_exception_handler # 已由 GlobalExceptionHandlerMiddleware 接管
)

app.add_exception_handler(AppException, app_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(StarletteHTTPException, http_exception_handler)

# CORS Pattern (Allow all for Dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.API_V1_STR)

# Mount static files
import os
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/admin", StaticFiles(directory=static_dir, html=True), name="admin")

@app.get("/health")
async def health_check():
    """
    健康检查接口 (Health Check)
    用于 Docker/K8s 存活探针检查，确认服务是否正常运行。
    """
    import time
    logger.info("health_check_called", status="ok")
    return {
        "status": "ok", 
        "project": settings.PROJECT_NAME,
        "timestamp": time.time()
    }

@app.get("/")
async def root():
    """
    根路由 (Root Endpoint)
    用于验证 API 服务是否成功启动。
    """
    logger.info("root_accessed")
    return {"message": "欢迎使用智能医院 Agent API (Welcome to Smart Hospital Agent API)"}

# 移除过时的 shutdown_event，统一由 lifespan 管理

# 全局生命周期已由 lifespan 统一管理

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
