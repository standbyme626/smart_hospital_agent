import time
import uuid
import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from app.core.config import settings

logger = structlog.get_logger("request_logger")

class GlobalExceptionHandlerMiddleware(BaseHTTPMiddleware):
    """
    全局异常捕获中间件 (Global Exception Handler Middleware)
    1. 捕获所有未处理的异常 (Uncaught Exceptions)。
    2. 记录详细的结构化错误日志 (Stack Trace, Request Context)。
    3. 返回标准化的错误响应，防止敏感信息泄露。
    """
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        request_id = (
            request.headers.get("X-Request-ID")
            or getattr(getattr(request, "state", object()), "request_id", "")
            or str(uuid.uuid4())
        )
        request.state.request_id = request_id
        
        # 将 Request ID 绑定到当前 Context
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)
        
        try:
            response = await call_next(request)
            
            # 记录请求耗时 (可选，仅在非 200 或慢请求时记录)
            process_time = time.time() - start_time
            if response.status_code >= 400:
                logger.warning(
                    "request_failed",
                    method=request.method,
                    path=request.url.path,
                    status_code=response.status_code,
                    duration_s=process_time
                )
            
            return response
            
        except Exception as exc:
            process_time = time.time() - start_time
            
            # 捕获异常并记录完整日志
            logger.error(
                "uncaught_exception",
                error=str(exc),
                error_type=type(exc).__name__,
                method=request.method,
                path=request.url.path,
                duration_s=process_time,
                exc_info=True # 包含堆栈跟踪
            )
            
            # 返回友好错误响应
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal Server Error",
                    "message": "An unexpected error occurred. Please contact support.",
                    "request_id": request_id
                }
            )

class RequestLogMiddleware(BaseHTTPMiddleware):
    """
    请求日志中间件 (Request Logging Middleware)
    确保每个请求都有 Request ID，并绑定到 Structlog 上下文。
    """
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        structlog.contextvars.bind_contextvars(request_id=request_id)
        
        response = await call_next(request)
        
        response.headers["X-Request-ID"] = request_id
        return response
