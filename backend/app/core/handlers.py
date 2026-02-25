from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import structlog

from app.core.exceptions import AppException

logger = structlog.get_logger(__name__)

async def app_exception_handler(request: Request, exc: AppException):
    """
    处理自定义业务异常
    """
    return JSONResponse(
        status_code=exc.code,
        content={
            "error": {
                "code": exc.code,
                "slug": exc.slug,
                "message": exc.msg,
                "details": exc.details
            }
        }
    )

async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    处理 Pydantic 校验异常 (422)
    """
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": 422,
                "slug": "validation_error",
                "message": "Input validation failed",
                "details": {"errors": exc.errors()}
            }
        }
    )

async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """
    处理 FastAPI/Starlette 内置 HTTP 异常 (404, 405 etc)
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.status_code,
                "slug": "http_error",
                "message": exc.detail,
                "details": {}
            }
        }
    )

async def global_exception_handler(request: Request, exc: Exception):
    """
    兜底捕获所有未处理异常 (500)
    防止堆栈泄露到前端，并记录详细日志
    """
    # 记录完整堆栈
    logger.error("unhandled_exception", error=str(exc), path=request.url.path, exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": 500,
                "slug": "internal_server_error",
                "message": "An unexpected error occurred. Please contact support.",
                "details": {"request_id": str(request.state.request_id)} if hasattr(request.state, "request_id") else {}
            }
        }
    )
