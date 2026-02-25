"""
Metrics 导出端点

提供 Prometheus 抓取的 HTTP 接口
"""

from fastapi import APIRouter
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

router = APIRouter()


@router.get("/metrics")
async def metrics():
    """
    Prometheus metrics 端点
    
    访问: GET /metrics
    返回: Prometheus 文本格式的指标数据
    """
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )
