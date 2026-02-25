from prometheus_client import Counter, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import time

# 定义指标
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "handler", "status"]
)

REQUEST_TIME = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency",
    ["method", "handler"]
)

# RAG 业务指标
RAG_RETRIEVAL_LATENCY = Histogram(
    "rag_retrieval_latency_seconds",
    "Time spent in RAG retrieval pipeline",
    ["stage"]  # e.g., "milvus", "bm25", "rerank", "total"
)

RAG_CACHE_HIT = Counter(
    "rag_cache_hits_total",
    "Total number of RAG cache hits",
    ["cache_type"] # e.g., "semantic", "exact"
)

RAG_CACHE_MISS = Counter(
    "rag_cache_misses_total",
    "Total number of RAG cache misses",
    ["cache_type"]
)

class PrometheusMiddleware(BaseHTTPMiddleware):
    """
    Prometheus 监控中间件 (Prometheus Monitoring Middleware)
    
    自动采集 HTTP 请求的黄金指标：
    - 吞吐量 (Requests per second)
    - 延迟 (Latency)
    - 错误率 (Error rate, via status codes)
    """
    async def dispatch(self, request, call_next):
        """
        拦截请求并记录指标
        """
        method = request.method
        handler = request.url.path
        
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        
        status = str(response.status_code)
        
        # 记录指标
        REQUEST_COUNT.labels(method=method, handler=handler, status=status).inc()
        REQUEST_TIME.labels(method=method, handler=handler).observe(process_time)
        
        return response

def metrics(request):
    """
    暴露 Prometheus 指标端点 (/metrics)
    供 Prometheus Server 抓取。
    """
    return Response(generate_latest(), media_type="text/plain")
