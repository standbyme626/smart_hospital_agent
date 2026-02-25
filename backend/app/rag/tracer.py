import logging
import json
from datetime import datetime
import os

# 简单的文件日志 Tracer
from app.core.config import settings

# [Modernization] 使用集中配置的 TRACE_LOG_PATH
TRACE_LOG_PATH = settings.TRACE_LOG_PATH
# os.makedirs(os.path.dirname(TRACE_LOG_PATH), exist_ok=True) # 已由 settings.LOG_DIR 自动处理

class RAGTracer:
    """
    RAG 链路追踪器
    记录每次检索的详细信息，用于评估和调试
    """
    def __init__(self):
        self.logger = logging.getLogger("rag_tracer")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.FileHandler(TRACE_LOG_PATH)
            formatter = logging.Formatter('%(asctime)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def trace(self, query: str, final_results: list, latencies: dict, details: dict = None):
        """
        记录一次检索的 Trace
        """
        trace_data = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "latencies_ms": latencies,
            "details": details or {},
            "result_count": len(final_results),
            "top_result_score": final_results[0]["score"] if final_results else 0.0
        }
        self.logger.info(json.dumps(trace_data, ensure_ascii=False))

_tracer = None

def get_tracer():
    global _tracer
    if _tracer is None:
        _tracer = RAGTracer()
    return _tracer
