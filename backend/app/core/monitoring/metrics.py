
from prometheus_client import Counter, Histogram, Gauge

# --- Latency Metrics (Histogram) ---
EMBEDDING_LATENCY = Histogram(
    "embedding_latency_seconds",
    "Time spent computing embeddings",
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
)

RERANK_LATENCY = Histogram(
    "rerank_latency_seconds",
    "Time spent reranking documents",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
)

LLM_LATENCY = Histogram(
    "llm_latency_seconds",
    "Time spent generating LLM response (TTFT + Generation)",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

# --- Request Counters ---
EMBEDDING_REQUESTS = Counter(
    "embedding_requests_total",
    "Total number of embedding requests"
)

RERANK_REQUESTS = Counter(
    "rerank_requests_total",
    "Total number of rerank requests"
)

LLM_REQUESTS = Counter(
    "llm_requests_total",
    "Total number of LLM inference requests",
    ["model_name"]
)

# --- Cache Metrics ---
CACHE_HITS = Counter(
    "semantic_cache_hits_total",
    "Total number of semantic cache hits"
)

CACHE_MISSES = Counter(
    "semantic_cache_misses_total",
    "Total number of semantic cache misses"
)

# --- Resource Metrics (Gauge) ---
GPU_MEMORY_USAGE = Gauge(
    "gpu_memory_usage_ratio",
    "Current GPU memory usage ratio (0.0 - 1.0)"
)

MODEL_POOL_ACTIVE = Gauge(
    "model_pool_active_count",
    "Number of currently loaded models in GPU memory",
    ["model_type"]
)

MODEL_LOAD_EVENTS = Counter(
    "model_load_events_total",
    "Total number of model loading events (cold start/reload)",
    ["model_name"]
)

MODEL_OFFLOAD_EVENTS = Counter(
    "model_offload_events_total",
    "Total number of model offloading events",
    ["model_name"]
)
