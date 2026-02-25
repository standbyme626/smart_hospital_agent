-- RAG 链路追踪表
CREATE TABLE IF NOT EXISTS rag_traces (
    id SERIAL PRIMARY KEY,
    trace_id UUID NOT NULL DEFAULT gen_random_uuid(),
    session_id VARCHAR(64),
    
    -- 输入与输出
    query TEXT NOT NULL,
    final_answer TEXT,
    
    -- 核心指标 (用于快速分析)
    total_latency_ms FLOAT,
    retrieval_score FLOAT, -- 最终结果的平均得分
    
    -- 详细追踪数据 (JSONB 用于灵活扩展)
    -- 结构示例:
    -- {
    --   "steps": {
    --     "vector_search": {"latency": 120, "count": 20},
    --     "bm25_search": {"latency": 50, "count": 20},
    --     "rrf_fusion": {"latency": 5},
    --     "rerank": {"latency": 300, "count": 3}
    --   },
    --   "config": {"top_k": 3, "use_router": false}
    -- }
    details JSONB DEFAULT '{}',
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 索引
CREATE INDEX IF NOT EXISTS idx_rag_traces_trace_id ON rag_traces(trace_id);
CREATE INDEX IF NOT EXISTS idx_rag_traces_created_at ON rag_traces(created_at);
