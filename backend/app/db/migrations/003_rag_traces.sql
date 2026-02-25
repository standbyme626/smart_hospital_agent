"""
RAG 追踪系统数据库 Schema

用于记录每次 RAG 查询的完整链路:
- 用户查询
- 检索到的文档及相似度
- LLM 生成的回答
- Token 消耗及耗时
"""

CREATE TABLE IF NOT EXISTS rag_traces (
id SERIAL PRIMARY KEY,
    
    -- 查询信息
    query TEXT NOT NULL,
    query_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    session_id VARCHAR(255),
    
    -- 检索信息
    retrieved_docs JSONB, -- [{chunk_id, content, score, metadata}]
    retrieval_latency_ms INT,
    
    -- LLM 信息
    llm_answer TEXT,
    llm_prompt TEXT,
    llm_model VARCHAR(100),
    prompt_tokens INT,
    completion_tokens INT,
    total_tokens INT,
    llm_latency_ms INT,
    
    -- 评估信息 (可选,由离线任务填充)
    faithfulness_score FLOAT,
    relevancy_score FLOAT,
    
    -- 元数据
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_query_timestamp (query_timestamp),
    INDEX idx_session_id (session_id)
);

COMMENT ON TABLE rag_traces IS 'RAG 查询链路追踪记录';
COMMENT ON COLUMN rag_traces.retrieved_docs IS '检索到的文档列表(JSON数组)';
COMMENT ON COLUMN rag_traces.faithfulness_score IS 'RAGAs 评估得分: 答案与检索文档的忠实度';
COMMENT ON COLUMN rag_traces.relevancy_score IS 'RAGAs 评估得分: 答案与问题的相关性';
