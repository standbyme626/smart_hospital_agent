-- RAG 知识库元数据表
CREATE TABLE IF NOT EXISTS medical_chunks (
    id SERIAL PRIMARY KEY,
    chunk_id VARCHAR(64) UNIQUE NOT NULL, -- 对应 Milvus 中的 ID
    content TEXT NOT NULL,                -- 文本内容
    disease_name VARCHAR(200),            -- 关联疾病名
    department VARCHAR(100),              -- 所属科室
    icd10_code VARCHAR(20),               -- ICD-10 编码
    keywords TEXT[],                      -- 关键词数组
    source_doc VARCHAR(500),              -- 来源文档
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_disease ON medical_chunks(disease_name);
CREATE INDEX IF NOT EXISTS idx_dept ON medical_chunks(department);
CREATE INDEX IF NOT EXISTS idx_icd10 ON medical_chunks(icd10_code);
CREATE INDEX IF NOT EXISTS idx_chunk_id ON medical_chunks(chunk_id);

-- 患者信息表
CREATE TABLE IF NOT EXISTS patients (
    id VARCHAR(32) PRIMARY KEY, -- 如 P001
    name VARCHAR(100) NOT NULL,
    age INT,
    gender VARCHAR(10),
    allergies TEXT[],
    created_at TIMESTAMP DEFAULT NOW()
);

-- 电子病历表
CREATE TABLE IF NOT EXISTS ehr_records (
    id SERIAL PRIMARY KEY,
    patient_id VARCHAR(32) REFERENCES patients(id),
    visit_date TIMESTAMP DEFAULT NOW(),
    chief_complaint TEXT, -- 主诉
    diagnosis VARCHAR(200),
    treatment_plan TEXT,
    department VARCHAR(100)
);

CREATE INDEX IF NOT EXISTS idx_patient_id ON ehr_records(patient_id);

-- ==========================================
-- RAG 链路追踪表 (Added for Day 2)
-- ==========================================
CREATE TABLE IF NOT EXISTS rag_traces (
    id SERIAL PRIMARY KEY,
    trace_id UUID NOT NULL DEFAULT gen_random_uuid(),
    session_id VARCHAR(64),
    
    -- 输入与输出
    query TEXT NOT NULL,
    final_answer TEXT,
    
    -- 核心指标
    total_latency_ms FLOAT,
    retrieval_score FLOAT, 
    
    -- 详细追踪数据 (JSONB)
    details JSONB DEFAULT '{}',
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_rag_traces_trace_id ON rag_traces(trace_id);
CREATE INDEX IF NOT EXISTS idx_rag_traces_created_at ON rag_traces(created_at);
