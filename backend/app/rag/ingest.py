"""
⚠️ DEPRECATED: This script is deprecated and may cause data inconsistency.
Please use 'rebuild_milvus_v4_aligned.py' for production ingestion to ensure 
Milvus IDs are aligned with PostgreSQL medical_chunks table.

Standard Ingestion Flow:
1. Load data into PostgreSQL 'medical_chunks' first.
2. Use 'rebuild_milvus_v4_aligned.py' to sync SQL data to Milvus.
"""
import json
import torch
import time
import os
import sys
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from torch.utils.data import DataLoader, Dataset
from app.core.config import settings

# ================= 配置信息 (Configuration) =================
# 模型路径
MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", os.path.join(settings.PROJECT_ROOT, "models", "Qwen3-Embedding-0.6B"))
# 数据路径
JSONL_FILE = os.path.join(settings.PROJECT_ROOT, "data", "augmented_only.jsonl")
# 进度记录文件
CHECKPOINT_FILE = "ingest_checkpoint.txt"
# Milvus 配置
MILVUS_HOST = settings.MILVUS_HOST
MILVUS_PORT = settings.MILVUS_PORT
COLLECTION_NAME = "huatuo_knowledge"  # 医疗知识库集合名称
DIMENSION = 1024  # 模型维度

# 性能参数
BATCH_SIZE = 16
MAX_LENGTH = 512
INSERT_BUFFER_SIZE = 1000
NUM_WORKERS = 0 # 避免多进程问题，先设为0

# ==========================================================

class MedicalDataset(Dataset):
    """
    医疗数据集类 (Medical Dataset Class)
    用于加载和处理 augmented_only.jsonl 文件。
    """
    def __init__(self, file_path, start_line=0):
        """
        初始化数据集 (Initialize Dataset)
        
        Args:
            file_path: 数据文件路径
            start_line: 起始行号，用于断点续传
        """
        self.data = []
        print(f"正在加载数据集 (从第 {start_line} 行开始)...")
        if not os.path.exists(file_path):
            print(f"错误: 文件不存在 {file_path}")
            return
            
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i < start_line:
                    continue
                self.data.append(line)
        print(f"✅ 加载完成，共 {len(self.data)} 条数据待处理")

    def __len__(self):
        """
        获取数据集大小 (Get Dataset Length)
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        获取单条数据 (Get Item)
        提取 jsonl 中的 text 字段作为向量化内容。
        """
        line = self.data[idx]
        try:
            item = json.loads(line)
            # 优先使用 text 字段，如果不存在则组合问题和答案
            text = item.get("text", "")
            if not text:
                # 尝试其他字段并组合
                q = item.get("问题", "")
                a = item.get("答案", "")
                text = f"问题：{q} 答案：{a}"
            
            # 同时返回元数据以便存入数据库 (这里简化仅返回文本，如果需要存 metadata 需要修改 Dataset 返回结构)
            return text
        except:
            return ""

def load_checkpoint():
    """
    加载进度断点 (Load Checkpoint)
    
    Returns:
        int: 上次处理到的行号
    """
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            try:
                return int(f.read().strip())
            except ValueError:
                return 0
    return 0

def save_checkpoint(line_idx):
    """
    保存进度断点 (Save Checkpoint)
    
    Args:
        line_idx: 当前处理到的行号
    """
    with open(CHECKPOINT_FILE, 'w') as f:
        f.write(str(line_idx))

def connect_milvus():
    """
    连接 Milvus 并初始化集合 (Connect Milvus & Init Collection)
    
    Returns:
        Collection: Milvus 集合对象
    """
    print(f"正在连接 Milvus ({MILVUS_HOST}:{MILVUS_PORT})...")
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
    
    if utility.has_collection(COLLECTION_NAME):
        print(f"集合 {COLLECTION_NAME} 已存在，加载中...")
        collection = Collection(COLLECTION_NAME)
        collection.load()
        return collection
    
    print(f"创建新集合 {COLLECTION_NAME}...")
    fields = [
        # 主键 ID，自动增长
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        # 文本内容，存储原始 QA 对
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=5000), # 稍微调小一点避免过大
        # 向量数据
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION)
    ]
    schema = CollectionSchema(fields, "Smart Hospital Medical Knowledge Base")
    collection = Collection(COLLECTION_NAME, schema)
    
    # 创建索引
    print("创建向量索引...")
    index_params = {
        "metric_type": "COSINE", 
        "index_type": "HNSW", 
        "params": {"M": 16, "efConstruction": 256}
    }
    collection.create_index(field_name="vector", index_params=index_params)
    collection.load()
    return collection

def main():
    """
    主函数 (Main Function)
    执行数据加载、向量化和入库流程。
    """
    # 1. 获取上次进度
    start_line = load_checkpoint()
    print(f"🔄 检测到断点：从第 {start_line} 行继续运行")

    # 2. 准备模型
    print(f"正在加载模型 {MODEL_PATH}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            MODEL_PATH, trust_remote_code=True, torch_dtype=torch.float16 if device=="cuda" else torch.float32
        ).to(device)
        model.eval()
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    # 3. 连接数据库
    try:
        collection = connect_milvus()
    except Exception as e:
        print(f"Milvus 连接失败: {e}")
        return
    
    # 4. 加载数据
    dataset = MedicalDataset(JSONL_FILE, start_line=start_line)
    if len(dataset) == 0:
        print("没有数据需要处理，退出。")
        return
        
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=(device=="cuda"))

    texts_buffer = []
    vectors_buffer = []
    start_time = time.time()
    total_processed_this_run = 0
    
    pbar = tqdm(total=len(dataset), desc="🚀 向量化进度", unit="条")

    try:
        with torch.no_grad():
            for batch in dataloader:
                # 过滤空数据
                batch = [t for t in batch if t and len(t) > 5] 
                if not batch: 
                    pbar.update(BATCH_SIZE) # 即使跳过也要更新进度条
                    continue

                # Tokenize & Embedding
                inputs = tokenizer(batch, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to(device)
                outputs = model(**inputs)
                
                # 获取向量 (Last Token Pooling，与参考代码一致)
                embeddings = outputs.last_hidden_state[:, -1, :]
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                vectors = embeddings.to(torch.float32).cpu().numpy().tolist()
                
                texts_buffer.extend(batch)
                vectors_buffer.extend(vectors)

                # 达到 buffer 大小，写入 Milvus 并保存进度
                if len(texts_buffer) >= INSERT_BUFFER_SIZE:
                    # 补充 Metadata (department, disease, source) 以匹配 Schema
                    department_buffer = ["General"] * len(texts_buffer)
                    disease_buffer = [""] * len(texts_buffer)
                    source_buffer = ["Huatuo26M"] * len(texts_buffer)
                    
                    collection.insert([texts_buffer, vectors_buffer, department_buffer, disease_buffer, source_buffer])
                    
                    # 更新进度记录
                    total_processed_this_run += len(texts_buffer)
                    current_total_line = start_line + total_processed_this_run
                    save_checkpoint(current_total_line)
                    
                    texts_buffer = []
                    vectors_buffer = []

                # 更新进度条展示
                pbar.update(len(batch))
                elapsed = time.time() - start_time
                tps = total_processed_this_run / elapsed if elapsed > 0 else 0
                pbar.set_postfix({
                    "TPS": f"{tps:.1f}/s",
                })

    except KeyboardInterrupt:
        print("\n检测到手动停止，正在保存当前缓冲区数据...")
        if texts_buffer:
            department_buffer = ["General"] * len(texts_buffer)
            disease_buffer = [""] * len(texts_buffer)
            source_buffer = ["Huatuo26M"] * len(texts_buffer)
            collection.insert([texts_buffer, vectors_buffer, department_buffer, disease_buffer, source_buffer])
            save_checkpoint(start_line + total_processed_this_run + len(texts_buffer))
        print("进度已保存，下次运行将自动续传。")
        return
    except Exception as e:
        print(f"\n发生错误: {e}")
        if texts_buffer:
            department_buffer = ["General"] * len(texts_buffer)
            disease_buffer = [""] * len(texts_buffer)
            source_buffer = ["Huatuo26M"] * len(texts_buffer)
            collection.insert([texts_buffer, vectors_buffer, department_buffer, disease_buffer, source_buffer])
            save_checkpoint(start_line + total_processed_this_run + len(texts_buffer))
        return

    # 处理剩余尾数
    if texts_buffer:
        department_buffer = ["General"] * len(texts_buffer)
        disease_buffer = [""] * len(texts_buffer)
        source_buffer = ["Huatuo26M"] * len(texts_buffer)
        collection.insert([texts_buffer, vectors_buffer, department_buffer, disease_buffer, source_buffer])
        save_checkpoint(start_line + total_processed_this_run + len(texts_buffer))

    collection.flush()
    print(f"\n✅ 处理完成！当前集合内数据总量: {collection.num_entities}")

if __name__ == "__main__":
    # 为了让 app.core.config 能正常工作，需要添加路径到 sys.path
    sys.path.append(os.getcwd())
    main()
