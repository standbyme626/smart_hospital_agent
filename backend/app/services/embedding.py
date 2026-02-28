
import os
import time
import torch
import threading
import gc
import numpy as np
import torch.nn.functional as F
from app.core import torch_patch  # noqa: F401  # Ensure torch int1-int8 patch before transformers import
from transformers import AutoTokenizer, AutoModel
import transformers.utils
# import transformers.tokenization_utils_tokenizers  <-- Removed to fix Import Error with newer transformers

if not hasattr(transformers.utils, "is_offline_mode"):
    def _is_offline_mode():
        return os.environ.get("TRANSFORMERS_OFFLINE", "0").upper() in ("1", "ON", "YES", "TRUE")
    transformers.utils.is_offline_mode = _is_offline_mode

try:
    from optimum.onnxruntime import ORTModelForFeatureExtraction
except (ImportError, AttributeError):
    # optimum is not compatible with the current transformers version
    # print("[Warning] Failed to import optimum.onnxruntime. ONNX support will be disabled.")
    ORTModelForFeatureExtraction = None
try:
    import onnxruntime as ort
except ImportError:
    ort = None

from app.core.config import settings
from app.core.monitoring.metrics import EMBEDDING_LATENCY, EMBEDDING_REQUESTS, MODEL_LOAD_EVENTS
from app.core.models.vram_manager import vram_manager

class EmbeddingService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingService, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        # Device policy: auto | cuda | cpu
        requested_device = (settings.EMBEDDING_DEVICE or "auto").strip().lower()
        if requested_device in {"cuda", "gpu"} and torch.cuda.is_available():
            self.device = "cuda"
        elif requested_device == "cpu":
            self.device = "cpu"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[EmbeddingService] Initializing with device: {self.device}")
        
        self.model_path = settings.EMBEDDING_MODEL_PATH
        self.onnx_path = os.path.join(self.model_path, "onnx")
        self.is_on_gpu = False
        self.model = None
        self.tokenizer = None
        self.use_gguf = False
        self.use_onnx = False
        self.use_onnx_runtime = False
        self.onnx_input_names = set()
        self.onnx_providers = []
        self._lock = threading.Lock() # Add thread safety for loading
        self.degraded_mode = False
        self.dim = 1024
        
        # [VRAM] Register to VRAM Manager
        vram_manager.register_model("embedding", self)
        
        # Auto-load on init? No, lazy load is better for startup speed.
        # But for "Server Ready" status, we might want to load.
        # Currently the code calls _reload() in __init__ if we follow the original logic? 
        # No, the original code had `self._reload()` at the end of `_initialize`.
        # I will keep it but fix the loading logic inside _reload.
        self._reload()

    def unload(self):
        """Unload model from VRAM (Called by VRAMManager)."""
        with self._lock:
            if self.model is None and self.tokenizer is None:
                return

            print("[EmbeddingService] Unloading model...")
            self.model = None
            self.tokenizer = None
            self.is_on_gpu = False
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            print("[EmbeddingService] Model unloaded.")

    def is_loaded(self) -> bool:
        return self.model is not None

    def _reload(self):
        """加载或重载模型到 GPU (线程安全)"""
        with self._lock:
            if self.is_loaded():
                vram_manager.mark_used("embedding")
                return

            # [VRAM] Orchestrate
            vram_manager.orchestrate_pre_inference(required_mb=600)
            
            # Continue loading...

        MODEL_LOAD_EVENTS.labels(model_name="embedding").inc()
        
        # 1. 优先检查是否为 GGUF 模型 (单文件)
        if self.model_path.endswith(".gguf"):
            print(f"[EmbeddingService] Loading GGUF model from {self.model_path}...")
            try:
                from llama_cpp import Llama
                # n_gpu_layers=-1 表示尽可能将所有层加载到 GPU
                # embedding=True 开启嵌入模式
                self.model = Llama(
                    model_path=self.model_path,
                    embedding=True,
                    n_gpu_layers=-1 if self.device == "cuda" else 0,
                    verbose=False
                )
                self.use_gguf = True
                self.is_on_gpu = (self.device == "cuda")
                print("[EmbeddingService] GGUF Model loaded successfully.")
                return
            except Exception as e:
                print(f"[EmbeddingService] Failed to load GGUF model: {e}")
                # 如果 GGUF 加载失败，通常没有 fallback，直接抛出
                raise e

        # 2. 尝试加载 ONNX 模型 (优先 optimum，缺失时直接使用 onnxruntime)
        onnx_model_file = os.path.join(self.onnx_path, "model.onnx")
        if os.path.exists(onnx_model_file):
            print(f"[EmbeddingService] Loading ONNX model from {self.onnx_path} on {self.device}...")
            try:
                available_providers = ort.get_available_providers() if ort is not None else []
                if self.device == "cuda" and "CUDAExecutionProvider" not in available_providers:
                    raise RuntimeError(
                        f"onnxruntime CUDAExecutionProvider unavailable (available={available_providers}); "
                        "refuse CPU ONNX fallback in CUDA mode"
                    )

                if self.tokenizer is None:
                    tok_path = self.onnx_path if os.path.exists(os.path.join(self.onnx_path, "tokenizer_config.json")) else self.model_path
                    self.tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)

                if ORTModelForFeatureExtraction is not None:
                    self.model = ORTModelForFeatureExtraction.from_pretrained(
                        self.onnx_path,
                        provider="CUDAExecutionProvider" if self.device == "cuda" else "CPUExecutionProvider",
                        provider_options={"device_id": 0} if self.device == "cuda" else None
                    )
                    self.use_onnx_runtime = False
                elif ort is not None:
                    providers = ["CPUExecutionProvider"]
                    if self.device == "cuda":
                        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    self.model = ort.InferenceSession(onnx_model_file, providers=providers)
                    self.onnx_input_names = {i.name for i in self.model.get_inputs()}
                    self.onnx_providers = list(self.model.get_providers())
                    if self.device == "cuda" and "CUDAExecutionProvider" not in self.onnx_providers:
                        raise RuntimeError(
                            f"onnxruntime session is not on CUDA (providers={self.onnx_providers}); "
                            "fallback to PyTorch CUDA"
                        )
                    self.use_onnx_runtime = True
                else:
                    raise RuntimeError("Neither optimum.onnxruntime nor onnxruntime is available")

                self.use_onnx = True
                if self.use_onnx_runtime:
                    self.is_on_gpu = "CUDAExecutionProvider" in self.onnx_providers
                else:
                    self.is_on_gpu = (self.device == "cuda")
                print("[EmbeddingService] ONNX Model loaded successfully.")
                return
            except Exception as e:
                print(f"[EmbeddingService] Failed to load ONNX model, falling back to PyTorch: {e}")

        # Fallback to PyTorch
        print(f"[EmbeddingService] Loading PyTorch model from {self.model_path} on {self.device}...")
        self.use_onnx = False
        self.use_onnx_runtime = False
        try:
            if self.tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(
                self.model_path, trust_remote_code=True, dtype=torch.float16 if self.device=="cuda" else torch.float32
            ).to(self.device)
            self.model.eval()
            self.is_on_gpu = (self.device == "cuda")
            print("[EmbeddingService] PyTorch Model loaded successfully.")
        except Exception as e:
            print(f"[EmbeddingService] Failed to load model: {e}")
            print("[Warning] Embedding Model failed to load. Entering DEGRADED MODE (Dummy Embeddings).")
            self.degraded_mode = True
            # raise e  <-- Do not raise, allow degradation

    def _onnx_np_inputs(self, texts, max_length: int):
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="np",
        )
        ort_inputs = {k: v.astype(np.int64) for k, v in inputs.items() if k in self.onnx_input_names}
        if "position_ids" in self.onnx_input_names and "position_ids" not in ort_inputs:
            batch_size, seq_len = ort_inputs["input_ids"].shape
            ort_inputs["position_ids"] = np.broadcast_to(np.arange(seq_len, dtype=np.int64), (batch_size, seq_len))
        return ort_inputs, inputs["attention_mask"].astype(np.float32)

    @staticmethod
    def _mean_pool_numpy(last_hidden_state: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        mask = np.expand_dims(attention_mask, axis=-1)
        sum_embeddings = np.sum(last_hidden_state * mask, axis=1)
        sum_mask = np.clip(np.sum(mask, axis=1), 1e-9, None)
        embeddings = sum_embeddings / sum_mask
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.clip(norms, 1e-12, None)

    def offload(self):
        """从 GPU 卸载模型以节省显存"""
        if not self.is_on_gpu or self.model is None:
            return
        
        print(f"[EmbeddingService] Offloading model to CPU/Freeing VRAM...")
        try:
            # 对于 ONNX，直接销毁 session 是释放显存最彻底的方式
            self.model = None
            self.is_on_gpu = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("[EmbeddingService] Model offloaded successfully.")
        except Exception as e:
            print(f"[EmbeddingService] Offload failed: {e}")

    def get_embedding(self, text: str) -> list:
        start_time = time.time()
        EMBEDDING_REQUESTS.inc()
        
        # [VRAM] Mark used
        vram_manager.mark_used("embedding")
            
        if not self.is_loaded() and not self.degraded_mode:
            self._reload()
            
        if self.degraded_mode:
             return [0.001] * self.dim

        # [GGUF] Handle GGUF inference
        if self.use_gguf:
            response = self.model.create_embedding(text)
            embedding = response['data'][0]['embedding']
            latency = time.time() - start_time
            EMBEDDING_LATENCY.observe(latency)
            return embedding

        if self.use_onnx and self.use_onnx_runtime:
            ort_inputs, attention_mask = self._onnx_np_inputs([text], max_length=512)
            outputs = self.model.run(None, ort_inputs)
            last_hidden_state = np.asarray(outputs[0], dtype=np.float32)
            embedding = self._mean_pool_numpy(last_hidden_state, attention_mask)
            latency = time.time() - start_time
            EMBEDDING_LATENCY.observe(latency)
            return embedding[0].tolist()

        inputs = self.tokenizer([text], padding=True, truncation=True, max_length=512, return_tensors="pt")
        if not self.use_onnx:
            inputs = inputs.to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
        else:
            inputs = {k: v.to(self.device) for k, v in inputs.items() if v is not None}
            ort_inputs = {k: v for k, v in inputs.items() if k in self.model.input_names}
            if "position_ids" in self.model.input_names and "position_ids" not in ort_inputs:
                seq_len = ort_inputs["input_ids"].shape[1]
                ort_inputs["position_ids"] = torch.arange(seq_len, device=self.device).unsqueeze(0)
            outputs = self.model(**ort_inputs)

        if not self.use_onnx:
            last_hidden_state = outputs.last_hidden_state
        else:
            last_hidden_state = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"]
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * mask, dim=1)
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
        embedding = sum_embeddings / sum_mask
        embedding = F.normalize(embedding, p=2, dim=1)

        latency = time.time() - start_time
        EMBEDDING_LATENCY.observe(latency)
        
        return embedding[0].cpu().tolist()

    def batch_get_embeddings(self, texts: list, batch_size: int = 16) -> list:
        # [VRAM] Mark used
        vram_manager.mark_used("embedding")

        if self.degraded_mode:
            return [[0.001] * self.dim for _ in texts]

        # [GGUF] Handle GGUF inference
        if hasattr(self, 'use_gguf') and self.use_gguf:
            if not self.is_loaded() and not self.degraded_mode:
                self._reload()
            
            if self.degraded_mode:
                return [[0.001] * self.dim for _ in texts]
            
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                response = self.model.create_embedding(batch)
                # Ensure results are sorted by index
                sorted_data = sorted(response['data'], key=lambda x: x['index'])
                batch_embeddings = [item['embedding'] for item in sorted_data]
                all_embeddings.extend(batch_embeddings)
            return all_embeddings

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            if self.use_onnx and self.use_onnx_runtime:
                ort_inputs, attention_mask_np = self._onnx_np_inputs(batch, max_length=512)
                outputs = self.model.run(None, ort_inputs)
                last_hidden_state_np = np.asarray(outputs[0], dtype=np.float32)
                embeddings_np = self._mean_pool_numpy(last_hidden_state_np, attention_mask_np)
                all_embeddings.extend(embeddings_np.tolist())
                continue

            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )

            if not self.use_onnx:
                inputs = inputs.to(self.device)
                with torch.no_grad():
                    outputs = self.model(**inputs)
            else:
                inputs = {k: v.to(self.device) for k, v in inputs.items() if v is not None}
                ort_inputs = {k: v for k, v in inputs.items() if k in self.model.input_names}
                if "position_ids" in self.model.input_names and "position_ids" not in ort_inputs:
                    batch_size_actual = ort_inputs["input_ids"].shape[0]
                    seq_len = ort_inputs["input_ids"].shape[1]
                    ort_inputs["position_ids"] = torch.arange(seq_len, device=self.device).expand(batch_size_actual, seq_len)
                outputs = self.model(**ort_inputs)
            
            # Mean Pooling
            last_hidden_state = outputs.last_hidden_state
            attention_mask = inputs["attention_mask"]
            
            # a. 将 attention_mask 扩展到与 last_hidden_state 相同的维度
            mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            
            # b. 计算所有有效 token 的向量总和
            sum_embeddings = torch.sum(last_hidden_state * mask, dim=1)
            
            # c. 计算有效 token 的总数
            sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
            
            # d. 均值向量 = sum_embeddings / sum_mask
            embeddings = sum_embeddings / sum_mask
            
            # 保持现有的 F.normalize 逻辑
            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.extend(embeddings.cpu().tolist())
        return all_embeddings
