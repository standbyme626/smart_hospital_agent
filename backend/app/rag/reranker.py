import os
import time
import torch
import logging
import asyncio
import numpy as np
from typing import List, Tuple
from app.core import torch_patch  # noqa: F401  # Ensure torch int1-int8 patch before transformers import
from transformers import AutoTokenizer, AutoModelForCausalLM, logging as transformers_logging
try:
    from optimum.onnxruntime import ORTModelForCausalLM
except (ImportError, AttributeError):
    print("[Warning] Failed to import optimum.onnxruntime. ONNX support will be disabled.")
    ORTModelForCausalLM = None
try:
    import onnxruntime as ort
except ImportError:
    ort = None
from app.core.monitoring.metrics import RERANK_LATENCY, RERANK_REQUESTS, MODEL_LOAD_EVENTS

# 抑制 Transformers 的警告信息
transformers_logging.set_verbosity_error()

class QwenReranker:
    """
    Qwen3-Reranker 封装类 (支持 ONNX 加速)
    针对 Qwen3-Reranker 的 CausalLM 架构进行了优化，使用 yes/no logit 评分
    
    [Refactor Phase 2] Added Async Support
    """
    def __init__(self, model_path: str):
        from app.core.config import settings
        requested_device = (settings.RERANKER_DEVICE or "auto").strip().lower()
        if requested_device in {"cuda", "gpu"} and torch.cuda.is_available():
            self.device = "cuda"
        elif requested_device == "cpu":
            self.device = "cpu"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = model_path
        self.onnx_path = os.path.join(model_path, "onnx")
        self.instruction = 'Given a web search query, retrieve relevant passages that answer the query'
        self.is_on_gpu = False
        self.model = None
        self.tokenizer = None
        self.use_onnx = False
        self.use_onnx_runtime = False
        self.onnx_input_names = set()
        self.onnx_providers = []
        
        # 注册到模型池
        try:
            from app.core.infra import ModelPoolManager
            ModelPoolManager.register("reranker", self)
        except ImportError:
            pass

        self._reload()

    def _reload(self):
        """加载模型到内存/显存"""
        if self.model is not None and self.is_on_gpu:
            return

        MODEL_LOAD_EVENTS.labels(model_name="reranker").inc()
        start_time = time.time()
        print(f"[Reranker] Loading model from {self.model_path} on {self.device}...")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True, use_fast=False)
            
            # 获取 Yes/No 的 token ID (Qwen specific)
            # 注意：不同模型的 Yes/No token 可能不同，这里假设是 Qwen/Llama 通用的 "Yes"/"No"
            self.token_yes_id = self.tokenizer.encode("Yes", add_special_tokens=False)[0]
            self.token_no_id = self.tokenizer.encode("No", add_special_tokens=False)[0]

            # 尝试加载 ONNX (优先 optimum，缺失时直接 onnxruntime)
            self.use_onnx = False
            self.use_onnx_runtime = False
            onnx_model_file = os.path.join(self.onnx_path, "model.onnx")
            if os.path.exists(onnx_model_file):
                try:
                    print(f"[Reranker] Loading ONNX model from {self.onnx_path}...")
                    available_providers = ort.get_available_providers() if ort is not None else []
                    if self.device == "cuda" and "CUDAExecutionProvider" not in available_providers:
                        raise RuntimeError(
                            f"onnxruntime CUDAExecutionProvider unavailable (available={available_providers}); "
                            "refuse CPU ONNX fallback in CUDA mode"
                        )
                    if ORTModelForCausalLM:
                        self.model = ORTModelForCausalLM.from_pretrained(
                            self.onnx_path,
                            provider="CUDAExecutionProvider" if self.device == "cuda" else "CPUExecutionProvider"
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
                except Exception as e:
                    print(f"[Reranker] ONNX load failed, fallback to PyTorch: {e}")

            if not self.use_onnx:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path, 
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    trust_remote_code=True
                )
                self.model.to(self.device)
                self.model.eval()

            if self.use_onnx and self.use_onnx_runtime:
                self.is_on_gpu = "CUDAExecutionProvider" in self.onnx_providers
            else:
                self.is_on_gpu = self.device == "cuda"
            print(f"[Reranker] Loaded in {time.time() - start_time:.2f}s")
            
        except Exception as e:
            print(f"[Reranker] Failed to load model: {e}. Switching to Passthrough Mode.")
            self.model = None
            # Do not raise, allow system to function without reranking

    def _format_input(self, query: str, text: str) -> str:
        return f"{self.instruction}\nQuery: {query}\nDocument: {text}\nRelevant:"

    async def arerank(self, query: str, docs: list[dict], top_k: int = 3) -> list[dict]:
        """[Async] 异步重排序"""
        return await asyncio.to_thread(self.rerank, query, docs, top_k)

    def rerank(self, query: str, docs: list[dict], top_k: int = 3) -> list[dict]:
        """
        使用 CausalLM 架构进行重排序评分
        """
        if not docs:
            return []
            
        # Graceful degradation
        if self.model is None or self.tokenizer is None:
            return docs[:top_k]

        start_time = time.time()
        RERANK_REQUESTS.inc()
        # 确保模型已加载且更新使用时间
        try:
            from app.core.infra import ModelPoolManager
            ModelPoolManager.mark_used("reranker")
        except ImportError:
            pass
            
        if not self.is_on_gpu:
            self._reload()
            
        formatted_texts = [self._format_input(query, d.get("content", "")) for d in docs]
        
        if self.use_onnx and self.use_onnx_runtime:
            inputs = self.tokenizer(
                formatted_texts,
                padding=True,
                truncation=True,
                return_tensors='np',
                max_length=1024
            )
            ort_inputs = {k: v.astype(np.int64) for k, v in inputs.items() if k in self.onnx_input_names}
            if "position_ids" in self.onnx_input_names and "position_ids" not in ort_inputs:
                batch_size = ort_inputs["input_ids"].shape[0]
                seq_len = ort_inputs["input_ids"].shape[1]
                ort_inputs["position_ids"] = np.broadcast_to(np.arange(seq_len, dtype=np.int64), (batch_size, seq_len))
            logits = np.asarray(self.model.run(None, ort_inputs)[0], dtype=np.float32)
        else:
            inputs = self.tokenizer(
                formatted_texts, 
                padding=True, 
                truncation=True, 
                return_tensors='pt', 
                max_length=1024
            )
            if not self.use_onnx:
                inputs = inputs.to(self.device)
            else:
                inputs = {k: v.to(self.device) for k, v in inputs.items() if v is not None}

            if self.use_onnx:
                ort_inputs = {k: v for k, v in inputs.items() if k in self.model.input_names}
                # 手动补全 position_ids
                if "position_ids" in self.model.input_names and "position_ids" not in ort_inputs:
                    batch_size = ort_inputs["input_ids"].shape[0]
                    seq_len = ort_inputs["input_ids"].shape[1]
                    ort_inputs["position_ids"] = torch.arange(seq_len, device=self.device).expand(batch_size, seq_len)
                
                outputs = self.model(**ort_inputs)
                logits = outputs.logits # (batch, seq, vocab)
            else:
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits

        if isinstance(logits, np.ndarray):
            last_token_logits = logits[:, -1, :]  # (batch, vocab)
            yes_logits = last_token_logits[:, self.token_yes_id]
            no_logits = last_token_logits[:, self.token_no_id]
            scores = np.stack([no_logits, yes_logits], axis=1)
            scores = scores - np.max(scores, axis=1, keepdims=True)
            probs = np.exp(scores)
            probs = probs / np.clip(np.sum(probs, axis=1, keepdims=True), 1e-12, None)
            final_scores = probs[:, 1]
        else:
            with torch.no_grad():
                # 提取最后一个 Token 的 Logits
                last_token_logits = logits[:, -1, :] # (batch, vocab)
                
                # 提取 yes/no 的 logit 并进行 log_softmax
                yes_logits = last_token_logits[:, self.token_yes_id]
                no_logits = last_token_logits[:, self.token_no_id]
                
                scores_tensor = torch.stack([no_logits, yes_logits], dim=1) # (batch, 2)
                scores_tensor = torch.nn.functional.log_softmax(scores_tensor, dim=1)
                
                # 获取 'yes' 的概率作为最终分数
                final_scores = torch.exp(scores_tensor[:, 1]).cpu().float().numpy()
        
        # 将分数回填并排序
        for i in range(min(len(docs), len(final_scores))):
            docs[i]["score"] = float(final_scores[i])
            docs[i]["source"] = "reranked"
            
        docs.sort(key=lambda x: x["score"], reverse=True)
        
        latency = time.time() - start_time
        RERANK_LATENCY.observe(latency)
        
        return docs[:top_k]
