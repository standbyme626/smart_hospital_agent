import json
import os
import logging
from typing import List, Dict, Optional
from app.core.models.local_slm import LocalSLMService

logger = logging.getLogger(__name__)

class QualityCollector:
    """
    [Optimization Plan 5] MLOps: 负样本自动采集与 DPO 数据格式化工具
    核心功能：在系统运行或测试过程中，捕获模型异常输出并转化为 DPO 训练样本。
    """
    def __init__(self, output_file: Optional[str] = None):
        if output_file is None:
            # 默认存储在项目根目录的 data 文件夹
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
            self.output_file = os.path.join(base_dir, "data", "dpo_negative_samples.jsonl")
        else:
            self.output_file = output_file
            
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)

    def collect_negative_sample(self, prompt: str, expected_category: str, actual_output: str, is_uncertain: bool = False):
        """
        添加一个负样本到采集库。
        
        Args:
            prompt: 用户原始输入
            expected_category: 系统预期的正确分类
            actual_output: 模型生成的错误输出 (包含 <think> 块)
            is_uncertain: 是否因为模型触发了 UNCERTAIN 逻辑而采集
        """
        # 构造理想的 Chosen 回答 (符合 DPO v11.2 的 CoT + 标签格式)
        chosen_response = (
            f"<think>\n用户主诉为：{prompt}。经过分析，这符合 {expected_category} 的特征。\n</think>\n"
            f"建议挂号科室：【{'全科' if expected_category != 'CRISIS' else '急诊科'}】\n"
            f"意图分类：{expected_category}"
        )
        
        sample = {
            "prompt": prompt,
            "chosen": chosen_response,
            "rejected": actual_output,
            "metadata": {
                "expected": expected_category,
                "is_uncertain": is_uncertain,
                "timestamp": os.popen('date +"%Y-%m-%d %H:%M:%S"').read().strip(),
                "source": "automated_quality_collector"
            }
        }
        
        self._append_to_file(sample)
        logger.info(f"🚩 [QualityCollector] Sample captured: {prompt[:20]}... -> Expected: {expected_category}")

    def _append_to_file(self, sample: Dict):
        """线程/进程安全的追加逻辑 (简单实现)"""
        try:
            with open(self.output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"❌ [QualityCollector] Failed to save sample: {e}")

    async def run_benchmark_collection(self, test_data: List[Dict]):
        """
        批量运行测试并采集负样本 (用于 MLOps 离线评估阶段)
        """
        slm = LocalSLMService()
        categories = ["VAGUE_SYMPTOM", "COMPLEX_SYMPTOM", "GREETING", "CRISIS", "INFO"]
        
        logger.info(f"🧪 [QualityCollector] Starting batch collection from {len(test_data)} cases...")
        
        count = 0
        for case in test_data:
            text = case.get("text", "")
            expected = case.get("expected", "")
            
            try:
                # 执行推理
                actual_category = await slm.constrained_classify(text, categories, reasoning=True)
                raw_output = getattr(slm, "_last_raw_output", "")
                
                # 采集判定：1. 分类错误 2. 模型明确表示不确定 (UNCERTAIN)
                is_mismatch = (actual_category != expected)
                is_uncertain = "UNCERTAIN" in actual_category or "不确定" in raw_output
                
                if is_mismatch or is_uncertain:
                    self.collect_negative_sample(text, expected, raw_output, is_uncertain)
                    count += 1
            
            except Exception as e:
                logger.error(f"❌ [QualityCollector] Error processing case '{text[:20]}': {e}")
        
        logger.info(f"✅ [QualityCollector] Collection finished. Samples added: {count}")
