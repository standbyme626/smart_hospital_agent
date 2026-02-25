import os
import sys
import json
import asyncio
import uuid
import pandas as pd
from typing import List, Dict, Any
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

# 强制修正导入路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../.."))
backend_dir = os.path.join(project_root, "backend")
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.core.config import settings
from app.core.llm.llm_factory import get_judge_llm
from langchain_huggingface import HuggingFaceEmbeddings

class MedicalRAGEvaluator:
    """
    医疗 RAG 自动化评估器
    基于 RAGAS 框架，提供忠实度、精度和相关性的量化评估
    """
    def __init__(self):
        # [V6.2 Update] 使用具备“模型+密钥”双重自愈能力的裁判模型
        self.judge_llm = get_judge_llm()
        
        # 使用本地 Embedding 模型进行评估中的向量计算
        import torch
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL_PATH,
            model_kwargs={'device': 'cpu'}
        )
        # [Fix] 强制转换模型为 float32 以避免 numpy 不支持 bfloat16 的问题
        if hasattr(self.embeddings, "_client"):
            self.embeddings._client.to(torch.float32)
        
        # 配置 RAGAS 指标
        self.metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
        ]

    async def run_evaluation(self, test_data: List[Dict[str, Any]], report_path: str = "ragas_report.csv"):
        """
        执行评估流程
        """
        print(f"[DEBUG] 开始 RAGAS 评估，样本量: {len(test_data)}")
        
        dataset = Dataset.from_list(test_data)
        
        # 执行评估
        result = evaluate(
            dataset=dataset,
            metrics=self.metrics,
            llm=self.judge_llm,
            embeddings=self.embeddings, # 显式传入 Embedding
            raise_exceptions=True       # 开启异常抛出，便于调试
        )
        
        # 导出报告
        df = result.to_pandas()
        df.to_csv(report_path, index=False)
        
        # 故障根因分析分析
        self._generate_diagnostic_summary(df)
        
        return result

    def _generate_diagnostic_summary(self, df: pd.DataFrame):
        """
        根据评估分数自动诊断系统瓶颈
        """
        avg_faithfulness = df["faithfulness"].mean()
        avg_relevance = df["answer_relevancy"].mean()
        avg_precision = df["context_precision"].mean()
        
        print("\n" + "="*50)
        print("🏥 医疗 RAG 系统自动化诊断报告")
        print("="*50)
        print(f"1. 忠实度 (Faithfulness): {avg_faithfulness:.4f}")
        print(f"2. 答案相关性 (Relevance): {avg_relevance:.4f}")
        print(f"3. 上下文精度 (Precision): {avg_precision:.4f}")
        print("-" * 50)
        
        # 根因映射逻辑
        if avg_faithfulness < 0.7:
            print("🚩 诊断：【生成层故障】存在严重幻觉风险。建议：加强 System Prompt 约束或更换更高参数模型。")
        if avg_precision < 0.7:
            print("🚩 诊断：【检索层故障】检索噪音过大。建议：优化 Reranker 排序或改进 Chunking 策略。")
        if avg_relevance < 0.7:
            print("🚩 诊断：【理解层故障】答非所问。建议：优化 Query Rewriting 逻辑。")
        
        if avg_faithfulness >= 0.8 and avg_precision >= 0.8:
            print("✅ 诊断：系统运行稳健，具备工业级交付能力。")
        print("="*50 + "\n")

async def test_evaluator():
    # 扩展测试数据集，覆盖多种医疗场景
    evaluator = MedicalRAGEvaluator()
    sample_data = [
        {
            "question": "血红蛋白 90 是什么意思？",
            "contexts": ["血红蛋白参考范围为 110-160g/L。90g/L 属于轻中度贫血。"],
            "answer": "您的血红蛋白为 90g/L，低于正常范围，属于贫血。建议咨询医生了解具体原因。",
            "ground_truth": "血红蛋白 90g/L 低于 110-160g/L 的标准范围，属于贫血范畴。"
        },
        {
            "question": "头疼伴随呕吐应该挂什么科？",
            "contexts": ["神经内科主要诊治头痛、头晕、肢体麻木等症状。如果伴有剧烈呕吐，需警惕颅内压增高。"],
            "answer": "建议您挂神经内科。头痛伴随呕吐可能涉及神经系统问题，请尽快就医。",
            "ground_truth": "头痛伴呕吐建议优先挂神经内科，排除颅内压力异常。"
        },
        {
            "question": "阿司匹林肠溶片怎么吃？",
            "contexts": ["阿司匹林肠溶片应在饭前用适量水送服。肠溶片设计是为了在肠道释放，减少胃部刺激。"],
            "answer": "阿司匹林肠溶片通常建议饭前服用，用温开水送服。请勿压碎或咀嚼。",
            "ground_truth": "应饭前送服，保持药片完整，利用其肠溶特性保护胃粘膜。"
        },
        {
            "question": "糖尿病患者可以吃西瓜吗？",
            "contexts": ["西瓜的血糖生成指数(GI)较高，但含糖量相对较低。糖尿病患者在血糖控制稳定的情况下可少量食用。"],
            "answer": "糖尿病患者在血糖平稳时可以少量吃西瓜（建议不超过 200g），但要注意监测血糖波动。",
            "ground_truth": "血糖稳定时可限量食用，西瓜 GI 值高但含糖量低，关键在于控制总量。"
        },
        {
            "question": "过敏性鼻炎有哪些症状？",
            "contexts": ["过敏性鼻炎典型症状包括阵发性喷嚏、清水样鼻涕、鼻痒和鼻塞。常伴有眼痒、结膜充血。"],
            "answer": "过敏性鼻炎常表现为打喷嚏、流清鼻涕、鼻子痒和鼻塞。部分患者还会眼睛红痒。",
            "ground_truth": "主要症状为喷嚏、清涕、鼻痒鼻塞，可能伴有眼部过敏症状。"
        }
    ]
    await evaluator.run_evaluation(sample_data)

if __name__ == "__main__":
    import sys
    # 修复导入路径以便直接运行
    sys.path.append(os.path.join(os.getcwd(), "backend"))
    asyncio.run(test_evaluator())
