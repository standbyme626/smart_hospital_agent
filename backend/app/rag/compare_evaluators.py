import os
import sys
import asyncio
import pandas as pd
from typing import List, Dict, Any
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
)

# 路径修正
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../.."))
backend_dir = os.path.join(project_root, "backend")
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.core.config import settings
from app.core.llm.llm_factory import SmartRotatingLLM
from langchain_huggingface import HuggingFaceEmbeddings
import torch

class ComparisonEvaluator:
    def __init__(self):
        # 初始化两种裁判：本地优先 vs 云端优先
        self.local_judge = SmartRotatingLLM(temperature=0.0, max_tokens=512, prefer_local=True)
        self.cloud_judge = SmartRotatingLLM(temperature=0.0, max_tokens=512, prefer_local=False)
        
        # 公用的 Embedding
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL_PATH,
            model_kwargs={'device': 'cpu'}
        )
        if hasattr(self.embeddings, "_client"):
            self.embeddings._client.to(torch.float32)
            
        self.metrics = [faithfulness, answer_relevancy, context_precision]

    async def run_comparison(self, test_data: List[Dict[str, Any]]):
        dataset = Dataset.from_list(test_data)
        
        print("\n🚀 [1/2] 正在运行【本地模型 Qwen3-1.7B】打分...")
        local_result = evaluate(
            dataset=dataset,
            metrics=self.metrics,
            llm=self.local_judge,
            embeddings=self.embeddings,
            raise_exceptions=True
        )
        local_df = local_result.to_pandas()
        
        print("\n🚀 [2/2] 正在运行【云端大模型 Qwen-Turbo/Plus】打分...")
        cloud_result = evaluate(
            dataset=dataset,
            metrics=self.metrics,
            llm=self.cloud_judge,
            embeddings=self.embeddings,
            raise_exceptions=True
        )
        cloud_df = cloud_result.to_pandas()
        
        # 合并对比结果
        comparison_report = []
        for i in range(len(test_data)):
            comparison_report.append({
                "Question": test_data[i]["question"],
                "Local_Faithfulness": local_df.iloc[i]["faithfulness"],
                "Cloud_Faithfulness": cloud_df.iloc[i]["faithfulness"],
                "Local_Relevancy": local_df.iloc[i]["answer_relevancy"],
                "Cloud_Relevancy": cloud_df.iloc[i]["answer_relevancy"],
                "Local_Precision": local_df.iloc[i]["context_precision"],
                "Cloud_Precision": cloud_df.iloc[i]["context_precision"],
            })
            
        report_df = pd.DataFrame(comparison_report)
        report_df.to_csv("evaluator_comparison_20.csv", index=False)
        
        # 打印汇总对比
        print("\n" + "="*60)
        print("📊 本地 vs 云端裁判评分横向对比汇总")
        print("="*60)
        print(f"{'指标':<20} | {'本地平均分':<12} | {'云端平均分':<12} | {'差异':<8}")
        print("-" * 60)
        
        metrics_names = ["faithfulness", "answer_relevancy", "context_precision"]
        for m in metrics_names:
            l_avg = local_df[m].mean()
            c_avg = cloud_df[m].mean()
            diff = l_avg - c_avg
            print(f"{m:<20} | {l_avg:<12.4f} | {c_avg:<12.4f} | {diff:<8.4f}")
        print("="*60)
        print("详细对比已保存至: evaluator_comparison_20.csv\n")

async def main():
    # 构造 20 个具有代表性的医疗 RAG 测试用例
    test_cases = [
        {"question": "高血压患者能不能吃咸菜？", "contexts": ["高血压患者需限制钠盐摄入，建议每日食盐量不超过5g。咸菜含有极高的钠盐。"], "answer": "不可以吃，咸菜盐分太高会升高血压。", "ground_truth": "不能吃。咸菜含钠盐过高，高血压患者应限盐。"},
        {"question": "布洛芬和对乙酰氨基酚能一起吃吗？", "contexts": ["布洛芬和对乙酰氨基酚同属解热镇痛药，合用会增加肾毒性和胃肠道风险。"], "answer": "可以一起吃，效果更好。", "ground_truth": "不能。合用会增加副作用风险。"},
        {"question": "感冒了流黄鼻涕是病毒还是细菌？", "contexts": ["病毒感染初期多为清涕，继发细菌感染或炎症后期可能出现黄脓涕。"], "answer": "流黄鼻涕通常提示可能存在细菌感染。", "ground_truth": "可能提示继发细菌感染，但也可能是病毒感染后期表现。"},
        {"question": "备孕期间能不能照 X 光？", "contexts": ["X射线具有电离辐射，可能影响生殖细胞。建议照X光后3个月再怀孕。"], "answer": "可以照，没影响。", "ground_truth": "不建议。X射线辐射可能影响胚胎，建议间隔3个月。"},
        {"question": "孩子发烧 38.5 度怎么办？", "contexts": ["小儿发热超过38.5℃建议使用退烧药（如美林），并配合物理降温。"], "answer": "给孩子吃退烧药，并多喝水。超过38.5度需要药物干预。", "ground_truth": "建议服用退烧药并物理降温。"},
        {"question": "体检发现胆囊息肉 0.5cm 严重吗？", "contexts": ["小于1cm的胆囊息肉多为良性，建议每半年复查彩超。若大于1cm或增长快需手术。"], "answer": "0.5cm 比较小，通常不严重，定期复查即可。", "ground_truth": "通常不严重，建议定期随访观察。"},
        {"question": "抗生素吃三天感觉好了能停吗？", "contexts": ["抗生素需足疗程使用，擅自停药易导致细菌耐药和病情反复。"], "answer": "感觉好了就可以停，少吃药对身体好。", "ground_truth": "不能停。必须遵医嘱完成整个疗程。"},
        {"question": "脂肪肝怎么调理？", "contexts": ["脂肪肝调理核心是“管住嘴、迈开腿”，禁酒、减重、低脂饮食。"], "answer": "多运动，少吃油腻，戒酒，控制体重。", "ground_truth": "通过控制饮食、加强运动和减重来改善。"},
        {"question": "腰间盘突出能睡软床吗？", "contexts": ["腰椎间盘突出患者建议睡硬板床，以维持腰椎生理曲度。"], "answer": "软床舒服，可以睡软床。", "ground_truth": "不建议。应睡硬板床以保护腰椎。"},
        {"question": "贫血吃什么补得快？", "contexts": ["缺铁性贫血建议食用动物肝脏、血豆腐、瘦肉等富含血红素铁的食物。"], "answer": "多吃红枣和赤豆，这些补血最快。", "ground_truth": "应多吃动物性食品如肝脏、血制品。"},
        {"question": "过敏性哮喘能治愈吗？", "contexts": ["哮喘目前无法根治，但通过规范化治疗可以实现长期临床控制。"], "answer": "可以完全治愈，再也不复发。", "ground_truth": "无法根治，但可以达到临床控制。"},
        {"question": "尿酸高就是痛风吗？", "contexts": ["高尿酸血症是痛风的病理基础，但仅有约10%-20%的高尿酸患者会发展为痛风。"], "answer": "尿酸高就代表你已经得痛风了。", "ground_truth": "不一定。高尿酸是痛风的前兆，但不等同于痛风。"},
        {"question": "胃溃疡能不能喝咖啡？", "contexts": ["咖啡因会刺激胃酸分泌，加重胃溃疡症状，建议急性期禁饮。"], "answer": "少喝一点没关系。", "ground_truth": "不建议喝。会刺激胃酸分泌加重病情。"},
        {"question": "中暑了喝藿香正气水有用吗？", "contexts": ["藿香正气水含有乙醇，不适用于脱水型中暑。主要用于暑湿感冒。"], "answer": "非常有效果，是中暑首选药。", "ground_truth": "需对症。对于酒精敏感或脱水性中暑不宜使用。"},
        {"question": "近视手术后会反弹吗？", "contexts": ["近视手术是切削角膜，本身不反弹，但若不注意用眼习惯可能产生新近视。"], "answer": "手术做完就一劳永逸，绝对不反弹。", "ground_truth": "手术本身不反弹，但需注意用眼卫生防止新近视。"},
        {"question": "心脏早搏一定要吃药吗？", "contexts": ["偶发早搏且无症状者通常无需治疗；频发或有明显症状者需用药。"], "answer": "早搏必须吃药，否则有生命危险。", "ground_truth": "视情况而定。无症状偶发者常不需服药。"},
        {"question": "甲减需要终身服药吗？", "contexts": ["大多数原发性甲减患者需要终身服用左甲状腺素钠替代治疗。"], "answer": "看心情，指标好了就能停。", "ground_truth": "大多数情况下需要终身替代治疗。"},
        {"question": "带状疱疹会传染吗？", "contexts": ["带状疱疹本身不传染，但水疱液含病毒，可能导致未患过水痘的人感染水痘。"], "answer": "不会传染，放心吧。", "ground_truth": "不直接传染带状疱疹，但可能传播水痘病毒。"},
        {"question": "长期吃降压药伤肾吗？", "contexts": ["高血压本身才伤肾。规范使用降压药反而能保护肾脏，减少并发症。"], "answer": "是药三分毒，降压药肯定伤肾。", "ground_truth": "不。规范降压能保护肾功能，高血压本身更伤肾。"},
        {"question": "抽烟对伤口愈合有影响吗？", "contexts": ["香烟中的尼古丁会导致血管收缩，减少组织血供，延缓伤口愈合。"], "answer": "没影响，少抽两根就行。", "ground_truth": "有影响。会延缓愈合速度，增加感染风险。"}
    ]
    
    evaluator = ComparisonEvaluator()
    await evaluator.run_comparison(test_cases)

if __name__ == "__main__":
    asyncio.run(main())
